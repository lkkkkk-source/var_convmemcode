import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.helpers import DropPath, drop_path
from typing import Optional, List, Tuple


# this file only provides the 3 blocks used in VAR transformer
__all__ = ['FFN', 'AdaLNSelfAttn', 'AdaLNBeforeHead']


# automatically import fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError: pass
# automatically import faster attention implementations
try: from xformers.ops import memory_efficient_attention
except ImportError: pass
try: from flash_attn import flash_attn_func              # qkv: BLHc, ret: BLHcq
except ImportError: pass
try: from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc
except ImportError:
    def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
        attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
        if attn_mask is not None: attn.add_(attn_mask)
        return (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., fused_if_available=True):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()

    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(self.fused_mlp_func(
                x=x, weight1=self.fc1.weight, weight2=self.fc2.weight, bias1=self.fc1.bias, bias2=self.fc2.bias,
                activation='gelu_approx', save_pre_act=self.training, return_residual=False, checkpoint_lvl=0,
                heuristic=0, process_group=None,
            ))
        else:
            return self.drop(self.fc2( self.act(self.fc1(x)) ))

    def extra_repr(self) -> str:
        return f'fused_mlp_func={self.fused_mlp_func is not None}'


# Per-scale kernel subset: coarse scales use large kernels, fine scales use small kernels
_SCALE_KERNEL_MAP = {}  # populated lazily
def _get_active_kernels(pn: int, all_scales: List[int]) -> List[int]:
    """Select kernel subset based on spatial resolution pn."""
    if pn <= 5:
        return [k for k in all_scales if k >= 7]    # coarse: large kernels only
    elif pn <= 10:
        return [k for k in all_scales if k >= 5]    # mid: medium + large
    else:
        return [k for k in all_scales if k <= 7]    # fine: small + medium


class SelfAttention(nn.Module):
    def __init__(
        self, block_idx, embed_dim=768, num_heads=12, depth=16,
        attn_drop=0., proj_drop=0., attn_l2_norm=False, flash_if_available=True,
        # Axial texture enhancement parameters
        enable_texture=False,
        texture_scales=[3, 5, 7, 11],
        texture_per_head_kernels=False,
        # Knitting memory parameters
        enable_memory=False,
        memory_num_patterns=16,
        memory_size=8,
        memory_num_scales=10,
        # Class-aware memory parameters
        use_class_aware_memory=False,
        num_categories=22,
        cat_rank=4,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, embed_dim // num_heads
        self.embed_dim = embed_dim
        self.depth = depth
        self.attn_l2_norm = attn_l2_norm

        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)

        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        self.attn_drop: float = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = flash_if_available and memory_efficient_attention is not None

        self.caching, self.cached_k, self.cached_v = False, None, None

        # ========== Axial + 2D texture enhancement branch ==========
        self.enable_texture = enable_texture
        if enable_texture:
            self.texture_scales = texture_scales
            self.texture_per_head_kernels = texture_per_head_kernels

            padding_mode = 'circular'  # V2: circular for periodic knitting patterns

            if texture_per_head_kernels:
                conv_channels = num_heads * self.head_dim
            else:
                conv_channels = self.head_dim

            self._dilations_per_scale = {
                3: [1, 2, 3],
                5: [1, 2, 3, 4],
                7: [1, 2, 3, 4],
                11: [1, 2, 3, 4],
            }
            self._dilations = [1, 2, 3, 4]
            self.row_ops = nn.ModuleDict()
            self.col_ops = nn.ModuleDict()
            self.diag_ops = nn.ModuleDict()  # V2: 2D depthwise conv branch

            def _make_dw_pw(channels, k_h, k_w, dil_h, dil_w, pad_mode=padding_mode):
                pad_h = (k_h // 2) * dil_h
                pad_w = (k_w // 2) * dil_w
                return nn.Sequential(
                    nn.Conv2d(
                        channels, channels,
                        kernel_size=(k_h, k_w),
                        dilation=(dil_h, dil_w),
                        padding=(pad_h, pad_w),
                        padding_mode=pad_mode,
                        groups=channels,
                        bias=False
                    ),
                    nn.Conv2d(channels, channels, kernel_size=1, bias=True)
                )

            for k in texture_scales:
                valid_dilations = self._dilations_per_scale.get(k, [1, 2, 3, 4])
                for r in valid_dilations:
                    key = f'k{k}_r{r}'
                    eff_k = max(3, (k // r) | 1)
                    self.row_ops[key] = _make_dw_pw(conv_channels, 1, eff_k, 1, r)
                    self.col_ops[key] = _make_dw_pw(conv_channels, eff_k, 1, r, 1)
                    # V2: 2D depthwise conv (diagonal/full 2D structure)
                    self.diag_ops[key] = _make_dw_pw(conv_channels, eff_k, eff_k, r, r)

            if block_idx == (depth // 2):  # Print for first enabled layer
                print(f"  [Texture Operators V2] Created {len(self.row_ops)} row/col/diag operator triplets:")
                print(f"    Keys: {sorted(self.row_ops.keys())}")
                print(f"    padding_mode=circular")

            # Gates: row, col, diag
            self.row_gates = nn.Linear(self.head_dim, len(texture_scales))
            self.col_gates = nn.Linear(self.head_dim, len(texture_scales))
            self.diag_gates = nn.Linear(self.head_dim, len(texture_scales))

            # Progressive gate initialization
            if depth > 1:
                init_logit = -2.5 + (block_idx / (depth - 1)) * 2.0
            else:
                init_logit = -2.0

            self.texture_gate_logit = nn.Parameter(torch.full((1, num_heads, 1, 1), init_logit))

            for m in [self.row_gates, self.col_gates, self.diag_gates]:
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

            # V2: texture modulates K/V (and Q with small coeff) via low-rank projections
            tex_rank = 64
            self.Wk_tex = nn.Sequential(nn.Linear(embed_dim, tex_rank, bias=False), nn.Linear(tex_rank, embed_dim, bias=False))
            self.Wv_tex = nn.Sequential(nn.Linear(embed_dim, tex_rank, bias=False), nn.Linear(tex_rank, embed_dim, bias=False))
            self.Wq_tex = nn.Sequential(nn.Linear(embed_dim, tex_rank, bias=False), nn.Linear(tex_rank, embed_dim, bias=False))
            self.tex_q_coeff = 0.3  # Q gets smaller texture modulation

            print(f"  [Texture Layer {block_idx}] init_gate_logit={init_logit:.3f} "
                  f"(sigmoid={torch.sigmoid(torch.tensor(init_logit)).item():.3f}), "
                  f"mode=kv_modulation+circular+2d_branch")

        # ========== Knitting Pattern Memory ==========
        self.enable_memory = enable_memory
        self.use_class_aware_memory = use_class_aware_memory
        if enable_memory:
            if use_class_aware_memory:
                from models.class_aware_memory import ClassAwareKnittingMemoryV2
                self.knitting_memory = ClassAwareKnittingMemoryV2(
                    embed_dim=embed_dim,
                    num_categories=num_categories,
                    num_scales=memory_num_scales,
                    shared_patterns=memory_num_patterns,
                    shared_memory_size=memory_size,
                    cat_rank=cat_rank,
                    block_idx=block_idx,
                    depth=depth,
                )
            else:
                from models.knitting_memory import KnittingPatternMemory
                self.knitting_memory = KnittingPatternMemory(
                    embed_dim=embed_dim,
                    num_patterns=memory_num_patterns,
                    memory_size=memory_size,
                    num_scales=memory_num_scales,
                    block_idx=block_idx,
                    depth=depth,
                )

        # Pre-computed texture execution plans (lazily built per begin_ends layout).
        # Training uses one full-token layout, while AR inference uses a different
        # local layout at each scale; caching only one plan would let the initial
        # 1x1 inference stage disable texture for all later stages.
        self._texture_plan_cache = {} if enable_texture else None
        self._texture_plan = None if enable_texture else []
        self._last_texture_gate_stats = {}

    def kv_caching(self, enable: bool):
        self.caching, self.cached_k, self.cached_v = enable, None, None

    def _select_op_key(self, k: int, pn: int) -> str:
        if k <= pn:
            r = 1
        else:
            r = min(self._dilations[-1], math.ceil(k / pn))
        return f'k{k}_r{r}'

    def _safe_apply_conv_seq(self, op, x):
        """
        Safely apply conv sequence, checking if circular padding is valid.
        Returns None if padding would wrap more than once.
        """
        H, W = x.shape[-2:]

        for m in op.modules():
            if isinstance(m, nn.Conv2d) and m.padding_mode == 'circular':
                if isinstance(m.padding, tuple):
                    pad_h, pad_w = m.padding
                else:
                    pad_h = pad_w = m.padding

                if H <= pad_h or W <= pad_w:
                    return None

        return op(x)

    def _validate_padding(self, op, pn: int) -> bool:
        """Check if circular padding is valid for spatial size pn. Called once at plan build time."""
        for m in op.modules():
            if isinstance(m, nn.Conv2d) and m.padding_mode == 'circular':
                pad_h, pad_w = m.padding if isinstance(m.padding, tuple) else (m.padding, m.padding)
                if pn <= pad_h or pn <= pad_w:
                    return False
        return True

    def _build_texture_plan(self, begin_ends):
        """Build static execution plan on first forward. Maps each scale to pre-validated ops."""
        texture_plan = []
        for start_idx, end_idx in begin_ends:
            scale_len = end_idx - start_idx
            pn = int(math.sqrt(scale_len))
            if pn * pn != scale_len or pn < 2:
                texture_plan.append(None)
                continue

            active_kernels = _get_active_kernels(pn, self.texture_scales)
            if not active_kernels:
                texture_plan.append(None)
                continue

            row_plan, col_plan, diag_plan = [], [], []
            for idx, k in enumerate(self.texture_scales):
                if k not in active_kernels:
                    continue
                op_key = self._select_op_key(k, pn)
                if op_key in self.row_ops and self._validate_padding(self.row_ops[op_key], pn):
                    row_plan.append((self.row_ops[op_key], idx))
                if op_key in self.col_ops and self._validate_padding(self.col_ops[op_key], pn):
                    col_plan.append((self.col_ops[op_key], idx))
                if op_key in self.diag_ops and self._validate_padding(self.diag_ops[op_key], pn):
                    diag_plan.append((self.diag_ops[op_key], idx))

            if not (row_plan or col_plan or diag_plan):
                texture_plan.append(None)
                continue
            texture_plan.append((pn, start_idx, end_idx, row_plan, col_plan, diag_plan))

        self._texture_plan = texture_plan
        return texture_plan

    def _compute_texture_modulation(
        self,
        x_BLC: torch.Tensor,
        begin_ends: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        Compute texture features from input, returning [B, L, C] modulation.
        Uses row + col + diag branches with per-scale kernel selection.
        """
        B, L, C = x_BLC.shape
        H, c = self.num_heads, self.head_dim

        # Reshape to [B, H, L, c] for per-head processing
        x_heads = x_BLC.view(B, L, H, c).permute(0, 2, 1, 3)  # [B, H, L, c]

        tex_output = torch.zeros(B, H, L, c, device=x_BLC.device, dtype=x_BLC.dtype)

        # Build/reuse execution plan for this exact layout. Full teacher-forcing
        # training and step-wise AR inference have different begin_ends layouts.
        plan_key = tuple((int(s), int(e)) for s, e in begin_ends)
        if self._texture_plan_cache is None:
            texture_plan = self._texture_plan
        elif plan_key in self._texture_plan_cache:
            texture_plan = self._texture_plan_cache[plan_key]
            self._texture_plan = texture_plan
        else:
            texture_plan = self._build_texture_plan(begin_ends)
            self._texture_plan_cache[plan_key] = texture_plan

        for plan_entry in texture_plan:
            if plan_entry is None:
                continue
            pn, start_idx, end_idx, row_plan, col_plan, diag_plan = plan_entry

            if end_idx > L:
                continue

            scale_tokens = x_heads[:, :, start_idx:end_idx, :]  # [B, H, pn*pn, c]

            if self.texture_per_head_kernels:
                scale_2d = scale_tokens.reshape(B, H, pn, pn, c)
                scale_2d = scale_2d.permute(0, 1, 4, 2, 3).reshape(B, H * c, pn, pn)
            else:
                scale_2d = scale_tokens.reshape(B, H, pn, pn, c)
                scale_2d = scale_2d.permute(0, 1, 4, 2, 3).reshape(B * H, c, pn, pn)

            # Execute pre-validated ops directly (no runtime padding checks)
            row_features = [op(scale_2d)[:, :, :pn, :pn] for op, _ in row_plan]
            row_indices = [idx for _, idx in row_plan]
            col_features = [op(scale_2d)[:, :, :pn, :pn] for op, _ in col_plan]
            col_indices = [idx for _, idx in col_plan]
            diag_features = [op(scale_2d)[:, :, :pn, :pn] for op, _ in diag_plan]
            diag_indices = [idx for _, idx in diag_plan]

            if not (row_features or col_features or diag_features):
                continue

            # Compute global feature for gating
            if self.texture_per_head_kernels:
                global_feat = scale_2d.reshape(B, H, c, pn, pn).mean(dim=(3, 4)).reshape(B * H, c)
            else:
                global_feat = scale_2d.mean(dim=(2, 3))

            # Gate computation for each branch
            def _weighted_mix(features, valid_indices, gate_fn):
                if len(features) == 0:
                    return torch.zeros_like(scale_2d)
                logits = gate_fn(global_feat)[:, valid_indices]
                weights = torch.sigmoid(logits)
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
                weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                stacked = torch.stack(features, dim=1)
                return (stacked * weights).sum(dim=1)

            row_mix = _weighted_mix(row_features, row_indices, self.row_gates)
            col_mix = _weighted_mix(col_features, col_indices, self.col_gates)
            diag_mix = _weighted_mix(diag_features, diag_indices, self.diag_gates)

            # Combine: row + col + diag
            enhanced_2d = row_mix + col_mix + diag_mix

            if self.texture_per_head_kernels:
                enhanced_2d = enhanced_2d.reshape(B, H, c, pn, pn)
            else:
                enhanced_2d = enhanced_2d.reshape(B, H, c, pn, pn)

            enhanced_scale = enhanced_2d.permute(0, 1, 3, 4, 2).reshape(B, H, pn * pn, c)
            tex_output[:, :, start_idx:end_idx, :] = enhanced_scale

        # Reshape back to [B, L, C]
        return tex_output.permute(0, 2, 1, 3).reshape(B, L, C)

    def forward(self, x, attn_bias, begin_ends=None, category_ids=None):
        B, L, C = x.shape

        # VAR consistency check (during training)
        if self.enable_texture and begin_ends is not None and self.training:
            total = sum(e - s for s, e in begin_ends)
            assert total == L, f"begin_ends must cover all tokens: total={total}, L={L}"

        # ========== QKV projection (before memory/texture modulation) ==========
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)
        main_type = qkv.dtype

        using_flash = self.using_flash and attn_bias is None and qkv.dtype != torch.float32
        if using_flash or self.using_xform:
            q, k, v = qkv.unbind(dim=2); dim_cat = 1
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); dim_cat = 2

        # ========== Memory K/V modulation (V2: modulates K/V, not x) ==========
        if self.enable_memory and begin_ends is not None and len(begin_ends) == self.knitting_memory.num_scales:
            if self.use_class_aware_memory:
                mem_k, mem_v, _div_loss, _sep_loss = self.knitting_memory(x, begin_ends, category_ids)
            else:
                mem_k, mem_v, _div_loss, _sep_loss = self.knitting_memory(x, begin_ends)

            # Reshape mem_k/mem_v from [B, L, C] to match k/v shape
            if using_flash or self.using_xform:
                # k shape: [B, L, H, c]
                mem_k_shaped = mem_k.view(B, L, self.num_heads, self.head_dim)
                mem_v_shaped = mem_v.view(B, L, self.num_heads, self.head_dim)
            else:
                # k shape: [B, H, L, c]
                mem_k_shaped = mem_k.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                mem_v_shaped = mem_v.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            k = k + mem_k_shaped
            v = v + mem_v_shaped

        # ========== Texture K/V modulation (V2: modulates K/V via projections) ==========
        if self.enable_texture and begin_ends is not None:
            tex_feat = self._compute_texture_modulation(x, begin_ends)  # [B, L, C]
            texture_weight = torch.sigmoid(self.texture_gate_logit)  # [1, H, 1, 1]

            # Project texture features to K/V/Q space
            tex_k = self.Wk_tex(tex_feat)  # [B, L, C]
            tex_v = self.Wv_tex(tex_feat)
            tex_q = self.Wq_tex(tex_feat)

            if using_flash or self.using_xform:
                # shapes: [B, L, H, c]
                tex_k_shaped = tex_k.view(B, L, self.num_heads, self.head_dim)
                tex_v_shaped = tex_v.view(B, L, self.num_heads, self.head_dim)
                tex_q_shaped = tex_q.view(B, L, self.num_heads, self.head_dim)
                tw = texture_weight.permute(0, 2, 1, 3)  # [1, 1, H, 1]
                k = k + tw * tex_k_shaped
                v = v + tw * tex_v_shaped
                q = q + self.tex_q_coeff * tw * tex_q_shaped
            else:
                # shapes: [B, H, L, c]
                tex_k_shaped = tex_k.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                tex_v_shaped = tex_v.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                tex_q_shaped = tex_q.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                k = k + texture_weight * tex_k_shaped
                v = v + texture_weight * tex_v_shaped
                q = q + self.tex_q_coeff * texture_weight * tex_q_shaped

        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash or self.using_xform: scale_mul = scale_mul.transpose(1, 2)
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)

        if self.caching:
            if self.cached_k is None:
                self.cached_k = k; self.cached_v = v
            else:
                k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)

        dropout_p = self.attn_drop if self.training else 0.0
        if using_flash:
            oup = flash_attn_func(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), dropout_p=dropout_p, softmax_scale=self.scale).view(B, L, C)
        elif self.using_xform:
            oup = memory_efficient_attention(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), attn_bias=None if attn_bias is None else attn_bias.to(dtype=main_type).expand(B, self.num_heads, -1, -1), p=dropout_p, scale=self.scale).view(B, L, C)
        else:
            oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias, dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)

        return self.proj_drop(self.proj(oup))

    def get_texture_gate_stats(self) -> dict:
        """Return compact per-layer texture gate summaries for trainer logging."""
        if not self.enable_texture:
            return {
                'enabled': False,
                'gate_mean': 0.0,
                'gate_std': 0.0,
                'gate_min': 0.0,
                'gate_max': 0.0,
                'head_flatline_ratio': 1.0,
            }

        with torch.no_grad():
            gate_vals = torch.sigmoid(self.texture_gate_logit.detach()).view(-1)
            gate_mean = gate_vals.mean().item()
            gate_std = gate_vals.std(unbiased=False).item()
            gate_min = gate_vals.min().item()
            gate_max = gate_vals.max().item()
            head_flatline_ratio = (gate_vals < 0.02).float().mean().item()

        self._last_texture_gate_stats = {
            'enabled': True,
            'gate_mean': gate_mean,
            'gate_std': gate_std,
            'gate_min': gate_min,
            'gate_max': gate_max,
            'head_flatline_ratio': head_flatline_ratio,
        }
        return self._last_texture_gate_stats

    def extra_repr(self) -> str:
        gate_val = torch.sigmoid(self.texture_gate_logit).mean().item() if self.enable_texture else 0.0
        return (f'enable_texture={self.enable_texture}, '
                f'gate_init={gate_val:.3f}, '
                f'per_head_kernels={getattr(self, "texture_per_head_kernels", False)}, '
                f'enable_memory={self.enable_memory}')


class AdaLNSelfAttn(nn.Module):
    def __init__(
        self, block_idx, last_drop_p, embed_dim, cond_dim, shared_aln: bool, norm_layer,
        num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., attn_l2_norm=False,
        flash_if_available=False, fused_if_available=True,
        begin_ends: Optional[List[Tuple[int, int]]] = None,
        # Axial texture parameters
        enable_texture: bool = False,
        texture_scales: List[int] = [3, 5, 7, 11],
        texture_per_head_kernels: bool = False,
        depth: int = 16,
        # Knitting memory parameters
        enable_memory: bool = False,
        memory_num_patterns: int = 16,
        memory_size: int = 8,
        memory_num_scales: int = 10,
        # Class-aware memory parameters
        use_class_aware_memory: bool = False,
        num_categories: int = 22,
        cat_rank: int = 4,
    ):
        super(AdaLNSelfAttn, self).__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Axial texture-aware attention with knitting memory
        self.attn = SelfAttention(
            block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads,
            depth=depth,
            attn_drop=attn_drop, proj_drop=drop, attn_l2_norm=attn_l2_norm,
            flash_if_available=flash_if_available,
            enable_texture=enable_texture,
            texture_scales=texture_scales,
            texture_per_head_kernels=texture_per_head_kernels,
            enable_memory=enable_memory,
            memory_num_patterns=memory_num_patterns,
            memory_size=memory_size,
            memory_num_scales=memory_num_scales,
            use_class_aware_memory=use_class_aware_memory,
            num_categories=num_categories,
            cat_rank=cat_rank,
        )

        self.ffn = FFN(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio),
                      drop=drop, fused_if_available=fused_if_available)

        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            lin = nn.Linear(cond_dim, 6*embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)

        self.fused_add_norm_fn = None
        self.begin_ends = begin_ends

    def forward(self, x, cond_BD, attn_bias, begin_ends=None, category_ids=None, prog_si: int = -1):
        if self.shared_aln:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2)
        else:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)

        # Use passed begin_ends if provided, otherwise fall back to self.begin_ends
        be = begin_ends if begin_ends is not None else self.begin_ends

        # Attention with axial texture enhancement and knitting memory
        x = x + self.drop_path(self.attn(
            self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1),
            attn_bias=attn_bias,
            begin_ends=be,
            category_ids=category_ids
        ).mul_(gamma1))

        # FFN
        x = x + self.drop_path(self.ffn(
            self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)
        ).mul(gamma2))

        return x

    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}, enable_texture={self.attn.enable_texture}'


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, norm_layer):   # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2*C))

    def forward(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)
