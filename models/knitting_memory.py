"""
Knitting Pattern Memory V2 for VAR (non-class-aware)
针织花型记忆库 V2

V2 改进:
1. 输出 K/V 调制向量, 不再直接加到 x
2. 新增 slot separation loss
3. 接口与 ClassAwareKnittingMemoryV2 一致
4. 保留分尺度记忆 + 严格因果性 + 正交初始化
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class KnittingPatternMemory(nn.Module):
    """
    针织花型记忆库 (非类别感知版本)

    输出: (mem_k, mem_v, diversity_loss, slot_sep_loss)
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_patterns: int = 16,
        memory_size: int = 8,
        num_scales: int = 10,
        block_idx: int = 0,
        depth: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patterns = num_patterns
        self.memory_size = memory_size
        self.slots_per_scale = num_patterns * memory_size
        self.num_scales = num_scales
        self.block_idx = block_idx

        # 1. 分尺度记忆槽
        self.memory_per_scale = nn.ParameterDict({
            f'scale_{i}': nn.Parameter(torch.randn(num_patterns, memory_size, embed_dim))
            for i in range(num_scales)
        })
        for mem in self.memory_per_scale.values():
            self._init_orthogonal(mem)

        # 2. Query/Key/Value 投影
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # 3. K/V 输出投影 + 门控 (低秩)
        mem_rank = 64
        self.Wk_mem = nn.Sequential(nn.Linear(embed_dim, mem_rank), nn.Linear(mem_rank, embed_dim))
        self.Wv_mem = nn.Sequential(nn.Linear(embed_dim, mem_rank), nn.Linear(mem_rank, embed_dim))

        if depth > 1:
            init_logit = -2.0 + (block_idx / (depth - 1)) * 2.0
        else:
            init_logit = -2.0
        self.gk_logit = nn.Parameter(torch.tensor(init_logit))
        self.gv_logit = nn.Parameter(torch.tensor(init_logit))

        # 4. 温度参数
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.1)))
        self.override_temperature = None

        # 5. 因果可见性 mask
        self.register_buffer('slot_visibility', self._build_slot_visibility(num_scales))

        # 6. 监控
        self.register_buffer('slot_usage_ema', torch.zeros(num_scales, self.slots_per_scale))
        self.ema_momentum = 0.99
        self.last_diversity_loss = torch.tensor(0.0)
        self.last_slot_sep_loss = torch.tensor(0.0)
        self._last_usage_concentration = 0.0
        self._last_dead_slot_ratio = 0.0
        self._last_dead_slot_count = 0
        self._last_total_visible_slots = 0
        self._last_flatline_flag = False
        self._dead_slot_eps = 1e-4
        self._flatline_loss_eps = 1e-8

        # Slot sep loss cache
        self._slot_sep_cache = None
        self._forward_count = 0
        self._slot_sep_interval = 10

        # Pre-computed visible indices
        self._visible_indices_list = [list(range(i + 1)) for i in range(num_scales)]

        # Compat: kept for checkpoint loading
        self.register_buffer('residual_scale', torch.tensor(1.0))

        print(f"  [KnittingMemoryV2 Layer {block_idx}] "
              f"patterns={num_patterns}, slots_per_scale={self.slots_per_scale}, "
              f"num_scales={num_scales}, "
              f"gk_init={torch.sigmoid(torch.tensor(init_logit)).item():.3f}")

    def _init_orthogonal(self, memory: torch.Tensor):
        with torch.no_grad():
            flat = memory.view(-1, self.embed_dim)
            nn.init.orthogonal_(flat)
            xavier_std = math.sqrt(2.0 / (self.embed_dim + self.embed_dim))
            current_std = flat.std()
            if current_std > 0:
                flat.mul_(xavier_std / current_std)
            memory.copy_(flat.view_as(memory))

    def _build_slot_visibility(self, num_scales: int) -> torch.Tensor:
        visibility = torch.zeros(num_scales, num_scales, dtype=torch.bool)
        for i in range(num_scales):
            visibility[i, :i + 1] = True
        return visibility

    def get_current_temperature(self) -> float:
        if self.override_temperature is not None:
            return self.override_temperature
        return torch.exp(self.log_temperature).clamp(0.05, 1.0)

    def freeze_learnable_temperature(self):
        self.log_temperature.requires_grad_(False)
        print(f"  [Layer {self.block_idx}] Learnable temperature frozen at {torch.exp(self.log_temperature).item():.4f}")

    def forward(
        self,
        x: torch.Tensor,
        begin_ends: List[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            mem_k: [B, L, C] K 调制向量
            mem_v: [B, L, C] V 调制向量
            diversity_loss: 标量
            slot_sep_loss: 标量
        """
        B, L, C = x.shape
        num_scales = len(begin_ends)
        device = x.device
        assert num_scales == self.num_scales

        query = self.query_proj(x)

        # Batch K/V projection: single call instead of N separate calls
        mem_stacked = torch.stack([
            self.memory_per_scale[f'scale_{i}'].view(-1, C) for i in range(num_scales)
        ])  # [num_scales, slots, C]
        all_keys = self.key_proj(mem_stacked)    # [num_scales, slots, C]
        all_values = self.value_proj(mem_stacked)  # [num_scales, slots, C]

        mem_combined = torch.zeros_like(x)
        all_attn_weights = []
        temp = self.get_current_temperature()

        for i in range(num_scales):
            start, end = begin_ends[i]
            scale_len = end - start
            if scale_len <= 0:
                continue

            q_scale = query[:, start:end, :]
            # Pre-computed visible indices (causal: scale i sees [0..i])
            k_visible = all_keys[:i+1].reshape(-1, C)    # [(i+1)*slots, C]
            v_visible = all_values[:i+1].reshape(-1, C)   # [(i+1)*slots, C]

            scale_factor = 1.0 / math.sqrt(C)
            scores = torch.matmul(q_scale, k_visible.T) * scale_factor
            attn = F.softmax(scores / temp, dim=-1)
            retrieved = torch.matmul(attn, v_visible)

            mem_combined[:, start:end, :] = retrieved
            all_attn_weights.append((i, attn))

        # K/V output with gates
        gk = torch.sigmoid(self.gk_logit)
        gv = torch.sigmoid(self.gv_logit)
        mem_k = gk * self.Wk_mem(mem_combined)
        mem_v = gv * self.Wv_mem(mem_combined)

        # Diversity loss
        if self.training and len(all_attn_weights) > 0:
            diversity_loss = self._compute_diversity_loss(all_attn_weights)
        else:
            diversity_loss = torch.tensor(0.0, device=device)

        # Slot separation loss (cached)
        if self.training:
            self._forward_count += 1
            if self._slot_sep_cache is None or self._forward_count % self._slot_sep_interval == 0:
                slot_sep_loss = self._compute_slot_sep_loss()
                self._slot_sep_cache = slot_sep_loss
            else:
                slot_sep_loss = self._slot_sep_cache.detach()
        else:
            slot_sep_loss = torch.tensor(0.0, device=device)

        # EMA update
        if self.training:
            self._update_slot_usage_ema(all_attn_weights)

        usage_concentration, dead_slot_ratio, dead_slot_count, total_visible_slots = self._compute_usage_health(all_attn_weights)
        self._last_usage_concentration = usage_concentration
        self._last_dead_slot_ratio = dead_slot_ratio
        self._last_dead_slot_count = dead_slot_count
        self._last_total_visible_slots = total_visible_slots
        div_val = diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else float(diversity_loss)
        sep_val = slot_sep_loss.item() if isinstance(slot_sep_loss, torch.Tensor) else float(slot_sep_loss)
        self._last_flatline_flag = (div_val <= self._flatline_loss_eps and sep_val <= self._flatline_loss_eps)

        self.last_diversity_loss = diversity_loss
        self.last_slot_sep_loss = slot_sep_loss

        return mem_k, mem_v, diversity_loss, slot_sep_loss

    def _compute_diversity_loss(self, attn_weights_list: list) -> torch.Tensor:
        total_loss = 0.0
        K = 6

        max_collapse_ratio = 0.0
        avg_num_slots = 0.0

        for _scale_idx, attn in attn_weights_list:
            num_slots_eff = attn.shape[-1]
            slot_usage = attn.mean(dim=(0, 1))
            max_usage = slot_usage.max()

            collapse_ratio = max_usage * num_slots_eff
            max_collapse_ratio = max(
                max_collapse_ratio,
                collapse_ratio.item() if isinstance(collapse_ratio, torch.Tensor) else collapse_ratio
            )
            avg_num_slots += num_slots_eff

            collapse_threshold = K / num_slots_eff
            total_loss = total_loss + F.relu(max_usage - collapse_threshold)

        self._last_collapse_ratio = max_collapse_ratio
        self._last_avg_num_slots = avg_num_slots / max(len(attn_weights_list), 1)

        return total_loss / max(len(attn_weights_list), 1)

    def _compute_slot_sep_loss(self) -> torch.Tensor:
        all_slots = []
        for i in range(self.num_scales):
            mem = self.memory_per_scale[f'scale_{i}'].view(-1, self.embed_dim)
            all_slots.append(mem)
        all_slots = torch.cat(all_slots, dim=0)
        slots_normed = F.normalize(all_slots, dim=-1)
        sim_matrix = slots_normed @ slots_normed.T
        N = sim_matrix.shape[0]
        mask = ~torch.eye(N, dtype=torch.bool, device=sim_matrix.device)
        off_diag = sim_matrix[mask]
        return (off_diag ** 2).mean()

    def _update_slot_usage_ema(self, attn_weights_list: list):
        if len(attn_weights_list) == 0:
            return
        with torch.no_grad():
            S = self.slots_per_scale
            for scale_idx, attn in attn_weights_list:
                expected_slots = (scale_idx + 1) * S
                if attn.size(-1) != expected_slots:
                    continue
                usage_flat = attn.mean(dim=(0, 1))
                usage_by_scale = usage_flat.view(scale_idx + 1, S)
                self.slot_usage_ema[:scale_idx + 1].mul_(self.ema_momentum).add_(
                    usage_by_scale, alpha=1 - self.ema_momentum
                )

    def _compute_usage_health(self, attn_weights_list: list) -> Tuple[float, float, int, int]:
        """Compute compact liveness diagnostics for trainer summary logging."""
        if len(attn_weights_list) == 0:
            return 0.0, 0.0, 0, 0

        concentration_vals = []
        dead_slots = 0
        total_slots = 0

        for _scale_idx, attn in attn_weights_list:
            slot_usage = attn.mean(dim=(0, 1))
            num_slots = slot_usage.numel()
            if num_slots == 0:
                continue
            concentration_vals.append(float(slot_usage.max().detach().cpu()) * float(num_slots))
            dead_slots += (slot_usage < self._dead_slot_eps).sum().item()
            total_slots += num_slots

        if total_slots == 0:
            return 0.0, 0.0, 0, 0

        usage_concentration = sum(concentration_vals) / max(len(concentration_vals), 1)
        dead_slot_ratio = dead_slots / total_slots
        return usage_concentration, dead_slot_ratio, dead_slots, total_slots

    def get_diagnostics(self) -> dict:
        with torch.no_grad():
            all_slots = []
            for i in range(self.num_scales):
                mem = self.memory_per_scale[f'scale_{i}']
                all_slots.append(mem.view(-1, self.embed_dim))
            all_slots = torch.cat(all_slots, dim=0)
            norms = all_slots.norm(dim=-1)
            slots_norm = F.normalize(all_slots, dim=-1)
            similarity = slots_norm @ slots_norm.T
            mask = ~torch.eye(len(all_slots), dtype=torch.bool, device=similarity.device)
            off_diag_sim = similarity[mask]
            usage = self.slot_usage_ema.flatten().cpu().numpy()
            current_temp = self.get_current_temperature()
            current_temp = float(current_temp)
            gk = float(torch.sigmoid(self.gk_logit))
            gv = float(torch.sigmoid(self.gv_logit))
            flatline = bool(self._last_flatline_flag)

            return {
                'layer': self.block_idx,
                'norm_mean': norms.mean().item(),
                'norm_std': norms.std().item(),
                'similarity_mean': off_diag_sim.mean().item(),
                'similarity_std': off_diag_sim.std().item(),
                'usage_mean': usage.mean(),
                'usage_std': usage.std(),
                'temperature': current_temp,
                'gk_weight': gk,
                'gv_weight': gv,
                'usage_concentration': self._last_usage_concentration,
                'dead_slot_ratio': self._last_dead_slot_ratio,
                'dead_slot_count': self._last_dead_slot_count,
                'total_visible_slots': self._last_total_visible_slots,
                'flatline': flatline,
                }

    def extra_repr(self) -> str:
        gk = torch.sigmoid(self.gk_logit).item()
        gv = torch.sigmoid(self.gv_logit).item()
        return (f'patterns={self.num_patterns}, slots_per_scale={self.slots_per_scale}, '
                f'embed_dim={self.embed_dim}, num_scales={self.num_scales}, '
                f'gk={gk:.3f}, gv={gv:.3f}')


if __name__ == '__main__':
    print("Testing KnittingPatternMemory V2...")

    B, C = 2, 1024
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    begin_ends = []
    cur = 0
    for pn in patch_nums:
        begin_ends.append((cur, cur + pn * pn))
        cur += pn * pn
    L = sum(pn * pn for pn in patch_nums)

    memory = KnittingPatternMemory(
        embed_dim=C, num_patterns=16, memory_size=8,
        num_scales=10, block_idx=0, depth=16,
    )

    x = torch.randn(B, L, C)
    mem_k, mem_v, div_loss, sep_loss = memory(x, begin_ends)

    print(f"mem_k shape: {mem_k.shape}")
    print(f"mem_v shape: {mem_v.shape}")
    print(f"Diversity loss: {div_loss.item():.6f}")
    print(f"Slot sep loss: {sep_loss.item():.6f}")

    print("\n[OK] Test passed!")
