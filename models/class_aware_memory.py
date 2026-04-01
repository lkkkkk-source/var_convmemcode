"""
Class-Aware Knitting Pattern Memory V2 for VAR
类别感知针织花型记忆库 V2

核心改进 (相比V1):
1. 类别记忆改为低秩残差: M_cat_eff = M_shared + A_cat @ B_scale
2. shared/cat 分开 attention, 不再联合 softmax
3. Alpha 融合门: alpha = sigmoid(MLP([mean(q), scale_embed, class_embed]))
4. 输出为 K/V 调制向量, 不再直接加到 x
5. 新增 slot separation loss 防止槽位学成相似
6. 保持分尺度因果可见性
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ClassAwareKnittingMemoryV2(nn.Module):
    """
    类别感知针织花型记忆库 V2

    结构:
    - 共享基础记忆: [num_scales, shared_patterns, shared_memory_size, embed_dim]
    - 类别残差记忆: M_cat_eff = M_shared + A_cat @ B (低秩)

    输出: (mem_k, mem_v, diversity_loss, slot_sep_loss)
    - mem_k/mem_v: [B, L, C] 用于调制 attention 的 K/V
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_categories: int = 22,
        num_scales: int = 10,
        shared_patterns: int = 8,
        shared_memory_size: int = 4,
        cat_rank: int = 8,
        block_idx: int = 0,
        depth: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_categories = num_categories
        self.num_scales = num_scales
        self.shared_patterns = shared_patterns
        self.shared_memory_size = shared_memory_size
        self.cat_rank = cat_rank
        self.block_idx = block_idx
        self.slots_per_scale = shared_patterns * shared_memory_size

        # ==================== 1. 共享基础记忆 ====================
        self.shared_memory = nn.ParameterDict({
            f'scale_{i}': nn.Parameter(
                torch.randn(shared_patterns, shared_memory_size, embed_dim)
            )
            for i in range(num_scales)
        })
        for mem in self.shared_memory.values():
            self._init_orthogonal(mem)

        # ==================== 2. 类别低秩残差记忆 ====================
        # M_cat_eff[cat, scale] = M_shared[scale] + (A_cat[cat, scale] @ B[scale])
        # A_cat: per-category, per-scale low-rank coefficients
        # B: shared low-rank basis projecting to (slots_per_scale * embed_dim)
        self.cat_A = nn.Parameter(
            torch.randn(num_categories, num_scales, cat_rank) * 0.01
        )
        self.cat_B = nn.Parameter(
            torch.randn(num_scales, cat_rank, self.slots_per_scale * embed_dim) * (1.0 / math.sqrt(cat_rank))
        )

        # ==================== 3. 类别嵌入 ====================
        self.category_embedding = nn.Embedding(num_categories, embed_dim)
        nn.init.normal_(self.category_embedding.weight, std=0.02)

        # ==================== 4. 尺度嵌入 ====================
        self.scale_embedding = nn.Embedding(num_scales, embed_dim)
        nn.init.normal_(self.scale_embedding.weight, std=0.02)

        # ==================== 5. Query/Key/Value 投影 (shared for both branches) ====================
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # ==================== 6. Alpha 融合门 (轻量化) ====================
        # alpha = sigmoid(MLP([mean(q), scale_embed, class_embed]))
        self.alpha_mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        # Init to slightly favor shared (output ~0 -> alpha ~0.5, but we bias towards shared)
        nn.init.zeros_(self.alpha_mlp[-1].weight)
        nn.init.constant_(self.alpha_mlp[-1].bias, -1.0)  # sigmoid(-1) ≈ 0.27

        # ==================== 7. K/V 输出投影 + 门控 (低秩) ====================
        mem_rank = 64
        self.Wk_mem = nn.Sequential(nn.Linear(embed_dim, mem_rank), nn.Linear(mem_rank, embed_dim))
        self.Wv_mem = nn.Sequential(nn.Linear(embed_dim, mem_rank), nn.Linear(mem_rank, embed_dim))

        # Per-K/V scalar gates, init small
        if depth > 1:
            init_logit = -2.0 + (block_idx / (depth - 1)) * 2.0
        else:
            init_logit = -2.0
        self.gk_logit = nn.Parameter(torch.tensor(init_logit))
        self.gv_logit = nn.Parameter(torch.tensor(init_logit))

        # ==================== 8. 温度参数 ====================
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.1)))
        self.override_temperature = None

        # ==================== 9. 因果可见性 mask ====================
        self.register_buffer('slot_visibility', self._build_slot_visibility(num_scales))

        # ==================== 10. 监控 ====================
        self.register_buffer('shared_usage_ema', torch.zeros(num_scales, self.slots_per_scale))
        self.ema_momentum = 0.99
        self.last_diversity_loss = torch.tensor(0.0)
        self.last_slot_sep_loss = torch.tensor(0.0)
        self._last_collapse_ratio = 0.0
        self._last_avg_num_slots = 0.0
        
        # Entropy monitoring (matching KnittingPatternMemory)
        self._last_usage_concentration = 0.0
        self._last_dead_slot_ratio = 0.0
        self._last_dead_slot_count = 0
        self._last_total_visible_slots = 0
        self._last_flatline_flag = False
        self._last_entropy_ratio_weighted = 0.0
        self._last_entropy_dispersion = 0.0
        self._last_effective_slot_usage_weighted = 0.0
        self._last_max_attn_weighted = 0.0
        self._last_entropy_ratio_per_scale = {}
        self._dead_slot_eps = 1e-4
        self._flatline_loss_eps = 1e-8

        # Pre-computed visible indices per scale (causal: scale i sees [0..i])
        self._visible_indices_list = [list(range(i + 1)) for i in range(num_scales)]

        # Slot sep loss cache (recompute every N steps to save 320x320 matmul)
        self._slot_sep_cache = None
        self._forward_count = 0
        self._slot_sep_interval = 10

        # Stats
        total_shared_params = num_scales * shared_patterns * shared_memory_size * embed_dim
        total_cat_params = num_categories * num_scales * cat_rank + num_scales * cat_rank * self.slots_per_scale * embed_dim
        print(f"  [ClassAwareMemoryV2 Layer {block_idx}] "
              f"shared_slots={self.slots_per_scale}/scale, "
              f"cat_rank={cat_rank}, "
              f"num_categories={num_categories}, "
              f"shared_params={total_shared_params/1e6:.2f}M, "
              f"cat_params={total_cat_params/1e6:.2f}M, "
              f"gk_init={torch.sigmoid(torch.tensor(init_logit)).item():.3f}")

    def _init_orthogonal(self, memory: torch.Tensor):
        """正交初始化 + Xavier缩放"""
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

    def _get_cat_memory(self, cat_id: int, scale_idx: int) -> torch.Tensor:
        """
        获取某类别某尺度的有效记忆: M_shared + low-rank residual

        Returns: [slots_per_scale, embed_dim]
        """
        shared = self.shared_memory[f'scale_{scale_idx}'].view(self.slots_per_scale, self.embed_dim)
        # A_cat[cat_id, scale_idx]: [cat_rank]
        # cat_B[scale_idx]: [cat_rank, slots_per_scale * embed_dim]
        a = self.cat_A[cat_id, scale_idx]  # [cat_rank]
        b = self.cat_B[scale_idx]  # [cat_rank, slots * C]
        delta = (a @ b).view(self.slots_per_scale, self.embed_dim)  # [slots, C]
        return shared + delta

    def _retrieve(self, q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, temp: float) -> torch.Tensor:
        """
        Standard scaled dot-product retrieval.

        Args:
            q: [B, seq_len, C] or [1, seq_len, C]
            keys: [num_slots, C]
            values: [num_slots, C]
            temp: temperature scalar

        Returns: [B, seq_len, C]
        """
        scale_factor = 1.0 / math.sqrt(self.embed_dim)
        scores = torch.matmul(q, keys.T) * scale_factor  # [B, seq_len, num_slots]
        attn = F.softmax(scores / temp, dim=-1)
        return torch.matmul(attn, values), attn  # [B, seq_len, C], [B, seq_len, num_slots]

    def _retrieve_batched(self, q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, temp: float):
        """
        Batched retrieval: each sample has its own key/value set.

        Args:
            q: [B, seq_len, C]
            keys: [B, num_slots, C]
            values: [B, num_slots, C]
            temp: temperature scalar
        """
        scale_factor = 1.0 / math.sqrt(self.embed_dim)
        scores = torch.bmm(q, keys.transpose(1, 2)) * scale_factor  # [B, seq_len, num_slots]
        attn = F.softmax(scores / temp, dim=-1)
        return torch.bmm(attn, values), attn

    def _get_cat_memory_batched(self, cat_ids: torch.Tensor, visible_indices: list) -> torch.Tensor:
        """
        Batch compute category memories for multiple categories across visible scales.

        Args:
            cat_ids: [U] unique category ids (all valid, >= 0)
            visible_indices: list of scale indices [0, 1, ..., i]

        Returns: [U, total_visible_slots, C]
        """
        V = len(visible_indices)
        vis_idx = torch.tensor(visible_indices, device=self.cat_B.device, dtype=torch.long)

        # Batched low-rank: cat_A[cats, vis]: [U, V, R], cat_B[vis]: [V, R, S*C]
        a = self.cat_A[cat_ids][:, vis_idx]   # [U, V, R]
        b = self.cat_B[vis_idx]               # [V, R, S*C]
        delta = torch.einsum('uvr,vrd->uvd', a, b)  # [U, V, S*C]
        delta = delta.view(len(cat_ids), V, self.slots_per_scale, self.embed_dim)  # [U, V, S, C]

        # Add shared base
        shared_vis = torch.stack([
            self.shared_memory[f'scale_{j}'].view(self.slots_per_scale, self.embed_dim)
            for j in visible_indices
        ])  # [V, S, C]
        cat_mem = shared_vis.unsqueeze(0) + delta  # [U, V, S, C]
        return cat_mem.reshape(len(cat_ids), V * self.slots_per_scale, self.embed_dim)  # [U, V*S, C]

    def forward(
        self,
        x: torch.Tensor,
        begin_ends: List[Tuple[int, int]],
        category_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: [B, L, C] 输入隐藏态 (pre-QKV)
            begin_ends: List[(start, end)] 每个尺度的token范围
            category_ids: [B] 类别ID, -1 表示无效 (CFG unconditional)

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

        # 1. 预计算 Query
        query = self.query_proj(x)  # [B, L, C]

        # 2. 预计算共享记忆 K/V (所有尺度, 单次批量调用)
        shared_mem_stacked = torch.stack([
            self.shared_memory[f'scale_{i}'].view(-1, C) for i in range(num_scales)
        ])  # [num_scales, slots_per_scale, C]
        shared_keys_all = self.key_proj(shared_mem_stacked)    # [num_scales, slots, C]
        shared_values_all = self.value_proj(shared_mem_stacked)  # [num_scales, slots, C]

        # 3. 确定是否使用类别记忆
        use_category = category_ids is not None
        has_valid_category = use_category and (category_ids >= 0).any()

        # Handle None case by creating a default tensor
        if category_ids is None:
            category_ids = torch.full((B,), -1, device=x.device, dtype=torch.long)

        # 4. 逐尺度检索
        mem_combined = torch.zeros_like(x)  # [B, L, C]
        all_attn_weights = []

        temp = self.get_current_temperature()

        for i in range(num_scales):
            start, end = begin_ends[i]
            scale_len = end - start
            if scale_len <= 0:
                continue

            q_scale = query[:, start:end, :]  # [B, scale_len, C]

            # 5. 获取可见的共享 K/V (pre-computed indices)
            visible_indices = self._visible_indices_list[i]
            k_shared = shared_keys_all[:i+1].reshape(-1, C)    # [(i+1)*slots, C]
            v_shared = shared_values_all[:i+1].reshape(-1, C)  # [(i+1)*slots, C]

            # 6. 共享分支检索
            o_shared, attn_shared = self._retrieve(q_scale, k_shared, v_shared, temp)
            # [B, scale_len, C], [B, scale_len, visible_slots]

            all_attn_weights.append((i, attn_shared))

            # 7. 类别分支检索 (批量向量化，消除逐样本循环)
            if has_valid_category:
                valid_mask = (category_ids >= 0)  # [B]
                safe_cat_ids = category_ids.clamp(min=0)  # [B], invalid→0 as dummy

                # 找唯一类别，批量计算低秩记忆
                unique_cats, inverse_idx = torch.unique(safe_cat_ids, return_inverse=True)

                # 批量: low-rank delta + shared → project K/V
                cat_mem_all = self._get_cat_memory_batched(unique_cats, visible_indices)  # [U, V*S, C]
                k_cat_all = self.key_proj(cat_mem_all)    # [U, V*S, C]
                v_cat_all = self.value_proj(cat_mem_all)  # [U, V*S, C]

                # 按样本映射到各自类别的 K/V
                k_cat = k_cat_all[inverse_idx]  # [B, V*S, C]
                v_cat = v_cat_all[inverse_idx]  # [B, V*S, C]

                # 批量 attention
                o_cat, _ = self._retrieve_batched(q_scale, k_cat, v_cat, temp)  # [B, scale_len, C]

                # 批量 alpha 门控
                q_mean = q_scale.mean(dim=1)  # [B, C]
                scale_emb = self.scale_embedding(
                    torch.tensor(i, device=device)
                ).unsqueeze(0).expand(B, -1)  # [B, C]
                class_emb = self.category_embedding(safe_cat_ids)  # [B, C]
                gate_input = torch.cat([q_mean, scale_emb, class_emb], dim=-1)  # [B, 3*C]
                alpha = torch.sigmoid(self.alpha_mlp(gate_input))  # [B, 1]
                alpha = alpha.unsqueeze(1).expand(B, scale_len, 1)  # [B, scale_len, 1]

                # 无效类别 alpha 置0 (等价于原始的 zeros 分支)
                alpha = alpha * valid_mask.float().view(B, 1, 1)

                # 融合: mem = (1 - alpha) * o_shared + alpha * o_cat
                mem_scale = (1.0 - alpha) * o_shared + alpha * o_cat
            else:
                mem_scale = o_shared

            mem_combined[:, start:end, :] = mem_scale

        # 8. K/V 投影 + 门控
        gk = torch.sigmoid(self.gk_logit)
        gv = torch.sigmoid(self.gv_logit)
        mem_k = gk * self.Wk_mem(mem_combined)  # [B, L, C]
        mem_v = gv * self.Wv_mem(mem_combined)  # [B, L, C]

        # 9. Diversity loss
        if self.training and len(all_attn_weights) > 0:
            diversity_loss = self._compute_diversity_loss(all_attn_weights)
        else:
            diversity_loss = torch.tensor(0.0, device=device)

        # 10. Slot separation loss (cached, recomputed every N steps)
        if self.training:
            self._forward_count += 1
            if self._slot_sep_cache is None or self._forward_count % self._slot_sep_interval == 0:
                slot_sep_loss = self._compute_slot_sep_loss()
                self._slot_sep_cache = slot_sep_loss
            else:
                slot_sep_loss = self._slot_sep_cache.detach()
        else:
            slot_sep_loss = torch.tensor(0.0, device=device)

        # Cache for external collection
        self.last_diversity_loss = diversity_loss
        self.last_slot_sep_loss = slot_sep_loss

        # EMA update
        if self.training:
            self._update_slot_usage_ema(all_attn_weights)

        usage_concentration, dead_slot_ratio, dead_slot_count, total_visible_slots = self._compute_usage_health(all_attn_weights)
        self._compute_entropy_health(all_attn_weights, begin_ends)
        self._last_usage_concentration = usage_concentration
        self._last_dead_slot_ratio = dead_slot_ratio
        self._last_dead_slot_count = dead_slot_count
        self._last_total_visible_slots = total_visible_slots
        div_val = diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else float(diversity_loss)
        sep_val = slot_sep_loss.item() if isinstance(slot_sep_loss, torch.Tensor) else float(slot_sep_loss)
        self._last_flatline_flag = (div_val <= self._flatline_loss_eps and sep_val <= self._flatline_loss_eps)

        return mem_k, mem_v, diversity_loss, slot_sep_loss

    def _update_slot_usage_ema(self, attn_weights_list: list):
        """Update slot usage EMA for monitoring"""
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
                self.shared_usage_ema[:scale_idx + 1].mul_(self.ema_momentum).add_(
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

    def _compute_entropy_health(self, attn_weights_list: list, begin_ends: List[Tuple[int, int]]) -> None:
        """Compute entropy metrics from real forward attention weights."""
        if len(attn_weights_list) == 0:
            self._last_entropy_ratio_weighted = 0.0
            self._last_entropy_dispersion = 0.0
            self._last_effective_slot_usage_weighted = 0.0
            self._last_max_attn_weighted = 0.0
            self._last_entropy_ratio_per_scale = {}
            return

        with torch.no_grad():
            entropy_ratios = []
            effective_slots = []
            max_attn_vals = []
            token_weights = []
            per_scale = {}

            for scale_idx, attn in attn_weights_list:
                if scale_idx >= len(begin_ends):
                    continue
                start, end = begin_ends[scale_idx]
                token_count = max(end - start, 0)
                if token_count <= 0:
                    continue

                num_slots = attn.shape[-1]
                entropy = -(attn * torch.log(attn.clamp_min(1e-8))).sum(dim=-1).mean()
                if num_slots > 1:
                    entropy_ratio = (entropy / math.log(num_slots)).item()
                else:
                    entropy_ratio = 0.0

                effective_slot_usage = math.exp(entropy.item())
                max_attn = attn.max(dim=-1).values.mean().item()

                entropy_ratios.append(entropy_ratio)
                effective_slots.append(effective_slot_usage)
                max_attn_vals.append(max_attn)
                token_weights.append(float(token_count))

                per_scale[f'scale_{scale_idx}'] = {
                    'entropy_ratio': entropy_ratio,
                    'effective_slot_usage': effective_slot_usage,
                    'num_slots': int(num_slots),
                    'token_count': int(token_count),
                }

            if not token_weights:
                self._last_entropy_ratio_weighted = 0.0
                self._last_entropy_dispersion = 0.0
                self._last_effective_slot_usage_weighted = 0.0
                self._last_max_attn_weighted = 0.0
                self._last_entropy_ratio_per_scale = {}
                return

            weight_sum = sum(token_weights)
            weighted_ratio = sum(r * w for r, w in zip(entropy_ratios, token_weights)) / weight_sum
            weighted_eff_slots = sum(u * w for u, w in zip(effective_slots, token_weights)) / weight_sum
            weighted_max_attn = sum(m * w for m, w in zip(max_attn_vals, token_weights)) / weight_sum
            weighted_var = sum(((r - weighted_ratio) ** 2) * w for r, w in zip(entropy_ratios, token_weights)) / weight_sum

            self._last_entropy_ratio_weighted = weighted_ratio
            self._last_entropy_dispersion = math.sqrt(max(weighted_var, 0.0))
            self._last_effective_slot_usage_weighted = weighted_eff_slots
            self._last_max_attn_weighted = weighted_max_attn
            self._last_entropy_ratio_per_scale = per_scale

    def _compute_diversity_loss(self, attn_weights_list: list) -> torch.Tensor:
        """多样性损失: hinge loss on max slot usage"""
        total_loss = 0.0
        K = 3

        max_collapse_ratio = 0.0
        avg_num_slots = 0.0

        for _scale_idx, attn in attn_weights_list:
            num_slots_eff = attn.shape[-1]
            slot_usage = attn.mean(dim=(0, 1))  # [num_slots]
            max_usage = slot_usage.max()

            collapse_ratio = max_usage * num_slots_eff
            max_collapse_ratio = max(max_collapse_ratio, collapse_ratio.item() if isinstance(collapse_ratio, torch.Tensor) else collapse_ratio)
            avg_num_slots += num_slots_eff

            collapse_threshold = K / num_slots_eff
            hinge_loss = F.relu(max_usage - collapse_threshold)
            total_loss = total_loss + hinge_loss

        self._last_collapse_ratio = max_collapse_ratio
        self._last_avg_num_slots = avg_num_slots / max(len(attn_weights_list), 1)

        return total_loss / max(len(attn_weights_list), 1)

    def _compute_slot_sep_loss(self) -> torch.Tensor:
        """
        槽位分离损失: mean_{i!=j} cos(M_i, M_j)^2

        鼓励不同记忆槽学到不同的内容, 防止"伪多样性"
        """
        all_slots = []
        for i in range(self.num_scales):
            mem = self.shared_memory[f'scale_{i}'].view(-1, self.embed_dim)
            all_slots.append(mem)
        all_slots = torch.cat(all_slots, dim=0)  # [total_slots, C]

        # Normalize
        slots_normed = F.normalize(all_slots, dim=-1)

        # Pairwise cosine similarity
        sim_matrix = slots_normed @ slots_normed.T  # [N, N]

        # Mask diagonal
        N = sim_matrix.shape[0]
        mask = ~torch.eye(N, dtype=torch.bool, device=sim_matrix.device)
        off_diag = sim_matrix[mask]

        # mean(cos^2)
        return (off_diag ** 2).mean()

    def get_diagnostics(self) -> dict:
        with torch.no_grad():
            all_shared = []
            for i in range(self.num_scales):
                mem = self.shared_memory[f'scale_{i}']
                all_shared.append(mem.view(-1, self.embed_dim))
            all_shared = torch.cat(all_shared, dim=0)
            shared_norms = all_shared.norm(dim=-1)

            # Compute slot similarity for shared memory
            slots_normed = F.normalize(all_shared, dim=-1)
            similarity = slots_normed @ slots_normed.T
            mask = ~torch.eye(len(all_shared), dtype=torch.bool, device=similarity.device)
            off_diag_sim = similarity[mask]

            # Usage EMA stats
            usage = self.shared_usage_ema.flatten().cpu().numpy()

            gk = torch.sigmoid(self.gk_logit).item()
            gv = torch.sigmoid(self.gv_logit).item()
            current_temp = self.get_current_temperature()
            if isinstance(current_temp, torch.Tensor):
                current_temp = current_temp.item()

            return {
                'layer': self.block_idx,
                'shared_norm_mean': shared_norms.mean().item(),
                'shared_norm_std': shared_norms.std().item(),
                'similarity_mean': off_diag_sim.mean().item(),
                'similarity_std': off_diag_sim.std().item(),
                'usage_mean': usage.mean(),
                'usage_std': usage.std(),
                'temperature': current_temp,
                'gk_weight': gk,
                'gv_weight': gv,
                'cat_A_norm': self.cat_A.norm().item(),
                'usage_concentration': self._last_usage_concentration,
                'dead_slot_ratio': self._last_dead_slot_ratio,
                'dead_slot_count': self._last_dead_slot_count,
                'total_visible_slots': self._last_total_visible_slots,
                'flatline': bool(self._last_flatline_flag),
                'entropy_ratio_weighted': self._last_entropy_ratio_weighted,
                'entropy_dispersion': self._last_entropy_dispersion,
                'effective_slot_usage_weighted': self._last_effective_slot_usage_weighted,
                'max_attn_weighted': self._last_max_attn_weighted,
                'entropy_ratio_per_scale': self._last_entropy_ratio_per_scale,
            }

    def extra_repr(self) -> str:
        gk = torch.sigmoid(self.gk_logit).item()
        gv = torch.sigmoid(self.gv_logit).item()
        return (f'embed_dim={self.embed_dim}, num_categories={self.num_categories}, '
                f'num_scales={self.num_scales}, '
                f'shared_slots={self.slots_per_scale}/scale, '
                f'cat_rank={self.cat_rank}, '
                f'gk={gk:.3f}, gv={gv:.3f}, '
                f'mode=separate_attention+low_rank_residual')


if __name__ == '__main__':
    print("Testing ClassAwareKnittingMemoryV2...")

    B, C = 2, 1024
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    begin_ends = []
    cur = 0
    for pn in patch_nums:
        begin_ends.append((cur, cur + pn * pn))
        cur += pn * pn
    L = sum(pn * pn for pn in patch_nums)

    print(f"Total tokens L={L}")

    memory = ClassAwareKnittingMemoryV2(
        embed_dim=C,
        num_categories=22,
        num_scales=10,
        shared_patterns=8,
        shared_memory_size=4,
        cat_rank=16,
        block_idx=8,
        depth=16,
    )

    x = torch.randn(B, L, C)
    category_ids = torch.tensor([0, 14])

    # Test 1: with category IDs
    print("\n--- Test with category IDs ---")
    mem_k, mem_v, div_loss, sep_loss = memory(x, begin_ends, category_ids)
    print(f"Input shape: {x.shape}")
    print(f"mem_k shape: {mem_k.shape}")
    print(f"mem_v shape: {mem_v.shape}")
    print(f"Diversity loss: {div_loss.item():.6f}")
    print(f"Slot sep loss: {sep_loss.item():.6f}")

    # Test 2: without category IDs
    print("\n--- Test without category IDs ---")
    mem_k2, mem_v2, div_loss2, sep_loss2 = memory(x, begin_ends, None)
    print(f"mem_k shape: {mem_k2.shape}")
    print(f"Diversity loss: {div_loss2.item():.6f}")
    print(f"Slot sep loss: {sep_loss2.item():.6f}")

    # Test 3: with invalid category (CFG unconditional)
    print("\n--- Test with invalid category (-1) ---")
    cat_mixed = torch.tensor([5, -1])
    mem_k3, mem_v3, div_loss3, sep_loss3 = memory(x, begin_ends, cat_mixed)
    print(f"mem_k shape: {mem_k3.shape}")

    # Test 4: gradient flow
    print("\n--- Test gradient flow ---")
    x_grad = torch.randn(B, L, C, requires_grad=True)
    mem_k4, mem_v4, div4, sep4 = memory(x_grad, begin_ends, category_ids)
    total = mem_k4.sum() + mem_v4.sum() + div4 + sep4
    total.backward()
    print(f"x gradient exists: {x_grad.grad is not None}")
    print(f"x gradient norm: {x_grad.grad.norm().item():.6f}")

    # Diagnostics
    stats = memory.get_diagnostics()
    print(f"\nDiagnostics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    total_params = sum(p.numel() for p in memory.parameters())
    print(f"\nTotal parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    print("\n[OK] All tests passed!")
