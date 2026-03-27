"""
Memory Entropy Monitor
监控Memory模块的Attention Entropy Ratio

用法:
1. 在trainer.py中导入并使用
2. 或者独立运行来分析checkpoint

Entropy Ratio 解释:
- 1.0: 完全均匀分布，memory没有选择性
- 0.5-0.7: 良好的选择性
- < 0.3: 可能过于集中，有坍缩风险
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class MemoryEntropyMonitor:
    """监控Memory模块的Attention分布"""

    def __init__(self, var_model, patch_nums: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)):
        self.var_model = var_model
        self.patch_nums = patch_nums
        self.embed_dim = 1024  # 默认值，会在首次计算时更新

        # 构建begin_ends
        self.begin_ends = []
        cur = 0
        for pn in patch_nums:
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn * pn
        self.total_tokens = cur

        # 找到所有memory层
        self.memory_layers = []
        for idx, block in enumerate(var_model.blocks):
            if hasattr(block.attn, 'knitting_memory'):
                self.memory_layers.append(idx)

        # 统计历史
        self.history = {layer: [] for layer in self.memory_layers}

    @torch.no_grad()
    def compute_entropy_ratio(
        self,
        x: torch.Tensor,
        layer_idx: int,
        scale_idx: int = 5,  # 默认使用中间尺度
        num_tokens: int = 10,  # 采样的token数
    ) -> Dict[str, float]:
        """
        计算指定层的attention entropy ratio

        Args:
            x: [B, L, C] 输入hidden states
            layer_idx: 要分析的层索引
            scale_idx: 要分析的尺度索引
            num_tokens: 采样的token数量

        Returns:
            dict: 包含entropy_ratio, max_attn, attn_std等指标
        """
        block = self.var_model.blocks[layer_idx]
        if not hasattr(block.attn, 'knitting_memory'):
            return {}

        mem = block.attn.knitting_memory
        B, L, C = x.shape
        self.embed_dim = C

        # 计算query
        query = mem.query_proj(x)

        # 获取指定尺度的memory（兼容 class-aware 和非 class-aware 两种版本）
        scale_key = f'scale_{scale_idx}'
        if hasattr(mem, 'shared_memory'):
            if scale_key not in mem.shared_memory:
                scale_key = 'scale_0'
            shared_mem = mem.shared_memory[scale_key]
        elif hasattr(mem, 'memory_per_scale'):
            if scale_key not in mem.memory_per_scale:
                scale_key = 'scale_0'
            _m = mem.memory_per_scale[scale_key]  # [num_patterns, memory_size, C]
            shared_mem = _m.view(-1, _m.shape[-1])  # [num_patterns*memory_size, C]
        else:
            return {}
        shared_flat = shared_mem.view(-1, C)
        k_shared = mem.key_proj(shared_flat)
        num_slots = k_shared.shape[0]

        # 采样tokens
        start, end = self.begin_ends[scale_idx]
        if end - start < num_tokens:
            token_indices = list(range(start, end))
        else:
            step = (end - start) // num_tokens
            token_indices = list(range(start, end, step))[:num_tokens]

        q_sample = query[:, token_indices, :]  # [B, num_tokens, C]

        # 计算attention scores
        scores = torch.matmul(q_sample, k_shared.T) / math.sqrt(C)
        temp = mem.get_current_temperature()
        if isinstance(temp, torch.Tensor):
            temp = temp.item()

        attn = F.softmax(scores / temp, dim=-1)  # [B, num_tokens, num_slots]

        # 计算entropy
        entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=-1).mean()
        max_entropy = math.log(num_slots)
        entropy_ratio = (entropy / max_entropy).item()

        # 其他统计
        max_attn = attn.max(dim=-1).values.mean().item()
        attn_std = attn.std(dim=-1).mean().item()

        # Top-k集中度
        top5_sum = attn.topk(5, dim=-1).values.sum(dim=-1).mean().item()

        return {
            'entropy_ratio': entropy_ratio,
            'max_attn': max_attn,
            'attn_std': attn_std,
            'top5_sum': top5_sum,
            'temperature': temp,
            'num_slots': num_slots,
        }

    @torch.no_grad()
    def compute_all_layers(
        self,
        x: torch.Tensor,
        scale_idx: int = 5,
    ) -> Dict[int, Dict[str, float]]:
        """计算所有memory层的entropy ratio"""
        results = {}
        for layer_idx in self.memory_layers:
            results[layer_idx] = self.compute_entropy_ratio(x, layer_idx, scale_idx)
        return results

    def log_to_tensorboard(
        self,
        tb_lg,
        x: torch.Tensor,
        step: int,
        scale_idx: int = 5,
    ):
        """将entropy ratio记录到tensorboard"""
        results = self.compute_all_layers(x, scale_idx)

        for layer_idx, metrics in results.items():
            if metrics:
                tb_lg.update(
                    head=f'Memory/entropy_ratio_layer{layer_idx}',
                    value=metrics['entropy_ratio'],
                    step=step
                )
                tb_lg.update(
                    head=f'Memory/max_attn_layer{layer_idx}',
                    value=metrics['max_attn'],
                    step=step
                )
                tb_lg.update(
                    head=f'Memory/top5_sum_layer{layer_idx}',
                    value=metrics['top5_sum'],
                    step=step
                )

        return results

    def update_history(self, results: Dict[int, Dict[str, float]]):
        """更新历史记录"""
        for layer_idx, metrics in results.items():
            if metrics and layer_idx in self.history:
                self.history[layer_idx].append(metrics['entropy_ratio'])

    def get_summary(self) -> str:
        """获取摘要信息"""
        lines = ["Memory Entropy Monitor Summary", "=" * 50]

        for layer_idx in self.memory_layers:
            history = self.history.get(layer_idx, [])
            if history:
                recent = history[-10:] if len(history) >= 10 else history
                avg = sum(recent) / len(recent)
                lines.append(f"Layer {layer_idx}: entropy_ratio={avg:.4f} (last {len(recent)} samples)")

                # 状态判断
                if avg > 0.95:
                    lines.append(f"  状态: ❌ 太均匀，memory没有选择性")
                elif avg > 0.8:
                    lines.append(f"  状态: ⚠️  较均匀，弱选择性")
                elif avg > 0.5:
                    lines.append(f"  状态: ✅ 良好")
                elif avg > 0.3:
                    lines.append(f"  状态: ✅ 较强选择性")
                else:
                    lines.append(f"  状态: ⚠️  可能过于集中")

        return "\n".join(lines)


def analyze_checkpoint(
    ckpt_path: str,
    device: str = 'cpu',
) -> Dict[int, Dict[str, float]]:
    """
    分析checkpoint中的memory entropy

    用法:
        python -c "from utils.memory_entropy_monitor import analyze_checkpoint; analyze_checkpoint('./local_output/ar-ckpt-best.pth')"
    """
    import sys
    sys.path.insert(0, '/data/lkh/VAR/VAR_convMem')
    from models import build_vae_var

    print(f"Loading checkpoint: {ckpt_path}")

    # 构建模型
    vae, var = build_vae_var(
        device=device,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        num_classes=22, depth=16,
        shared_aln=False, attn_l2_norm=True,
        flash_if_available=False, fused_if_available=False,
        init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,
        enable_memory=True, memory_num_patterns=4, memory_size=4,
        memory_enable_layers=[8, 12],
        use_class_aware_memory=True, num_categories=22,
    )

    # 加载权重
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('trainer', {}).get('var_wo_ddp', ckpt)
    var.load_state_dict(state_dict, strict=False)
    var.eval()

    # 创建监控器
    monitor = MemoryEntropyMonitor(var)

    # 生成测试输入
    B, C = 4, 1024
    L = sum(pn * pn for pn in (1, 2, 3, 4, 5, 6, 8, 10, 13, 16))
    x = torch.randn(B, L, C, device=device)

    # 计算entropy
    print("\n" + "=" * 60)
    print("Memory Entropy Analysis")
    print("=" * 60)

    results = monitor.compute_all_layers(x)

    for layer_idx, metrics in results.items():
        print(f"\nLayer {layer_idx}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # 状态判断
        er = metrics['entropy_ratio']
        if er > 0.95:
            print(f"  状态: ❌ 太均匀")
        elif er > 0.8:
            print(f"  状态: ⚠️  较均匀")
        elif er > 0.5:
            print(f"  状态: ✅ 良好")
        elif er > 0.3:
            print(f"  状态: ✅ 较强选择性")
        else:
            print(f"  状态: ⚠️  可能过于集中")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    analyze_checkpoint(args.ckpt, args.device)
