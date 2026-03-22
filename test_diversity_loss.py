"""
单元测试：验证 diversity loss 计算正确
"""
import torch
import torch.nn.functional as F

def compute_diversity_loss_v2(attn_weights_list, K=15):
    """简化版 diversity loss 计算"""
    if len(attn_weights_list) == 0:
        return torch.tensor(0.0)
    
    total_loss = 0.0
    max_collapse_ratio = 0.0
    
    for attn in attn_weights_list:
        num_slots_eff = attn.shape[-1]
        slot_usage = attn.mean(dim=(0, 1))
        max_usage = slot_usage.max()
        
        collapse_ratio = max_usage * num_slots_eff
        max_collapse_ratio = max(max_collapse_ratio, collapse_ratio)
        
        collapse_threshold = K / num_slots_eff
        hinge_loss = F.relu(max_usage - collapse_threshold)
        total_loss = total_loss + hinge_loss
    
    return total_loss / len(attn_weights_list), max_collapse_ratio

print("=== 单元测试：Diversity Loss ===")
print()

# 测试1：完全坍缩 (one-hot)
print("测试1: 完全坍缩 (one-hot)")
attn_collapse = torch.zeros(1, 128, 400)
attn_collapse[..., 0] = 1.0
loss, ratio = compute_diversity_loss_v2([attn_collapse], K=15)
print(f"  num_slots_eff: 400")
print(f"  max_usage: 1.0")
print(f"  collapse_ratio: {ratio:.2f} (1.0*400=400x均匀)")
print(f"  threshold: {15/400}")
print(f"  loss: {loss:.4f} (应>0)")
print()

# 测试2：均匀分布
print("测试2: 均匀分布")
attn_uniform = torch.ones(1, 128, 400) / 400
loss, ratio = compute_diversity_loss_v2([attn_uniform], K=15)
print(f"  num_slots_eff: 400")
print(f"  max_usage: ~0.0025")
print(f"  collapse_ratio: {ratio:.2f} (~1.0x均匀)")
print(f"  threshold: {15/400}")
print(f"  loss: {loss:.6f} (应≈0)")
print()

# 测试3：当前状态 (max_usage=0.036)
print("测试3: 当前训练状态 (max_usage≈0.036)")
attn_current = torch.zeros(1, 128, 400)
attn_current[..., 0] = 0.036
attn_current[..., 1:] = (1 - 0.036) / 399
loss, ratio = compute_diversity_loss_v2([attn_current], K=15)
print(f"  num_slots_eff: 400")
print(f"  max_usage: 0.036")
print(f"  collapse_ratio: {ratio:.2f} (当前状态)")
thr = 15/400
print(f"  threshold: {thr:.4f}")
print(f"  loss: {loss:.6f}")
trigger = "会触发loss" if loss > 0 else "不会触发loss"
print(f"  → {trigger} (max_usage {0.036} { '>' if 0.036 > thr else '<='} threshold {thr:.4f})")
print()

# 测试4：更集中状态
print("测试4: 更集中状态 (max_usage=0.05)")
attn_concentrated = torch.zeros(1, 128, 400)
attn_concentrated[..., 0] = 0.05
attn_concentrated[..., 1:] = (1 - 0.05) / 399
loss, ratio = compute_diversity_loss_v2([attn_concentrated], K=15)
print(f"  num_slots_eff: 400")
print(f"  max_usage: 0.05")
print(f"  collapse_ratio: {ratio:.2f}")
print(f"  threshold: {15/400}")
print(f"  loss: {loss:.4f}")
print(f"  → {'会触发loss' if loss > 0 else '不会触发loss'}")
print()

# 测试5：K=10 (更严格)
print("测试5: K=10 (更严格)")
loss_k10, _ = compute_diversity_loss_v2([attn_current], K=10)
thr_k10 = 10/400
print(f"  K=10, threshold: {thr_k10:.4f}")
print(f"  max_usage: 0.036")
print(f"  loss: {loss_k10:.4f}")
trigger_k10 = "会触发loss" if loss_k10 > 0 else "不会触发loss"
print(f"  → {trigger_k10}")

print()
print("=== 测试完成 ===")
print("如果看到:")
print("  - 测试1 loss>0: 坍缩检测正常")
print("  - 测试2 loss≈0: 均匀分布不惩罚")
print("  - 测试4 loss>0: 过度集中被惩罚")
print("则 diversity loss 计算正确!")
