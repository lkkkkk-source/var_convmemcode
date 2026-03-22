# VAR_convMem 新增方法详细说明

本文档详细描述在原有 Axial Texture Enhancement + Class-Aware Knitting Memory + Diversity Loss 基础上新增的四个方法。

---

## 1. Auxiliary Classification Head（辅助分类头）

### 动机

VAR 模型的前半段（Layer 0-7）负责学习全局语义，但缺乏显式的类别监督信号。Memory 模块从 Layer 8 开始注入结构先验，如果浅层没有建立清晰的类别表示，Memory 的结构校正效果会打折扣。辅助分类头在 Layer 4 施加早期类别监督，强迫模型在浅层就学会区分 22 种编织类别的结构特征。

### 实现

在 VAR 模型中添加一个轻量分类头，从第 `aux_tap_layer`（默认第 4 层）的中间隐藏状态抽取特征：

```python
# models/var.py
self.aux_cls_tap_layer = min(aux_cls_tap_layer, depth - 1)
self.aux_cls_head = nn.Sequential(
    nn.LayerNorm(C),          # C = embed_dim = 1024 (depth=16)
    nn.Linear(C, C // 2),     # 1024 -> 512
    nn.GELU(),
    nn.Linear(C // 2, num_classes),  # 512 -> 22
)
```

### 特征提取

在 `forward()` 中，当 Transformer 执行到第 `aux_tap_layer` 层时，截取该层输出，对每个 scale 的 token 做 mean pooling，再跨 scale 平均，得到全局特征：

```python
# forward() 中
for i, b in enumerate(self.blocks):
    x_BLC = b(x=x_BLC, ...)
    if i == self.aux_cls_tap_layer:
        x_blk_tap = x_BLC  # 截取第4层输出

# Scale-balanced pooling
scale_feats = []
for start, end in self.begin_ends:
    scale_feats.append(x_blk_tap[:, start:end, :].mean(dim=1))  # [B, C]
h_cls = torch.stack(scale_feats, dim=1).mean(dim=1)  # [B, C]
aux_cls_logits = self.aux_cls_head(h_cls.float())     # [B, 22]
```

### 损失计算

在 `forward()` 内部直接计算交叉熵损失（避免 DDP double-ready 问题），并排除 CFG label-drop 的样本：

```python
valid_mask = ~drop_mask  # 排除被随机drop的样本
if valid_mask.any():
    self.last_aux_cls_loss = F.cross_entropy(
        aux_cls_logits[valid_mask],
        original_labels[valid_mask]
    )
```

在 trainer 中以权重 `aux_cls_weight`（默认 0.10）加入总损失：

```
total_loss = CE_loss + 0.10 * aux_cls_loss + ...
```

### 设计要点

- **Layer 4 而非更深**：放在 Memory（Layer 8）之前，确保是"自然学出"的类别信号，而非被 Memory 注入的
- **Scale-balanced pooling**：对所有 10 个 scale 均匀采样，避免大 scale 主导特征
- **轻量设计**：仅 LayerNorm + 2层MLP，参数量约 0.8M，不会显著增加计算开销

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--aux_cls_weight` | 0.10 | 辅助分类损失权重 |
| `--aux_tap_layer` | 4 | 特征抽取层 |

---

## 2. Seam Loss（接缝连续性损失）

### 动机

VAR 的多尺度 token 预测将图像划分为 pn×pn 的 patch 网格。编织图案具有天然的周期性和平铺特性，图像的上下边缘、左右边缘在物理上应该是连续的（可以无缝拼接）。Seam Loss 鼓励模型生成的 token 在这些边界处保持特征一致性。

### 实现

在最终隐藏状态上，对每个 scale（pn ≥ 4）计算对边 token 的余弦距离：

```python
# models/var.py - compute_seam_loss()
def compute_seam_loss(self, x_final, begin_ends):
    for si, (start, end) in enumerate(begin_ends):
        pn = int(math.sqrt(end - start))
        if pn < 4:
            continue

        # 归一化隐藏状态
        h = F.normalize(x_final[:, start:end, :].float(), dim=-1)
        h = h.view(B, pn, pn, C)

        # 左右接缝: 第0列 vs 最后一列
        seam_lr = 1.0 - F.cosine_similarity(h[:, :, 0, :], h[:, :, -1, :], dim=-1)

        # 上下接缝: 第0行 vs 最后一行
        seam_ud = 1.0 - F.cosine_similarity(h[:, 0, :, :], h[:, -1, :, :], dim=-1)

        # 按scale加权（细尺度权重更高）
        w = self.seam_scale_weights[si]
        L_seam += w * (seam_lr.mean() + seam_ud.mean())
```

### Scale 权重

细尺度的接缝更重要（直接影响视觉质量），因此权重递增：

```python
seam_scale_weights = [0.5, 0.5, 0.5, 0.8, 0.8, 1.0, 1.0, 1.2, 1.5, 1.5]
#                     1×1  2×2  3×3  4×4  5×5  6×6  8×8 10×10 13×13 16×16
```

pn < 4 的 scale 跳过（空间太小，边界概念不明确）。

### Warmup

Seam Loss 使用线性 warmup，避免训练初期干扰主 CE loss 的收敛：

```python
# train.py
if epoch < args.seam_warmup:  # 默认10 epochs
    current_seam_weight = args.seam_weight * (epoch / args.seam_warmup)
else:
    current_seam_weight = args.seam_weight
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seam_weight` | 0.02 | Seam Loss 最终权重 |
| `--seam_warmup` | 10 | Warmup epoch 数 |

---

## 3. Slot Separation Loss（槽位分离损失）

### 动机

原有的 Diversity Loss 防止单个 pattern 内的 memory slots 坍缩到相同表示。但不同 patterns 之间也可能学到高度相似的内容，导致"伪多样性"——看起来有多个 pattern，实际上存储的信息重复。Slot Separation Loss 从全局角度鼓励所有 memory slots 之间保持差异。

### 与 Diversity Loss 的区别

| | Diversity Loss | Slot Separation Loss |
|---|---|---|
| **作用范围** | 单个 pattern 内的 slots | 所有 scales 的所有 slots |
| **计算方式** | attention 权重的熵 | memory 参数的余弦相似度 |
| **防止的问题** | 单 pattern 内 slot 坍缩 | 不同 patterns 之间退化 |
| **作用对象** | 运行时 attention 分布 | 静态 memory 参数 |

### 实现

对 shared memory 中所有 scale 的所有 slots 计算两两余弦相似度的平方均值：

```python
# models/class_aware_memory.py - _compute_slot_sep_loss()
def _compute_slot_sep_loss(self):
    all_slots = []
    for i in range(self.num_scales):
        mem = self.shared_memory[f'scale_{i}'].view(-1, self.embed_dim)
        all_slots.append(mem)
    all_slots = torch.cat(all_slots, dim=0)  # [total_slots, C]

    # 归一化后计算余弦相似度矩阵
    slots_normed = F.normalize(all_slots, dim=-1)
    sim_matrix = slots_normed @ slots_normed.T  # [N, N]

    # 去掉对角线（自身相似度=1）
    mask = ~torch.eye(N, dtype=torch.bool, device=sim_matrix.device)
    off_diag = sim_matrix[mask]

    # mean(cos²) — 平方惩罚使高相似度的 pair 受到更强的梯度
    return (off_diag ** 2).mean()
```

### 设计要点

- **cos² 而非 |cos|**：平方惩罚对高相似度 pair 梯度更大，对已经分离的 pair 梯度接近零，避免过度推开已经足够不同的 slots
- **跨 scale 计算**：不仅同 scale 内的 slots 要不同，不同 scale 之间也要保持差异
- **仅作用于 shared memory**：category-specific 的部分通过低秩矩阵（cat_A, cat_B）自然分化，不需要额外约束

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--slot_sep_weight` | 0.001 | Slot Separation Loss 权重 |

---

## 4. Circular Padding（循环填充）

### 动机

编织图案具有天然的周期性——同一种针法在水平和垂直方向上重复排列。标准的 zero padding 在特征图边缘引入零值，破坏了这种周期结构。Circular padding 将特征图视为环面（torus），边缘的卷积操作会"绕回"到对面，天然适配编织图案的周期性。

### 实现

在 Texture Enhancement 的所有 depthwise conv 中，将 `padding_mode` 从默认的 `'zeros'` 改为 `'circular'`：

```python
# models/basic_var.py - SelfAttention.__init__()
padding_mode = 'circular'  # V2: circular for periodic knitting patterns

def _make_dw_pw(channels, k_h, k_w, dil_h, dil_w, pad_mode=padding_mode):
    pad_h = (k_h // 2) * dil_h
    pad_w = (k_w // 2) * dil_w
    return nn.Sequential(
        nn.Conv2d(
            channels, channels,
            kernel_size=(k_h, k_w),
            dilation=(dil_h, dil_w),
            padding=(pad_h, pad_w),
            padding_mode=pad_mode,   # circular
            groups=channels,
            bias=False
        ),
        nn.Conv2d(channels, channels, kernel_size=1, bias=True)
    )
```

### 安全检查

Circular padding 要求 padding 大小不超过特征图尺寸（否则会"绕回"多次，产生错误结果）。对于小尺度的 token map（如 1×1、2×2），padding 可能超过空间尺寸，因此添加了安全检查：

```python
# models/basic_var.py - _safe_apply_conv_seq()
def _safe_apply_conv_seq(self, op, x):
    H, W = x.shape[-2:]
    for m in op.modules():
        if isinstance(m, nn.Conv2d) and m.padding_mode == 'circular':
            pad_h, pad_w = m.padding
            if H <= pad_h or W <= pad_w:
                return None  # 跳过，避免无效的circular padding
    return op(x)
```

同时，pn < 2 的 scale 直接跳过 texture 计算：

```python
# _compute_texture_modulation()
if pn < 2:
    continue  # Skip small spatial sizes to avoid circular padding issues
```

### 影响范围

Circular padding 应用于 texture 模块的所有三个分支：
- **Row ops**：水平方向 1D depthwise conv（kernel: 1×k）
- **Col ops**：垂直方向 1D depthwise conv（kernel: k×1）
- **Diag ops**：2D depthwise conv（kernel: k×k）

每个分支包含 4 种 kernel scale（3, 5, 7, 11）× 多种 dilation rate，共约 14 组 operator triplets per layer，覆盖 Layer 8-15 共 8 层。

### 参数

无额外参数。Circular padding 在 `--tex=True` 时自动启用。

---

## 总损失函数

综合以上新增方法，完整的训练损失为：

```
L_total = L_CE                                    # 主交叉熵损失（scale加权）
        + λ_div   * L_diversity                   # 多样性损失（已有）
        + λ_sep   * L_slot_separation             # 槽位分离损失（新增）
        + λ_seam  * L_seam                        # 接缝连续性损失（新增）
        + λ_aux   * L_aux_cls                     # 辅助分类损失（新增）
```

其中：
- `λ_div = 0.01`（warmup 期间从 0 线性增长）
- `λ_sep = 0.001`
- `λ_seam = 0.02`（warmup 10 epochs 从 0 线性增长）
- `λ_aux = 0.10`

### 训练调度时间线

```
Epoch 0:   L_CE + 0.10*L_aux
           (seam warmup中, div warmup中, memory temp=0.5)

Epoch 10:  L_CE + 0.10*L_aux + 0.02*L_seam + 0.001*L_sep
           (seam warmup完成, div仍在warmup)

Epoch 50:  L_CE + 0.10*L_aux + 0.02*L_seam + 0.01*L_div + 0.001*L_sep
           (所有warmup完成, memory temp冻结在0.2)

Epoch 50+: 所有损失权重保持不变，稳定训练
```
