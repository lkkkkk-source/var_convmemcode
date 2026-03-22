# VAR_convMem Ablation Experiments

## Overview

VAR_convMem extends the Visual AutoRegressive (VAR) model with three enhancement modules for knitting pattern generation:

1. **Axial Texture Enhancement** — Multi-scale depthwise convolutions (row/col/diagonal branches) for local periodic texture patterns
2. **Class-Aware Knitting Memory** — Learnable memory bank with shared + category-specific patterns for structural priors
3. **Auxiliary Supervision** — Early classification head (layer 4) + seam continuity loss for improved training

### Layer Placement Philosophy

The architecture follows a hierarchical design:
- **Layers 0-7:** Global semantic learning
- **Layer 4:** Auxiliary classification tap (early structural supervision)
- **Layers 8, 12:** Memory injection (structural priors, sparse placement)
- **Layers 8-15:** Texture enhancement (continuous local refinement)

This design ensures: **early classification → mid-layer structure → deep texture refinement**.

---

## Experiment 0: Full Model (Baseline)

**Purpose:** Establish baseline performance with all modules enabled.

### Training Command

```bash
#!/bin/bash
#SBATCH -J var_full_model
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:4
#SBATCH -o ./logs/var_full_model.txt

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate var
export HTTP_PROXY=http://211.67.63.75:3128
export HTTPS_PROXY=http://211.67.63.75:3128
export HF_ENDPOINT=https://hf-mirror.com

cd /path/to/VAR_convmemcode

torchrun --nproc_per_node=4 --master_port=18555 train.py \
  --data_path=./dataset_v3_patches \
  --exp_name=full_model \
  --bs=32 \
  --depth=16 \
  --ep=1000 \
  --fp16=1 \
  --alng=1e-3 \
  --wpe=0.1 \
  --twde=0.05 \
  --hflip=True \
  --workers=5 \
  --tex=True \
  --tex_layers=8_9_10_11_12_13_14_15 \
  --tex_scales=3_5_7_11 \
  --mem=1 \
  --mem_layers=8_12 \
  --mem_class_aware=1 \
  --mem_num_categories=22 \
  --mem_patterns=4 \
  --mem_size=4 \
  --mem_div_weight=0.01 \
  --mem_temp_warmup=50 \
  --aux_cls_weight=0.10 \
  --aux_tap_layer=4 \
  --seam_weight=0.02 \
  --seam_warmup=10 \
  --slot_sep_weight=0.001
```

### Evaluation Command

```bash
python test_var_convmem.py \
  --model_path=./local_output/full_model/ar-ckpt-best.pth \
  --vae_path=../model_path/vae_ch160v4096z32.pth \
  --data_path=./dataset_v3_patches \
  --depth=16 \
  --num_classes=22 \
  --num_samples=4400 \
  --batch_size=32 \
  --enable_texture \
  --texture_enable_layers=8_9_10_11_12_13_14_15 \
  --texture_scales=3_5_7_11 \
  --enable_memory \
  --memory_enable_layers=8_12 \
  --memory_num_patterns=4 \
  --memory_size=4 \
  --mem_class_aware \
  --cfg=1.5 \
  --top_k=900 \
  --top_p=0.96 \
  --seed=0 \
  --output_dir=./evaluation_results/full_model
```

**Expected Outcome:** Best FID/KID scores. This serves as the baseline for all ablations.

---

## Experiment 1: No Texture Enhancement

**Purpose:** Measure the contribution of axial texture enhancement to local pattern quality.

**Hypothesis:** Without texture, generated images may lack fine-grained periodic structures and local coherence.

### Training Command

```bash
torchrun --nproc_per_node=4 --master_port=18555 train.py \
  --data_path=./dataset_v3_patches \
  --exp_name=no_texture \
  --bs=32 \
  --depth=16 \
  --ep=1000 \
  --fp16=1 \
  --alng=1e-3 \
  --wpe=0.1 \
  --twde=0.05 \
  --hflip=True \
  --workers=5 \
  --mem=1 \
  --mem_layers=8_12 \
  --mem_class_aware=1 \
  --mem_num_categories=22 \
  --mem_patterns=4 \
  --mem_size=4 \
  --mem_div_weight=0.01 \
  --mem_temp_warmup=50 \
  --aux_cls_weight=0.10 \
  --aux_tap_layer=4 \
  --seam_weight=0.02 \
  --seam_warmup=10 \
  --slot_sep_weight=0.001
```

**Note:** `--tex` flag removed (defaults to False).

### Evaluation Command

```bash
python test_var_convmem.py \
  --model_path=./local_output/no_texture/ar-ckpt-best.pth \
  --vae_path=../model_path/vae_ch160v4096z32.pth \
  --data_path=./dataset_v3_patches \
  --depth=16 \
  --num_classes=22 \
  --num_samples=4400 \
  --batch_size=32 \
  --enable_memory \
  --memory_enable_layers=8_12 \
  --memory_num_patterns=4 \
  --memory_size=4 \
  --mem_class_aware \
  --cfg=1.5 \
  --top_k=900 \
  --top_p=0.96 \
  --seed=0 \
  --output_dir=./evaluation_results/no_texture
```

**Note:** `--enable_texture` and related flags removed.

---

## Experiment 2: No Memory Module

**Purpose:** Measure the contribution of class-aware memory to structural consistency and category-specific patterns.

**Hypothesis:** Without memory, the model may struggle with maintaining consistent structural motifs within each knitting category.

### Training Command

```bash
torchrun --nproc_per_node=4 --master_port=18555 train.py \
  --data_path=./dataset_v3_patches \
  --exp_name=no_memory \
  --bs=32 \
  --depth=16 \
  --ep=1000 \
  --fp16=1 \
  --alng=1e-3 \
  --wpe=0.1 \
  --twde=0.05 \
  --hflip=True \
  --workers=5 \
  --tex=True \
  --tex_layers=8_9_10_11_12_13_14_15 \
  --tex_scales=3_5_7_11 \
  --aux_cls_weight=0.10 \
  --aux_tap_layer=4 \
  --seam_weight=0.02 \
  --seam_warmup=10
```

**Note:** All `--mem*` flags removed.

### Evaluation Command

```bash
python test_var_convmem.py \
  --model_path=./local_output/no_memory/ar-ckpt-best.pth \
  --vae_path=../model_path/vae_ch160v4096z32.pth \
  --data_path=./dataset_v3_patches \
  --depth=16 \
  --num_classes=22 \
  --num_samples=4400 \
  --batch_size=32 \
  --enable_texture \
  --texture_enable_layers=8_9_10_11_12_13_14_15 \
  --texture_scales=3_5_7_11 \
  --cfg=1.5 \
  --top_k=900 \
  --top_p=0.96 \
  --seed=0 \
  --output_dir=./evaluation_results/no_memory
```

**Note:** `--enable_memory` and related flags removed.

---

## Experiment 3: Basic Memory (No Class-Aware)

**Purpose:** Isolate the benefit of class-aware memory over basic shared memory.

**Hypothesis:** Class-aware memory should improve category-specific pattern generation compared to shared memory patterns.

### Training Command

```bash
torchrun --nproc_per_node=4 --master_port=18555 train.py \
  --data_path=./dataset_v3_patches \
  --exp_name=basic_memory \
  --bs=32 \
  --depth=16 \
  --ep=1000 \
  --fp16=1 \
  --alng=1e-3 \
  --wpe=0.1 \
  --twde=0.05 \
  --hflip=True \
  --workers=5 \
  --tex=True \
  --tex_layers=8_9_10_11_12_13_14_15 \
  --tex_scales=3_5_7_11 \
  --mem=1 \
  --mem_layers=8_12 \
  --mem_patterns=16 \
  --mem_size=8 \
  --mem_div_weight=0.01 \
  --mem_temp_warmup=50 \
  --aux_cls_weight=0.10 \
  --aux_tap_layer=4 \
  --seam_weight=0.02 \
  --seam_warmup=10 \
  --slot_sep_weight=0.001
```

**Note:** `--mem_class_aware` removed (defaults to False). Using larger memory (16 patterns × 8 slots) to match capacity.

### Evaluation Command

```bash
python test_var_convmem.py \
  --model_path=./local_output/basic_memory/ar-ckpt-best.pth \
  --vae_path=../model_path/vae_ch160v4096z32.pth \
  --data_path=./dataset_v3_patches \
  --depth=16 \
  --num_classes=22 \
  --num_samples=4400 \
  --batch_size=32 \
  --enable_texture \
  --texture_enable_layers=8_9_10_11_12_13_14_15 \
  --texture_scales=3_5_7_11 \
  --enable_memory \
  --memory_enable_layers=8_12 \
  --memory_num_patterns=16 \
  --memory_size=8 \
  --cfg=1.5 \
  --top_k=900 \
  --top_p=0.96 \
  --seed=0 \
  --output_dir=./evaluation_results/basic_memory
```

**Note:** `--mem_class_aware` flag removed.

---

## Experiment 4: No Auxiliary Losses

**Purpose:** Measure the training benefit of auxiliary supervision (early classification + seam loss).

**Hypothesis:** Auxiliary losses improve training stability and convergence speed, but may have limited impact on final generation quality.

### Training Command

```bash
torchrun --nproc_per_node=4 --master_port=18555 train.py \
  --data_path=./dataset_v3_patches \
  --exp_name=no_aux_losses \
  --bs=32 \
  --depth=16 \
  --ep=1000 \
  --fp16=1 \
  --alng=1e-3 \
  --wpe=0.1 \
  --twde=0.05 \
  --hflip=True \
  --workers=5 \
  --tex=True \
  --tex_layers=8_9_10_11_12_13_14_15 \
  --tex_scales=3_5_7_11 \
  --mem=1 \
  --mem_layers=8_12 \
  --mem_class_aware=1 \
  --mem_num_categories=22 \
  --mem_patterns=4 \
  --mem_size=4 \
  --mem_div_weight=0.01 \
  --mem_temp_warmup=50 \
  --aux_cls_weight=0.0 \
  --seam_weight=0.0 \
  --slot_sep_weight=0.001
```

**Note:** `--aux_cls_weight=0.0` and `--seam_weight=0.0` disable auxiliary losses.

### Evaluation Command

```bash
python test_var_convmem.py \
  --model_path=./local_output/no_aux_losses/ar-ckpt-best.pth \
  --vae_path=../model_path/vae_ch160v4096z32.pth \
  --data_path=./dataset_v3_patches \
  --depth=16 \
  --num_classes=22 \
  --num_samples=4400 \
  --batch_size=32 \
  --enable_texture \
  --texture_enable_layers=8_9_10_11_12_13_14_15 \
  --texture_scales=3_5_7_11 \
  --enable_memory \
  --memory_enable_layers=8_12 \
  --memory_num_patterns=4 \
  --memory_size=4 \
  --mem_class_aware \
  --cfg=1.5 \
  --top_k=900 \
  --top_p=0.96 \
  --seed=0 \
  --output_dir=./evaluation_results/no_aux_losses
```

**Note:** Evaluation command unchanged (auxiliary losses don't affect inference).

---

## Experiment 5: Vanilla VAR

**Purpose:** Baseline comparison to the original VAR architecture without any enhancements.

**Hypothesis:** All three modules (texture + memory + aux losses) should provide significant improvements over vanilla VAR.

### Training Command

```bash
torchrun --nproc_per_node=4 --master_port=18555 train.py \
  --data_path=./dataset_v3_patches \
  --exp_name=vanilla_var \
  --bs=32 \
  --depth=16 \
  --ep=1000 \
  --fp16=1 \
  --alng=1e-3 \
  --wpe=0.1 \
  --twde=0.05 \
  --hflip=True \
  --workers=5
```

**Note:** All enhancement flags removed (texture, memory, auxiliary losses).

### Evaluation Command

```bash
python test_var_convmem.py \
  --model_path=./local_output/vanilla_var/ar-ckpt-best.pth \
  --vae_path=../model_path/vae_ch160v4096z32.pth \
  --data_path=./dataset_v3_patches \
  --depth=16 \
  --num_classes=22 \
  --num_samples=4400 \
  --batch_size=32 \
  --cfg=1.5 \
  --top_k=900 \
  --top_p=0.96 \
  --seed=0 \
  --output_dir=./evaluation_results/vanilla_var
```

**Note:** All enhancement flags removed.

---

## Layer Configuration Recommendations

### Default Configuration (Recommended)

For `depth=16`:
- **Texture layers:** `8_9_10_11_12_13_14_15` (or leave empty for automatic second-half)
- **Memory layers:** `8_12` (sparse placement, 2 layers)
- **Aux tap layer:** `4` (early supervision)

**Rationale:**
- Layer 4: Early classification supervision before memory injection
- Layers 8, 12: Memory provides structural priors at mid-depth, sparse to avoid overfitting
- Layers 8-15: Continuous texture enhancement for local refinement

### Alternative Configurations

**Small Dataset (Conservative):**
```bash
--mem_layers=8
--tex_layers=10_11_12_13_14_15
```
Single memory layer, texture starts later to reduce overfitting risk.

**Strong Structure Emphasis:**
```bash
--mem_layers=6_10_13
--tex_layers=8_9_10_11_12_13_14_15
```
More memory layers for stronger structural priors (use with caution on small datasets).

**Texture-Only (Lightweight):**
```bash
--tex_layers=12_13_14_15
```
Texture only in final layers, minimal parameter overhead.

---

## Evaluation Metrics

### Primary Metrics

- **FID (Fréchet Inception Distance):** Measures distribution similarity between generated and real images (lower is better)
- **KID (Kernel Inception Distance):** More robust alternative to FID, less biased by sample size (lower is better)

### Secondary Metrics

- **Per-class FID:** FID computed separately for each of the 22 knitting categories
- **Diversity:** Intra-class LPIPS distance (higher indicates more diverse generations)
- **Seam Continuity:** Visual inspection of periodic boundary alignment

### Evaluation Settings

**For FID/KID calculation:**
- `--cfg=1.5` (classifier-free guidance scale)
- `--num_samples=4400` (200 per class × 22 classes)
- `--batch_size=32`
- `--seed=0` (for reproducibility)

**For qualitative inspection:**
- `--cfg=5.0` (higher guidance for better quality)
- `--demo_only` (skip FID, generate demo grid only)

---

## Quick Reference: Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_path` | str | '../datasets' | Path to dataset |
| `--exp_name` | str | 'text' | Experiment name |
| `--bs` | int | 768 | Global batch size |
| `--depth` | int | 16 | VAR model depth |
| `--ep` | int | 250 | Training epochs |
| `--fp16` | int | 0 | Mixed precision (0: fp32, 1: fp16, 2: bf16) |
| `--workers` | int | 0 | DataLoader workers (0: auto) |
| `--hflip` | bool | False | Horizontal flip augmentation |
| **Texture Enhancement** |
| `--tex` | bool | False | Enable texture enhancement |
| `--tex_scales` | str | '3_5_7_11' | Texture kernel scales |
| `--tex_layers` | str | '' | Texture layers (empty = second half) |
| `--tex_per_head` | bool | False | Per-head kernels |
| **Memory Module** |
| `--mem` | bool | False | Enable memory |
| `--mem_layers` | str | '4_8_12' | Memory layers |
| `--mem_patterns` | int | 16 | Number of patterns |
| `--mem_size` | int | 8 | Slots per pattern |
| `--mem_class_aware` | bool | False | Class-aware memory |
| `--mem_num_categories` | int | 22 | Number of categories |
| `--mem_div_weight` | float | 0.01 | Diversity loss weight |
| `--mem_temp_warmup` | int | 50 | Temperature warmup epochs |
| **Auxiliary Losses** |
| `--aux_cls_weight` | float | 0.10 | Aux classification weight |
| `--aux_tap_layer` | int | 4 | Aux tap layer |
| `--seam_weight` | float | 0.02 | Seam loss weight |
| `--seam_warmup` | int | 10 | Seam warmup epochs |
| `--slot_sep_weight` | float | 0.001 | Slot separation weight |

---

## Quick Reference: Evaluation Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | './local_output/ar-ckpt-best.pth' | Model checkpoint |
| `--vae_path` | str | '../model_path/vae_ch160v4096z32.pth' | VAE checkpoint |
| `--data_path` | str | '../datasets' | Dataset path |
| `--depth` | int | 16 | Model depth |
| `--num_classes` | int | 22 | Number of classes |
| `--num_samples` | int | 4400 | Samples for FID |
| `--batch_size` | int | 50 | Generation batch size |
| `--cfg` | float | 1.5 | Classifier-free guidance |
| `--seed` | int | 0 | Random seed |
| `--top_k` | int | 900 | Top-k sampling |
| `--top_p` | float | 0.96 | Top-p sampling |
| `--output_dir` | str | './evaluation_results' | Output directory |
| **Texture Enhancement** |
| `--enable_texture` | flag | - | Enable texture |
| `--texture_scales` | str | '3_5_7_11' | Texture scales |
| `--texture_enable_layers` | str | None | Texture layers |
| `--texture_per_head_kernels` | flag | - | Per-head kernels |
| **Memory Module** |
| `--enable_memory` | flag | - | Enable memory |
| `--memory_enable_layers` | str | None | Memory layers |
| `--memory_num_patterns` | int | 16 | Number of patterns |
| `--memory_size` | int | 8 | Memory size |
| `--mem_class_aware` | flag | - | Class-aware memory |
| **Other** |
| `--demo_only` | flag | - | Skip FID, demo only |

---

## Critical Notes

### Train vs Eval Flag Mismatch

**Training uses:**
- `--tex` (bool), `--mem` (bool/int)

**Evaluation uses:**
- `--enable_texture` (flag), `--enable_memory` (flag)

### Parameter Matching

Memory parameters **must match exactly** between training and evaluation:
- `--mem_layers` ↔ `--memory_enable_layers`
- `--mem_patterns` ↔ `--memory_num_patterns`
- `--mem_size` ↔ `--memory_size`
- `--mem_class_aware` ↔ `--mem_class_aware`

Mismatch will cause checkpoint loading failure.

### Default Texture Layers

When `--tex_layers` is empty (training) or `--texture_enable_layers` is None (evaluation), the model defaults to `range(depth // 2, depth)`:
- For `depth=16`: layers 8-15
- For `depth=20`: layers 10-19
- For `depth=24`: layers 12-23

### Model Architecture

- **Width:** `depth × 64` (depth=16 → 1024-dim)
- **Heads:** `depth` (depth=16 → 16 heads)
- **Patch progression:** 1×1 → 2×2 → ... → 16×16 (10 scales, 680 tokens total)

---

## Running Experiments

### Step 1: Train All Configurations

```bash
# Submit all training jobs
sbatch submit_full_model.sh
sbatch submit_no_texture.sh
sbatch submit_no_memory.sh
sbatch submit_basic_memory.sh
sbatch submit_no_aux_losses.sh
sbatch submit_vanilla_var.sh
```

### Step 2: Evaluate All Checkpoints

```bash
# Run evaluation for each experiment
for exp in full_model no_texture no_memory basic_memory no_aux_losses vanilla_var; do
  python test_var_convmem.py \
    --model_path=./local_output/${exp}/ar-ckpt-best.pth \
    --output_dir=./evaluation_results/${exp} \
    [... other flags based on experiment ...]
done
```

### Step 3: Compare Results

```bash
# Collect FID/KID scores
python scripts/collect_metrics.py --results_dir=./evaluation_results
```

Expected output:
```
Experiment          | FID ↓  | KID ↓  | Per-class FID ↓
--------------------|--------|--------|----------------
Full Model          | XX.XX  | X.XXX  | XX.XX
No Texture          | XX.XX  | X.XXX  | XX.XX
No Memory           | XX.XX  | X.XXX  | XX.XX
Basic Memory        | XX.XX  | X.XXX  | XX.XX
No Aux Losses       | XX.XX  | X.XXX  | XX.XX
Vanilla VAR         | XX.XX  | X.XXX  | XX.XX
```

---

## Troubleshooting

### Checkpoint Loading Fails

**Error:** `RuntimeError: Error(s) in loading state_dict`

**Solution:** Verify memory parameters match exactly between training and evaluation:
```bash
# Training used:
--mem_layers=8_12 --mem_patterns=4 --mem_size=4

# Evaluation must use:
--memory_enable_layers=8_12 --memory_num_patterns=4 --memory_size=4
```

### OOM (Out of Memory)

**Solution:** Reduce batch size or use gradient accumulation:
```bash
--bs=16 --ac=2  # Effective batch size = 16 × 2 = 32
```

### Low FID Scores Not Improving

**Possible causes:**
1. Insufficient training epochs (try `--ep=1500`)
2. Learning rate too high/low (try `--tblr=5e-5` or `--tblr=2e-4`)
3. Memory overfitting (reduce `--mem_patterns` or use single layer `--mem_layers=8`)

---

## Citation

If you use this ablation framework, please cite:

```bibtex
@article{var_convmem2024,
  title={VAR_convMem: Visual Autoregressive Modeling with Texture Enhancement and Pattern Memory for Knitting Generation},
  author={[Your Name]},
  year={2024}
}
```
