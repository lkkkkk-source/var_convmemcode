# VAR Knitting Experiment Results

## TL;DR

- Current best confirmed result: **v2.2 class-aware rank2 mem8_12 @ cfg=4.5 → FID 42.6843, KID 0.008637**
- Best previous mainline: **v2.0 cyclic_shift @ cfg=4.5 → FID 44.18, KID 0.00890**
- Key conclusion: **class-aware memory is useful, but rank=4 was too strong; lowering to rank=2 improved FID and preserved the idea**
- Current next step: **v2.3 = keep `mem_layers=8_12` and test `mem_cat_rank=1`**

---

## Locked Evaluation Protocol

- Dataset: `./dataset_v3_patches`
- Default comparison protocol: **cfg=4.5, 8700 samples, seed=0, top_k=900, top_p=0.96**
- Texture/memory eval settings for current mainline family:
  - `texture_enable_layers=12_13_14_15`
  - `memory_enable_layers=12` unless explicitly varied
- Important: compare only runs under the same eval protocol unless noted

---

## Consolidated Results Table

| Run | CFG | Tex | Mem | Key change | FID | KID | Status | Notes |
|---|---:|:---:|:---:|---|---:|---:|---|---|
| baseline | 4.5 | ✗ | ✗ | no texture, no memory | 47.54 | 0.00931 | confirmed | baseline reference |
| v1.8 ablation | 4.5 | ✓ | ✗ | texture only | 47.63 | 0.01109 | confirmed | texture alone hurts |
| v1.9 ablation | 4.5 | ✗ | ✓ | memory only | 46.61 | 0.01027 | confirmed | memory alone mildly helps |
| v1.4_ft | 1.5 | ✓ | ✓ | fine-tune recipe | 46.94 | 0.00802 | confirmed | best KID among early runs |
| v1.5 | 1.5 | ✓ | ✓ | stable fine-tune | 48.28 | 0.01263 | confirmed | same recipe, low cfg |
| v1.5 | 2.0 | ✓ | ✓ | stable fine-tune | 47.43 | 0.01196 | confirmed | cfg sweep |
| v1.5 | 2.5 | ✓ | ✓ | stable fine-tune | 46.81 | 0.01146 | confirmed | cfg sweep |
| v1.5 | 3.5 | ✓ | ✓ | stable fine-tune | 45.52 | 0.01034 | confirmed | cfg sweep |
| v1.5 | 4.0 | ✓ | ✓ | stable fine-tune | 45.31 | 0.00995 | confirmed | cfg sweep |
| v1.5 | 4.5 | ✓ | ✓ | stable fine-tune | 44.81 | 0.00948 | confirmed | strongest pre-v2 4.5 baseline |
| v1.5 | 5.0 | ✓ | ✓ | stable fine-tune | 44.43 | 0.00900 | confirmed | cfg sweep |
| v1.5 | 5.5 | ✓ | ✓ | stable fine-tune | 44.26 | 0.00852 | confirmed | best historical FID, different cfg |
| v1.6 | 1.5 | ✓ | ✓ | unfreeze07_lr02 | 48.17 | 0.01226 | confirmed | regression |
| v1.7 | 4.5 | ✓ | ✓ | unfreeze09_lr015 | 45.95 | 0.00971 | confirmed | better than most v1.x, below v1.5@4.5 |
| v2.0 | 4.5 | ✓ | ✓ | `cyclic_shift=True` | 44.18 | 0.00890 | confirmed | first confirmed v2 improvement over v1.5@4.5 |
| v2.1 rank4 | 4.5 | ✓ | ✓ | class-aware memory, `cat_rank=4` | 46.7636 | 0.010353 | confirmed | innovation too strong; regressed badly |
| v2.1 rank2 | 4.5 | ✓ | ✓ | class-aware memory, `cat_rank=2` | **43.9799** | **0.009177** | **confirmed best @ cfg=4.5** | current best confirmed mainline |
| v2.2 rank2 mem8_12 | 4.5 | ✓ | ✓ | `cat_rank=2`, `mem_layers=8_12` | **42.6843** | **0.008637** | **confirmed best @ cfg=4.5** | current best confirmed mainline |
| v2.3 rank1 mem8_12 | 4.5 | ✓ | ✓ | `cat_rank=1`, `mem_layers=8_12` | pending | pending | planned | next experiment |

---

## Mainline Conclusion (Current)

### What we learned

1. **Texture + memory synergy is real**
   - `v1.8` (texture only) is worse than baseline.
   - `v1.9` (memory only) is somewhat better than baseline.
   - The full tex+mem recipe is clearly stronger than either part alone.

2. **CFG matters a lot, but it is not the whole story**
   - `v1.5` improves steadily from cfg 1.5 → 5.5.
   - For fairness, the current decision line is locked at **cfg=4.5**.

3. **`cyclic_shift=True` was a real improvement**
   - `v2.0` improved from `v1.5 @ 4.5 = 44.81` to `44.18`.

4. **Class-aware memory was not a bad idea; it was too strong in the first version**
   - `v2.1 rank4` regressed to `46.7636`.
   - `v2.1 rank2` improved to **43.9799**, beating `v2.0`.

### Current decision

> **Adopt `v2.2 class-aware rank2 mem8_12` as the new mainline baseline for cfg=4.5 experiments.**

---

## v2.x Detailed Notes

### v2.0 — cyclic shift

- Change: add `cyclic_shift=True` on top of the v1.5-style recipe
- Result: **FID 44.18, KID 0.00890**
- Decision: keep cyclic shift in future runs

### v2.1 rank4 — class-aware too strong

- Change: enable class-aware memory with rank 4
- Result: **FID 46.7636, KID 0.010353**
- Interpretation: subjective image quality looked decent, but overall distribution matching regressed
- Decision: do **not** keep rank4 as mainline

Per-class FID (`v2.1 rank4`):

| Class | FID |
|---|---:|
| 0 | 98.53 |
| 1 | 146.85 |
| 2 | 93.06 |
| 3 | 86.58 |
| 4 | 89.64 |
| 5 | 106.13 |

### v2.1 rank2 — class-aware kept, strength reduced

- Change: reduce class-aware residual rank from 4 → 2
- Result: **FID 43.9799, KID 0.009177**
- Interpretation: this strongly supports the hypothesis that class-aware memory was helpful, but rank4 overconstrained the distribution
- Decision: **promote to new mainline baseline**

Per-class FID (`v2.1 rank2`):

| Class | FID |
|---|---:|
| 0 | 102.08 |
| 1 | 135.23 |
| 2 | 86.05 |
| 3 | 85.09 |
| 4 | 86.11 |
| 5 | 101.97 |

Per-class comparison takeaways:

- `class 1` remains the hardest class
- `rank2` improved difficult classes relative to rank4, especially by avoiding broad regression
- The innovation is now worth keeping and extending, not abandoning

### v2.2 rank2 mem8_12 — expanded memory placement works

- Change: keep `cat_rank=2`, expand memory layers from `12` to `8_12`
- Result: **FID 42.6843, KID 0.008637**
- Interpretation: adding the earlier memory layer improved the mainline instead of destabilizing it
- Decision: **promote to new best confirmed mainline**

Per-class FID (`v2.2 rank2 mem8_12`):

| Class | FID |
|---|---:|
| 0 | 91.00 |
| 1 | 137.16 |
| 2 | 82.70 |
| 3 | 87.35 |
| 4 | 81.28 |
| 5 | 108.32 |

Per-class comparison takeaways:

- strong gains on `class 0`, `class 2`, and `class 4`
- `class 1` and `class 5` remain the hardest classes
- overall gain is broad enough to trust the total FID improvement

---

## Current Best Recipes

### Best confirmed @ cfg=4.5

**Run:** `v2.2 class-aware rank2 mem8_12`

Core settings:

```bash
--cyclic_shift=True
--tex=True --tex_layers=12_13_14_15 --tex_scales=3_5_7_11
--mem=1 --mem_layers=8_12 --mem_class_aware=1 --mem_num_categories=6 --mem_cat_rank=2
--mem_patterns=4 --mem_size=4
```

### Best historical FID across all recorded cfg values

**Run:** `v1.5 @ cfg=5.5`

- FID: **44.26**
- KID: **0.00852**
- Note: not directly comparable to the locked 4.5 protocol mainline decisions

---

## Next Experiment

### v2.3 — `mem_cat_rank=1` on `mem_layers=8_12`

Goal:

- Keep the successful `mem_layers=8_12` placement
- Test whether class-aware residual can be made even lighter while preserving the v2.2 gain

Planned training identity:

- `exp_name = fined_v2.3_classaware_rank1_mem8_12`
- log file = `./logs/fined_v2.3_classaware_rank1_mem8_12.log`

Decision rule:

- If **FID < 42.6843**: adopt `mem_cat_rank=1`
- Else: keep `mem_cat_rank=2`

---

## Evidence Index

- Historical summary values were provided and confirmed in-session
- `logs/fined_v2.0_cyclic.log` — v2.0 training log
- `logs/fined_v2.1_classaware.log` — v2.1 rank4 training log
- `logs/fined_v2.1_classaware_rank2_bs24.log` — v2.1 rank2 training log
- `logs/fined_v2.2_classaware_rank2_mem8_12.log` — v2.2 training log
- `evaluation_results/fined_v2.1_classaware_cfg4.5/` — v2.1 rank4 eval outputs
- `evaluation_results/fined_v2.1_classaware_rank2_bs24_cfg4.5/` — v2.1 rank2 eval outputs
- `evaluation_results/fined_v2.2_classaware_rank2_mem8_12_cfg4.5/` — v2.2 eval outputs
