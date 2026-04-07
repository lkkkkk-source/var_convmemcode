# VAR Knitting Experiment Results

## TL;DR

- Current best final-model result: <span style="color:#d14; font-weight:700;">v2.2 + learned local prior @ cfg=4.5 → FID 41.3366, KID 0.007898</span>
- Current best stable baseline (without extra learned prior): <span style="color:#d14; font-weight:700;">v2.2 class-aware rank2 mem8_12 @ cfg=4.5 → FID 42.6843, KID 0.008637</span>
- Best previous mainline: **v2.0 cyclic_shift @ cfg=4.5 → FID 44.18, KID 0.00890**
- Key conclusion: **class-aware memory is useful, but rank=4 was too strong; lowering to rank=2 improved FID and preserved the idea**
- Current next step: **treat learned local prior as the strongest enhanced model; keep `v2.2` as the stable baseline without auxiliary reranking**

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
| <span style="color:#d14; font-weight:700;">v2.2 rank2 mem8_12</span> | 4.5 | ✓ | ✓ | `cat_rank=2`, `mem_layers=8_12` | <span style="color:#d14; font-weight:700;">42.6843</span> | <span style="color:#d14; font-weight:700;">0.008637</span> | <span style="color:#d14; font-weight:700;">confirmed best @ cfg=4.5</span> | current best confirmed mainline |
| v2.3 rank1 mem8_12 warmup50 | 4.5 | ✓ | ✓ | `cat_rank=1`, `mem_layers=8_12`, `mem_temp_warmup=50` | 45.1522 | 0.010480 | confirmed regression | rank1 + late warmup underperformed badly |
| v2.3 rank1 mem8_12 warmup30 ep40 | 4.5 | ✓ | ✓ | `cat_rank=1`, `mem_layers=8_12`, `mem_temp_warmup=30` | 48.3270 | 0.011600 | confirmed regression | rank1 remained too weak |
| v2.4 rank3 mem8_12 | 4.5 | ✓ | ✓ | `cat_rank=3`, `mem_layers=8_12` | 45.3039 | 0.010132 | confirmed regression | rank3 also worse than rank2 |
| v2.5 axial-only | 4.5 | ✓ | ✓ | remove diagonal/full-2D texture branch | 44.4989 | 0.009104 | confirmed regression | axial-only did not beat original texture+memory |
| v2.6 tex-kv-lite | 4.5 | ✓ | ✓ | texture only modulates K/V, not Q | 45.7310 | 0.010647 | confirmed regression | weakening texture hurt more |
| v2.7 seam-continuity | 4.5 | ✓ | ✓ | activate seam path + switch to local continuity loss | 47.1005 | pending | confirmed regression | continuity smoothing hurt realism/FID |
| local prior rerank (heuristic) | 4.5 | ✓ | ✓ | sampling-time local token reranking | **40.7663** | **0.007513** | promising prototype | best metrics so far, but visible heuristic artifacts remain |
| <span style="color:#d14; font-weight:700;">learned local prior</span> | 4.5 | ✓ | ✓ | trained patch-level realism prior + sampling-time rerank | <span style="color:#d14; font-weight:700;">41.3366</span> | <span style="color:#d14; font-weight:700;">0.007898</span> | <span style="color:#d14; font-weight:700;">new best final-model result</span> | visually more natural than heuristic prior; LPIPS/SSIM did not improve |
| v2.8 train-time local prior loss | 4.5 | ✓ | ✓ | frozen learned local prior integrated into training as auxiliary loss | 46.6388 | 0.010858 | confirmed regression | sampling-time local prior works, but naive train-time integration fails |

---

## Multi-seed Stability of v2.2

`v2.2 rank2 mem8_12` is not only the best single confirmed mainline; it is also stable across seeds.

| Run | Seed | FID | KID | LPIPS | SSIM |
|---|---:|---:|---:|---:|---:|
| v2.2 rank2 mem8_12 | 0 | 42.6843 | 0.008637 | 0.7880 ± 0.1396 | 0.0612 ± 0.0459 |
| v2.2 rank2 mem8_12 | 1 | 42.6645 | 0.008467 | 0.7935 ± 0.1390 | 0.0613 ± 0.0454 |
| v2.2 rank2 mem8_12 | 2 | 42.1888 | 0.008425 | 0.7953 ± 0.1400 | 0.0616 ± 0.0462 |

Takeaway:

- `v2.2` is **stable**, not a lucky seed-0 result.
- FID remains in the **42.2–42.7** range across all tested seeds.
- This strongly supports using `v2.2` as the locked mainline baseline for the paper.

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

> <span style="color:#d14; font-weight:700;">Adopt `v2.2 class-aware rank2 mem8_12` as the new mainline baseline for cfg=4.5 experiments.</span>

Additional 2026-04 update:

> <span style="color:#d14; font-weight:700;">Keep `v2.2` as the final stable baseline model line. Promote the learned local prior version as the strongest enhanced final-model result, since it improves FID/KID over `v2.2` while avoiding the most obvious heuristic-rerank artifacts.</span>

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

### v2.3 rank1 mem8_12 warmup50 — failed ablation

- Change: reduce `cat_rank` from 2 → 1 while keeping `mem_layers=8_12`, with `mem_temp_warmup=50`
- Result: **FID 45.1522, KID 0.010480**
- Interpretation: this is clearly worse than `v2.2`; the regression is too large to treat as simple noise
- Likely reading: `rank1` may be too weak, and/or `warmup=50` stabilizes memory too late for this training window
- Decision: **do not adopt this configuration; continue with `warmup30` test before ruling out rank1 entirely**

Per-class FID (`v2.3 rank1 mem8_12 warmup50`):

| Class | FID |
|---|---:|
| 0 | 102.53 |
| 1 | 118.80 |
| 2 | 99.43 |
| 3 | 90.72 |
| 4 | 89.85 |
| 5 | 105.53 |

Per-class comparison takeaways:

- `class 1` improved relative to `v2.2`
- but `class 0`, `class 2`, and `class 4` regressed enough to hurt total FID badly
- this looks more like a poor configuration than a purely late-stage overfitting artifact

### v2.3 rank1 mem8_12 warmup30 — failed again

- Change: keep `rank1`, shorten `mem_temp_warmup` from 50 → 30
- Result (`ep40`): **FID 48.3270, KID 0.011600**
- Interpretation: the failure is not just due to late warmup; `rank1` itself looks too weak in this setup
- Decision: stop the `rank1` line

### v2.4 rank3 mem8_12 — failed challenger

- Change: keep `mem_layers=8_12`, raise rank from 2 → 3
- Result: **FID 45.3039, KID 0.010132**
- Interpretation: the effective class-aware memory strength appears to have a narrow sweet spot around `rank=2`
- Decision: stop further rank sweep; keep `rank2`

### v2.5 axial-only — remove diag/full-2D texture branch

- Change: rollback texture branch from `row/col/diag` to `row/col` only
- Result (`best`): **FID 44.4989, KID 0.009104**
- Interpretation: simply removing the diagonal/2D texture branch does not improve the mainline
- Decision: do not adopt axial-only as the final texture formulation

### v2.6 tex-kv-lite — texture only modulates K/V

- Change: keep texture branch but remove its direct Q modulation
- Result (`best`): **FID 45.7310, KID 0.010647**
- Interpretation: the problem is not solved by simply weakening texture from QKV → KV-only; this made the result even worse than axial-only
- Decision: stop this texture weakening line

### v2.7 seam-continuity — seam path fixed and local continuity loss enabled

- Change: repair seam loss so it is actually computed during training; replace old opposite-edge seam objective with local internal continuity constraints
- Result (`best`): **FID 47.1005**
- Interpretation: enforcing local continuity in this simple form reduced realism and over-smoothed texture
- Decision: stop this seam formulation; continuity is important, but this implementation is not the right one

### Heuristic local prior reranking — strongest metrics, still not final

- Change: add a sampling-time local token reranker based on neighborhood/codebook compatibility
- Result: **FID 40.7663, KID 0.007513, LPIPS 0.7877 ± 0.1387, SSIM 0.0610 ± 0.0463**
- Interpretation: local structural priors clearly help distribution matching, but this heuristic implementation can create visible artificial local patterns and rule-like repetitions
- Decision: keep as a highly promising prototype / future direction, but not as the final paper model in its current form

### Learned local prior — strongest practical enhancement so far

- Change: replace the heuristic reranker with a learned patch-level local realism prior, then use it for sampling-time reranking
- Result: **FID 41.3366, KID 0.007898, LPIPS 0.8019 ± 0.1596, SSIM 0.0604 ± 0.0454**
- Interpretation: the learned prior still improves FID/KID substantially over `v2.2`, while producing visually more natural results than the heuristic prior; however, LPIPS/SSIM do not improve at the same time
- Decision: treat this as the current strongest enhanced model result; keep `v2.2` as the stable baseline and use the learned prior version as the main upgraded variant in the paper

### v2.8 train-time local prior loss — failed training integration

- Change: freeze the learned local prior and integrate it into training as an auxiliary local realism loss
- Result: **FID 46.6388, KID 0.010858, LPIPS 0.7875 ± 0.1518, SSIM 0.0641 ± 0.0493**
- Interpretation: this confirms a train/eval mismatch; the sampling-time learned prior helps when used as a reranker, but a naive frozen-scorer training loss degrades generation quality
- Decision: stop this train-time integration formulation; keep learned local prior as a sampling-time enhancement only

---

## Current Best Recipes

### Best confirmed @ cfg=4.5

**Run:** <span style="color:#d14; font-weight:700;">`v2.2 class-aware rank2 mem8_12`</span>

Core settings:

```bash
--cyclic_shift=True
--tex=True --tex_layers=12_13_14_15 --tex_scales=3_5_7_11
--mem=1 --mem_layers=8_12 --mem_class_aware=1 --mem_num_categories=6 --mem_cat_rank=2
--mem_patterns=4 --mem_size=4
```

### Best prototype result (not final mainline)

**Run:** `v2.2 + heuristic local prior rerank`

- FID: **40.7663**
- KID: **0.007513**
- LPIPS: **0.7877 ± 0.1387**
- SSIM: **0.0610 ± 0.0463**
- Note: metrics improved strongly, but visual artifacts mean this should be treated as a prototype rather than the final model line

### Best enhanced final-model result

**Run:** <span style="color:#d14; font-weight:700;">`v2.2 + learned local prior`</span>

- FID: <span style="color:#d14; font-weight:700;">41.3366</span>
- KID: <span style="color:#d14; font-weight:700;">0.007898</span>
- LPIPS: **0.8019 ± 0.1596**
- SSIM: **0.0604 ± 0.0454**
- Note: strongest practical enhancement so far; improves FID/KID over `v2.2` without the most obvious heuristic-local-prior artifacts

### Best historical FID across all recorded cfg values

**Run:** `v1.5 @ cfg=5.5`

- FID: **44.26**
- KID: **0.00852**
- Note: not directly comparable to the locked 4.5 protocol mainline decisions

---

## Current Research Direction

### Locked final mainline for the paper

- `v2.2 class-aware rank2 mem8_12`
- This is the most stable and reproducible full model configuration.

### Strongest enhanced variant for the paper

- `v2.2 + learned local prior`
- This is the best current enhanced result when allowing an auxiliary learned patch-level reranking module.

### Most promising next-stage direction

- improve the local prior itself rather than directly forcing it into training
- Motivation: current `v2.2` still shows two visual weaknesses:
  - local pasted / stitched artifacts in some results
  - stitch-level texture realism that is still softer than real knitted fabric
- The heuristic local prior prototype strongly improved FID/KID, and the learned local prior further showed that a trained patch-level realism prior can improve the mainline without the same level of heuristic artifact.
- However, direct train-time integration of the frozen scorer failed, so the safer conclusion is to keep local prior as a sampling-time enhancement in the current paper stage.

---

## Evidence Index

- Historical summary values were provided and confirmed in-session
- `logs/fined_v2.0_cyclic.log` — v2.0 training log
- `logs/fined_v2.1_classaware.log` — v2.1 rank4 training log
- `logs/fined_v2.1_classaware_rank2_bs24.log` — v2.1 rank2 training log
- `logs/fined_v2.2_classaware_rank2_mem8_12.log` — v2.2 training log
- `logs/fined_v2.3_classaware_rank1_mem8_12.log` — v2.3 rank1 warmup50 training log
- `logs/fined_v2.4_classaware_rank3_mem8_12.log` — v2.4 rank3 training log
- `logs/fined_v2.5_rank2_mem8_12_axialonly_matchv22*.log` — v2.5 axial-only training/eval logs
- `logs/fined_v2.6_rank2_mem8_12_tex_kv_lite*.log` — v2.6 kv-lite training/eval logs
- `logs/fined_v2.7_rank2_mem8_12_seam_continuity.log` — v2.7 seam continuity training log
- `evaluation_results/fined_v2.1_classaware_cfg4.5/` — v2.1 rank4 eval outputs
- `evaluation_results/fined_v2.1_classaware_rank2_bs24_cfg4.5/` — v2.1 rank2 eval outputs
- `evaluation_results/fined_v2.2_classaware_rank2_mem8_12_cfg4.5/` — v2.2 eval outputs
- `evaluation_results/fined_v2.3_classaware_rank1_mem8_12_cfg4.5/` — v2.3 rank1 warmup50 eval outputs
- `evaluation_results/fined_v2.4_classaware_rank3_mem8_12_best_cfg4.5/` — v2.4 eval outputs
- `evaluation_results/fined_v2.5_rank2_mem8_12_axialonly_matchv22*_cfg4.5/` — v2.5 eval outputs
- `evaluation_results/fined_v2.6_rank2_mem8_12_tex_kv_lite_best_cfg4.5/` — v2.6 eval outputs
- `evaluation_results/fined_v2.2_classaware_rank2_mem8_12_seed{0,1,2}_cfg4.5/` — v2.2 multi-seed eval outputs
- `evaluation_results/fined_v2.2_classaware_rank2_mem8_12_learnedprior*_cfg4.5/` — learned local prior eval outputs
