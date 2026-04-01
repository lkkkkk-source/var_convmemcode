# PROJECT KNOWLEDGE BASE

**Generated:** 2026-04-01
**Commit:** `43c3b33`
**Branch:** `finetune`

## OVERVIEW
Core VAR transformer with axial texture enhancement and knitting pattern memory. Two memory variants: base (KnittingPatternMemory) and class-aware (ClassAwareKnittingMemoryV2). Texture path: row/col/diag depthwise conv with circular padding, modulates K/V/Q via low-rank projections. Memory path: per-scale memory banks with causal visibility, outputs K/V modulation vectors.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Model wiring | `__init__.py` (build_vae_var), `var.py` (VAR class) | Main construction path |
| Texture path | `basic_var.py` (SelfAttention._compute_texture_modulation, row_ops/col_ops/diag_ops) | Circular padding, per-scale kernel selection |
| Memory path | `knitting_memory.py` (KnittingPatternMemory), `class_aware_memory.py` (ClassAwareKnittingMemoryV2) | Base and class-aware variants |
| Quantization | `quant.py` (VectorQuantizer2) | Experimental code paths present |

## CONVENTIONS
- Texture and memory inject into attention K/V path, not residual x
- Layer lists use underscore-separated strings: `8_12`
- Memory uses causal slot visibility: scale i sees [0..i]
- Texture uses circular padding for periodic knitting patterns
- Gates initialized progressively across depth

## ANTI-PATTERNS
- Do not change train/eval flag names; checkpoint compatibility depends on them
- Do not use experimental quantization code paths in `quant.py` without validation
- Do not remove DDP quirks (find_unused_parameters) for sparse texture+memory branches
- Do not mismatch memory params between train and eval

## NOTES
- Texture plan built lazily on first forward, validated for circular padding
- Memory gates (gk, gv) and texture gates learnable, initialized progressively
- Memory losses: diversity + slot separation (cached every 10 steps)
- Class-aware memory: shared patterns + category-specific low-rank residuals
- Quantization warnings: experimental code paths in `quant.py` not validated for normal training
