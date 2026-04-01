# PROJECT KNOWLEDGE BASE

## OVERVIEW
Runtime/data/config layer. Centralizes Args, data pipelines, distributed helpers, memory training dynamics, logging, checkpointing.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Training hyperparameters | `arg_util.py` | `Args` class via Tap; auto-computes bs/lr/paths |
| Data loading/augmentation | `data.py` | ImageFolder-style; cyclic shift for periodic patterns |
| Distributed samplers | `data_sampler.py` | `InfiniteBatchSampler`, `DistInfiniteBatchSampler`, `EvalDistributedSampler` |
| Distributed init/logging | `misc.py` | `init_distributed_mode`, `MetricLogger`, `TensorboardLogger`, `SyncPrint` |
| Memory training dynamics | `memory_scheduler.py` | Temperature annealing + diversity weight coupling |
| Memory entropy monitoring | `memory_entropy_monitor.py` | Slot collapse detection; standalone checkpoint analysis |
| LR scheduling | `lr_control.py` | Warmup + cosine/linear schedules |
| AMP/grad clipping | `amp_sc.py` | FP16/BF16 autocast wrapper around optimizer |

## CONVENTIONS
- Args uses Tap (`explicit_bool=True`). Many fields auto-computed at runtime (bs, lr, paths).
- Layer lists use underscore-separated strings: `8_12`, not JSON arrays.
- Distributed samplers are infinite; `start_ep/start_it` for resume.
- Logging only on master/local_master; `SyncPrint` redirects stdout/stderr to files.
- Memory scheduler couples temperature (high→low) with diversity weight (low→high) during warmup.

## ANTI-PATTERNS
- Do not manually set auto-computed Args fields (batch_size, glb_batch_size, patch_nums, resos, data_load_reso, paths).
- Do not use finite samplers for training; use `DistInfiniteBatchSampler`.
- Do not log on all ranks; check `dist.is_master()` or `dist.is_local_master()`.
- Do not mix train/eval flag names; they differ intentionally (tex vs enable_texture, mem vs enable_memory).
- Do not call `memory_entropy_monitor.analyze_checkpoint` during training; it's for post-hoc analysis.

## NOTES
- `dist.py` at root, not in utils/. Imported by arg_util and misc.
- `misc.py` contains `auto_resume` for checkpoint recovery.
- `memory_scheduler.py` freezes learnable temperature after warmup to avoid conflict.
- `memory_entropy_monitor.py` can run standalone: `python -c "from utils.memory_entropy_monitor import analyze_checkpoint; analyze_checkpoint('path/to/ckpt.pth')"`.
- Validation often means `python -m compileall .` plus targeted script runs.
