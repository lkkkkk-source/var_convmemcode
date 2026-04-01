# PROJECT KNOWLEDGE BASE

**Generated:** 2026-04-01  
**Commit:** `43c3b33`  
**Branch:** `finetune`

## OVERVIEW
Research PyTorch codebase for VAR-based knitting image generation. Main extensions over base VAR: axial texture enhancement and knitting pattern memory (basic + class-aware / low-rank variants).

## STRUCTURE
```text
var_convmemcode/
├── train.py                 # main training entry
├── trainer.py               # training loop + monitoring
├── test_var_convmem.py      # eval / generation / FID-KID script
├── test_optimizations.py    # script-style correctness checks
├── test_diversity_loss.py   # standalone loss validation
├── models/                  # model architecture details → see models/AGENTS.md
├── utils/                   # arg parsing, data, schedulers, misc → see utils/AGENTS.md
├── scripts/                 # SLURM/HPC launch helpers → see scripts/AGENTS.md
└── CLAUDE.md                # extra architecture/context notes
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add/change training flags | `utils/arg_util.py` | canonical training CLI source |
| Change eval flags / outputs | `test_var_convmem.py` | eval names differ from train names |
| Modify training behavior | `trainer.py`, `train.py` | trainer owns losses, logging, monitoring |
| Change model wiring | `models/__init__.py`, `models/var.py`, `models/basic_var.py` | build path + transformer blocks |
| Memory module work | `models/knitting_memory.py`, `models/class_aware_memory.py` | sparse layer injection; class-aware variant separate |
| Texture module work | `models/basic_var.py` + `models/gabor_texture*` if present | injected into attention path |
| Data loading / augmentation | `utils/data.py`, `utils/data_sampler.py` | ImageFolder-style dataset |
| Runtime diagnostics | `trainer.py`, `utils/memory_entropy_monitor.py` | monitoring may be logging-only |
| Cluster job changes | `scripts/*.sh`, `submit_knit.sh` | environment-specific SLURM |

## CODE MAP
| Symbol / Unit | Type | Location | Role |
|---------------|------|----------|------|
| `Args` | class | `utils/arg_util.py` | centralized training hyperparameters |
| `build_vae_var` | function | `models/__init__.py` | constructs VQVAE + VAR and optional modules |
| `VARTrainer` | class | `trainer.py` | training loop, losses, monitoring, checkpoint-side behavior |
| `SelfAttention` / `AdaLNSelfAttn` | classes | `models/basic_var.py` | main transformer attention blocks |
| `KnittingPatternMemory` | class | `models/knitting_memory.py` | base memory bank with causal slot visibility |
| `ClassAwareKnittingMemoryV2` | class | `models/class_aware_memory.py` | shared memory + category low-rank residuals |

## CONVENTIONS
- Python only; no formal package/build system.
- Script-centric workflow: run files directly, not `python -m package`.
- Tests are also scripts, not a `tests/` package.
- Train/eval flags intentionally differ:
  - train: `--tex`, `--mem`, `--mem_layers`, `--mem_patterns`, `--mem_size`, `--mem_class_aware`
  - eval: `--enable_texture`, `--enable_memory`, `--memory_enable_layers`, `--memory_num_patterns`, `--memory_size`, `--mem_class_aware`
- Layer lists use underscore-separated strings: `8_12`, not JSON/YAML arrays.
- Keep changes local; repo style tolerates long arg-heavy lines.

## ANTI-PATTERNS (THIS PROJECT)
- Do **not** change train/eval flag names casually; checkpoint/eval compatibility depends on them.
- Do **not** mismatch memory-related params between train and eval.
- Do **not** remove `find_unused_parameters=True` / DDP quirks without tracing sparse texture+memory branches.
- Do **not** apply `freeze_layers` logic unless pretrained weights are actually loaded.
- Do **not** treat `test_var_convmem.py` as a unit test; it is an expensive eval script.
- Do **not** introduce new mandatory tooling configs unless explicitly requested.
- Do **not** use experimental code paths in `models/quant.py` for normal training without validating numerical behavior.

## UNIQUE STYLES
- Mixed English + Chinese comments/prints are normal; preserve local language context when editing nearby code.
- Research artifact directories (`logs/`, `local_output/`, `evaluation_results/`) are normal runtime outputs.
- HPC / SLURM scripts are first-class workflow docs, not incidental utilities.
- Validation often means `python -m compileall .` plus targeted script runs.

## COMMANDS
```bash
conda activate var
python -m pip install -r requirements.txt

# train
python train.py --data_path <DATASET_ROOT> --exp_name debug_run --bs 4 --depth 16 --ep 10 --fp16 1 --tex True --mem 1 --mem_layers 8_12 --mem_class_aware 1 --mem_num_categories 22 --mem_patterns 4 --mem_size 4

# distributed train
torchrun --nproc_per_node=2 train.py --data_path <DATASET_ROOT> --exp_name run_name --bs 16 --depth 16 --ep 1000 --fp16 1 --alng 1e-3 --wpe 0.1 --twde 0.05 --tex True --hflip True --workers 5 --mem 1 --mem_layers 8_12 --mem_class_aware 1 --mem_num_categories 22 --mem_patterns 4 --mem_size 4

# eval
python test_var_convmem.py --model_path <CKPT> --vae_path ./model_path/vae_ch160v4096z32.pth --data_path <DATASET_ROOT> --depth 16 --num_samples 4400 --batch_size 32 --num_classes 22 --enable_texture --enable_memory --memory_enable_layers 8_12 --memory_num_patterns 4 --memory_size 4 --mem_class_aware --cfg 1.5 --top_k 900 --top_p 0.96 --seed 0

# lightweight validation
python test_optimizations.py
python -c "import test_optimizations as t; t.test_texture_plan(); print('done')"
python test_diversity_loss.py
python -m compileall .
```

## NOTES
- LSP may be unavailable locally; rely on file reads + targeted greps when necessary.
- Current repo depth is shallow; child AGENTS only exist for `models/`, `utils/`, and `scripts/`.
- Child files should carry specifics; root file should stay project-wide.
