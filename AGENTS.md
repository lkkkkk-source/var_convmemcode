# AGENTS.md

Operational guide for coding agents working in this repository.
This project is a research-oriented PyTorch codebase for VAR-based knitting image generation.

## 1) Repository Facts

- Language: Python.
- ML stack: PyTorch 2.1.x.
- Package manager: pip inside conda environment.
- Main entrypoints: `train.py`, `test_var_convmem.py`, `test_optimizations.py`, `test_diversity_loss.py`.
- Distributed helper: `dist.py`.
- Core model code: `models/`.
- Training orchestration: `trainer.py`.
- Argument definitions: `utils/arg_util.py`.

## 2) Rules Files Status

- No `.cursorrules` file found.
- No `.cursor/rules/` directory found.
- No `.github/copilot-instructions.md` file found.
- Existing repository-specific guidance is in `CLAUDE.md`; treat it as a useful conventions source.

## 3) Environment Setup

- Recommended conda env name in this workspace: `var`.
- Activate env:
  - `conda activate var`
- Install dependencies:
  - `python -m pip install -r requirements.txt`
- Current pinned core dependency:
  - `torch~=2.1.0`

## 4) Build / Run / Test Commands

This repo does not define a formal build system (no `pyproject.toml`, `setup.cfg`, `Makefile`, or `tox.ini`).
Use script-based commands below.

### 4.1 Training

- Single-GPU debug-style run:
  - `python train.py --data_path <DATASET_ROOT> --exp_name debug_run --bs 4 --depth 16 --ep 10 --fp16 1 --tex True --mem 1 --mem_layers 8_12 --mem_class_aware 1 --mem_num_categories 22 --mem_patterns 4 --mem_size 4`
- Multi-GPU distributed run (example):
  - `torchrun --nproc_per_node=2 train.py --data_path <DATASET_ROOT> --exp_name class_aware_fixed_v3 --bs 16 --depth 16 --ep 1000 --fp16 1 --alng 1e-3 --wpe 0.1 --twde 0.05 --tex True --hflip True --workers 5 --mem 1 --mem_layers 8_12 --mem_class_aware 1 --mem_num_categories 22 --mem_patterns 4 --mem_size 4`

### 4.2 Evaluation / Inference

- Full evaluation script:
  - `python test_var_convmem.py --model_path ./local_output/ar-ckpt-best.pth --vae_path ../model_path/vae_ch160v4096z32.pth --data_path <DATASET_ROOT> --depth 16 --num_samples 4400 --batch_size 32 --num_classes 22 --enable_texture --enable_memory --memory_enable_layers 8_12 --memory_num_patterns 4 --memory_size 4 --mem_class_aware --cfg 1.5 --top_k 900 --top_p 0.96 --seed 0`
- Demo-only generation (skip FID/KID):
  - `python test_var_convmem.py --model_path <CKPT> --vae_path <VAE_CKPT> --data_path <DATASET_ROOT> --demo_only`

### 4.3 Tests

- Optimization equivalence tests:
  - `python test_optimizations.py`
- Diversity-loss script test:
  - `python test_diversity_loss.py`

### 4.4 Running a Single Test (Important)

There is no pytest config in-repo, and tests are script-centric.
Prefer one of these patterns:

- Run one function from `test_optimizations.py` directly:
  - `python -c "import test_optimizations as t; t.test_texture_plan(); print('done')"`
- Another single function example:
  - `python -c "import test_optimizations as t; t.test_class_aware_memory(); print('done')"`
- If pytest is available in your env, this may also work:
  - `pytest test_optimizations.py -k texture_plan -q`

### 4.5 Lint / Format / Type Check

- No official linter/formatter/type-check command is configured in this repository.
- If you need a sanity check before commit, use:
  - `python -m compileall .`
- Do not introduce new mandatory tooling configs unless explicitly requested.

## 5) Architecture and Critical Runtime Conventions

- Train/eval flags differ and must be matched carefully:
  - Train uses `--tex`, `--mem`, `--mem_layers`, `--mem_patterns`, `--mem_size`, `--mem_class_aware`.
  - Eval uses `--enable_texture`, `--enable_memory`, `--memory_enable_layers`, `--memory_num_patterns`, `--memory_size`, `--mem_class_aware`.
- Memory-related parameters must match between training and evaluation checkpoints.
- Layer lists are underscore-separated strings (example: `8_12`).
- Patch progression default is `1_2_3_4_5_6_8_10_13_16`.
- Model width/head convention: `embed_dim = depth * 64`, `num_heads = depth`.
- Training auto-resume uses `local_output/.../ar-ckpt-last.pth`.

## 6) Code Style Guidelines

These reflect observed repository style; follow existing patterns in touched files.

### 6.1 Imports

- Group imports in this order:
  1) Python stdlib
  2) Third-party libs
  3) Local project modules
- Keep one blank line between groups.
- Prefer explicit module imports over wildcard imports.
- Typical alias usage already in codebase is acceptable (`torch.nn.functional as F`, `import os.path as osp`).

### 6.2 Formatting and Layout

- Use 4-space indentation.
- Keep lines reasonably readable; this repo tolerates moderately long lines in argument-heavy code.
- Prefer trailing commas in multi-line argument lists.
- Preserve existing inline comment style when editing nearby code.
- Do not reformat entire files unless requested.

### 6.3 Typing

- Type hints are used partially; add hints for new public/helper functions where natural.
- Common patterns in repo:
  - `Optional[...]`, `Tuple[...]`, `List[...]`, `Union[...]`
  - tensor aliases in `trainer.py` (`Ten`, `ITen`, etc.)
- Do not over-engineer typing; stay consistent with local file style.

### 6.4 Naming Conventions

- Classes: `PascalCase` (e.g., `VARTrainer`, `KnittingPatternMemory`).
- Functions/variables: `snake_case`.
- Constants: `UPPER_CASE` for module-level constants.
- CLI flags in `arg_util.py` should remain short and stable; avoid breaking existing names.

### 6.5 Error Handling and Logging

- Prefer explicit checks with clear error messages for missing files/invalid config.
- Use `raise RuntimeError(...) from e` when wrapping subprocess/runtime failures.
- For optional deps in scripts, pattern is:
  - `try: import ... except ImportError: print guidance and exit/return`.
- Keep logging style pragmatic: `print(...)` is common in this research codebase.

### 6.6 PyTorch and Performance Practices

- Respect mixed precision and distributed semantics already implemented.
- Avoid device mismatch bugs; move tensors to the same device before arithmetic.
- Keep DDP behavior intact (for example `find_unused_parameters=True` is intentional here).
- Avoid changing numerical behavior silently in attention/memory paths.

### 6.7 Reproducibility and Determinism

- Preserve seed-handling patterns in `arg_util.py` and evaluation scripts.
- Do not remove deterministic toggles without reason.
- Keep checkpoint format compatibility when adding/removing model fields.

### 6.8 File and Script Conventions

- Training logic belongs in `train.py` + `trainer.py`.
- Model components belong in `models/`.
- Utility helpers belong in `utils/`.
- Prefer adding focused test functions to `test_optimizations.py` for new module equivalence checks.

## 7) Agent Execution Checklist

- Confirm environment: `conda activate var`.
- Install deps if needed: `python -m pip install -r requirements.txt`.
- For quick validation after edits:
  - `python test_optimizations.py` (or one targeted test function).
- If touching eval path, smoke-test with `--demo_only` to avoid expensive FID pipeline.
- If touching train/eval flags, verify train/eval naming correspondence.
- Keep changes minimal and local; avoid broad refactors unless requested.

## 8) Known Gaps / Notes

- `CLAUDE.md` mentions `test_fixes.py`, but that file is not present in current tree.
- Some scripts include Chinese comments/prints; keep existing language context in touched files.
- SLURM scripts in `scripts/` are environment-specific examples, not universal local commands.
