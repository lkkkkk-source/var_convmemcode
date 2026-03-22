# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VAR_convMem is a research project extending [VAR (Visual AutoRegressive modeling)](https://github.com/FoundationVision/VAR) for **knitting pattern image generation**. It adds two key modules to the base VAR architecture: **axial texture enhancement** and **knitting pattern memory** (both basic and class-aware variants). The model generates 256×256 images across 22 knitting technique categories (n000–n021) using multi-scale autoregressive token prediction with a VQVAE tokenizer.

## Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train (distributed, 2 GPU, recommended config)
```bash
torchrun --nproc_per_node=2 train.py \
  --data_path ../../datasets --exp_name class_aware_fixed_v3 \
  --bs 16 --depth 16 --ep 1000 --fp16 1 \
  --alng 1e-3 --wpe 0.1 --twde 0.05 \
  --tex True --hflip True --workers 5 \
  --mem 1 --mem_layers 8_12 --mem_class_aware 1 \
  --mem_num_categories 22 --mem_patterns 4 --mem_size 4
```

### Train (single GPU, debug)
```bash
python train.py --data_path ../../datasets --exp_name debug_run \
  --bs 4 --depth 16 --ep 10 --fp16 1 \
  --tex True --mem 1 --mem_layers 8_12 \
  --mem_class_aware 1 --mem_num_categories 22 \
  --mem_patterns 4 --mem_size 4
```

### Evaluate (FID with class-aware memory)
```bash
python test_var_convmem.py \
  --model_path ./local_output/ar-ckpt-best.pth \
  --vae_path ../model_path/vae_ch160v4096z32.pth \
  --data_path ../../datasets --depth 16 \
  --num_samples 4400 --batch_size 32 --num_classes 22 \
  --enable_texture --enable_memory --memory_enable_layers 8_12 \
  --memory_num_patterns 4 --memory_size 4 --mem_class_aware \
  --cfg 1.5 --top_k 900 --top_p 0.96 --seed 0
```

### Run tests
```bash
python test_fixes.py
python test_diversity_loss.py
```

## Architecture

### Core pipeline
`train.py` → `build_everything()` constructs data loaders, model, optimizer → `VARTrainer` (in `trainer.py`) runs training loop. Model is built via `models/__init__.py:build_vae_var()` which constructs VQVAE + VAR and initializes weights.

### Model hierarchy (`models/`)
- **`var.py` — `VAR`**: Main autoregressive model. Stacks `AdaLNSelfAttn` blocks (from `basic_var.py`) with class-conditional AdaLN. Performs multi-scale token prediction: 10 resolution stages (1×1 → 16×16, total 680 tokens). Integrates optional texture and memory modules at configurable layers.
- **`basic_var.py`**: Core Transformer blocks — `FFN`, `SelfAttention`, `AdaLNSelfAttn`, `AdaLNBeforeHead`. Supports flash attention, xformers, and fused ops.
- **`vqvae.py` / `basic_vae.py` / `quant.py`**: VQVAE tokenizer (frozen during training). `VectorQuantizer2` handles multi-scale quantization. Pretrained weights auto-downloaded from HuggingFace.
- **`knitting_memory.py` — `KnittingPatternMemory`**: Per-scale memory bank with causal visibility (current scale only sees historical + current slots). Uses ReZero residual, orthogonal init, temperature-regulated attention.
- **`class_aware_memory.py` — `ClassAwareKnittingMemory`**: Extends memory with shared base patterns + per-category specialized patterns (22 categories). Selected at runtime via class labels.
- **`gabor_texture*.py`**: Axial texture enhancement via multi-scale depthwise convolutions. Applied to attention outputs in configurable layers (default: second half of depth).

### Utilities (`utils/`)
- **`arg_util.py`**: All CLI arguments via `typed-argument-parser` (`Tap`). The `Args` class defines training hyperparameters.
- **`memory_scheduler.py`**: Temperature annealing (0.5→0.2 over warmup epochs) and diversity weight scheduling for memory modules.
- **`memory_entropy_monitor.py`**: Tracks attention entropy over memory slots for diagnostics.
- **`data.py`**: Dataset builder (ImageNet-style folder structure).
- **`misc.py`**: Checkpointing (`auto_resume`), logging, metric tracking.
- **`amp_sc.py`**: Mixed-precision optimizer wrapper.

### Distributed training
`dist.py` wraps `torch.distributed` with singleton state. Training uses `torchrun` + DDP. `--bs` is global batch size, divided across GPUs automatically. Learning rate auto-scales: `tlr = tblr * (bs / 256)`.

## Critical Conventions

- **Train vs eval flag mismatch**: Training uses `--tex`/`--mem` (bool/int), evaluation uses `--enable_texture`/`--enable_memory` (flags). Memory parameters (`--mem_layers`, `--mem_patterns`, `--mem_size`, `--mem_class_aware`) must match exactly between training and evaluation or checkpoint loading fails.
- **Layer specification**: Memory and texture layers use underscore-separated strings (e.g., `8_12` for layers 8 and 12).
- **Checkpoint auto-resume**: Training auto-resumes from `local_output/ar-ckpt-last.pth` if it exists.
- **Model width/heads**: `width = depth * 64`, `num_heads = depth`. So depth=16 gives 1024-dim, 16-head model.
- **Patch numbers**: Default 10-stage progression `1_2_3_4_5_6_8_10_13_16` (1×1 to 16×16 patches).
