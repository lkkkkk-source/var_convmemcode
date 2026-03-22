# Ablation Experiment Scripts

This directory contains SLURM submission scripts for all ablation experiments documented in `../ABLATION_EXPERIMENTS.md`.

## Directory Structure

```
scripts/
├── submit_full_model.sh       # Experiment 0: Full model (baseline)
├── submit_no_texture.sh       # Experiment 1: No texture enhancement
├── submit_no_memory.sh        # Experiment 2: No memory module
├── submit_basic_memory.sh     # Experiment 3: Basic memory (no class-aware)
├── submit_no_aux_losses.sh    # Experiment 4: No auxiliary losses
├── submit_vanilla_var.sh      # Experiment 5: Vanilla VAR
├── submit_all_ablations.sh    # Batch submission script
└── README.md                  # This file
```

## Quick Start

### Submit Individual Experiment

```bash
cd scripts/
sbatch submit_full_model.sh
```

### Submit All Experiments

```bash
cd scripts/
bash submit_all_ablations.sh
```

### Check Job Status

```bash
squeue -u $USER
```

### Monitor Training Logs

```bash
# Real-time monitoring
tail -f /share/home/u2415063010/myproject/VAR_convmemcode/logs/full_model.txt

# View all logs
ls -lh /share/home/u2415063010/myproject/VAR_convmemcode/logs/
```

## Experiment Configurations

| Script | Texture | Memory | Class-Aware | Aux Losses | Description |
|--------|---------|--------|-------------|------------|-------------|
| `submit_full_model.sh` | ✓ | ✓ | ✓ | ✓ | Full model (baseline) |
| `submit_no_texture.sh` | ✗ | ✓ | ✓ | ✓ | Remove texture enhancement |
| `submit_no_memory.sh` | ✓ | ✗ | ✗ | ✓ | Remove memory module |
| `submit_basic_memory.sh` | ✓ | ✓ | ✗ | ✓ | Basic memory (16×8) |
| `submit_no_aux_losses.sh` | ✓ | ✓ | ✓ | ✗ | Remove aux supervision |
| `submit_vanilla_var.sh` | ✗ | ✗ | ✗ | ✗ | Original VAR |

## Output Locations

### Training Checkpoints

```
local_output/
├── full_model/
│   ├── ar-ckpt-best.pth
│   ├── ar-ckpt-last.pth
│   └── log.txt
├── no_texture/
├── no_memory/
├── basic_memory/
├── no_aux_losses/
└── vanilla_var/
```

### SLURM Logs

```
logs/
├── full_model.txt
├── no_texture.txt
├── no_memory.txt
├── basic_memory.txt
├── no_aux_losses.txt
└── vanilla_var.txt
```

## Evaluation

After training completes, evaluate each checkpoint:

```bash
# Example: Evaluate full model
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

See `../ABLATION_EXPERIMENTS.md` for complete evaluation commands for each experiment.

## Customization

### Modify Training Epochs

Edit the `--ep` parameter in each script:

```bash
--ep=1500  # Increase to 1500 epochs
```

### Change Batch Size

Edit the `--bs` parameter:

```bash
--bs=16  # Reduce to 16 if OOM
```

### Adjust GPU Count

Modify both SLURM and torchrun parameters:

```bash
#SBATCH --gres=gpu:2  # Use 2 GPUs
torchrun --nproc_per_node=2 ...  # Match GPU count
```

### Change Master Port

If port 18555 is occupied:

```bash
torchrun --master_port=18556 ...
```

## Troubleshooting

### Job Fails Immediately

Check SLURM log for errors:
```bash
cat /share/home/u2415063010/myproject/VAR_convmemcode/logs/full_model.txt
```

### Out of Memory (OOM)

Reduce batch size or use gradient accumulation:
```bash
--bs=16 --ac=2  # Effective batch size = 32
```

### Checkpoint Loading Error

Ensure memory parameters match between training and evaluation. See `../ABLATION_EXPERIMENTS.md` for details.

### Port Already in Use

Change `--master_port` to a different value (e.g., 18556, 18557).

## Notes

- All scripts use 4 GPUs by default (adjust `--gres=gpu:4` and `--nproc_per_node=4` if needed)
- Training logs are saved to `/share/home/u2415063010/myproject/VAR_convmemcode/logs/`
- Model checkpoints are saved to `local_output/<exp_name>/`
- Each experiment takes approximately 24-48 hours on 4×A100 GPUs
- Total disk space required: ~50GB per experiment (checkpoints + logs)

## Citation

If you use these scripts, please cite the VAR_convMem paper (see `../ABLATION_EXPERIMENTS.md`).
