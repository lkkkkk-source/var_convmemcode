# PROJECT KNOWLEDGE BASE

## OVERVIEW
SLURM job submission scripts for HPC training runs. Ablation experiment variants, finetune configs, pretrained downloads. Cluster-specific environment setup baked in.

## STRUCTURE
```
scripts/
├── submit_full_model.sh       # Full model baseline
├── submit_no_texture.sh       # Texture ablation
├── submit_no_memory.sh        # Memory ablation
├── submit_basic_memory.sh     # Basic memory (no class-aware)
├── submit_no_aux_losses.sh    # Aux loss ablation
├── submit_vanilla_var.sh      # Vanilla VAR baseline
├── submit_finetune.sh         # Finetune from pretrained
├── submit_stable_finetune.sh  # Stable finetune variant
├── submit_all_ablations.sh    # Batch submit all experiments
└── download_pretrained.sh     # HF checkpoint download
```

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| Submit single experiment | `submit_*.sh` | Direct sbatch, no args |
| Submit all ablations | `submit_all_ablations.sh` | Loops through experiment array |
| Download pretrained VAR | `download_pretrained.sh` | HF mirror + proxy fallback |
| Change GPU count | Any `submit_*.sh` | Edit `--gres=gpu:N` and `--nproc_per_node=N` |
| Change epochs/batch size | Any `submit_*.sh` | Edit `--ep` or `--bs` in torchrun line |
| Change output paths | Any `submit_*.sh` | Edit `-o` SLURM directive and `cd` path |

## CONVENTIONS
- All scripts hardcode cluster path: `/share/home/u2415063010/myproject/var_convmemcode`
- Proxy always set: `HTTP_PROXY=http://211.67.63.75:3128`
- HF mirror always set: `HF_ENDPOINT=https://hf-mirror.com`
- Conda env: `var` (sourced from `/share/apps/anaconda3/etc/profile.d/conda.sh`)
- Default: 4 GPUs, master port 18555, 16 CPU tasks
- SLURM logs go to `logs/<exp_name>.txt`
- Checkpoints go to `local_output/<exp_name>/`
- Batch script sleeps 2s between submissions to avoid scheduler overload

## ANTI-PATTERNS
- Do NOT edit scripts casually without updating hardcoded paths for your cluster
- Do NOT change proxy/HF mirror settings without verifying network access
- Do NOT mismatch `--gres=gpu:N` with `--nproc_per_node=N` in torchrun
- Do NOT use port 18555 if already occupied; increment to 18556, 18557, etc.
- Do NOT submit finetune scripts without verifying pretrained checkpoint exists
- Do NOT assume `submit_all_ablations.sh` respects job dependencies; it submits blindly

## NOTES
- Scripts are cluster-specific; copy and modify paths for new environments
- Proxy settings are for Chinese network access; remove if not needed
- HF mirror is for faster downloads in China; use default HF endpoint elsewhere
- Ablation scripts correspond to experiments in `../ABLATION_EXPERIMENTS.md`
- Training time: ~24-48h per experiment on 4×A100
- Disk space: ~50GB per experiment (checkpoints + logs)
- Use `squeue -u $USER` to monitor jobs, `tail -f logs/<exp>.txt` for live output
