#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH="./local_output/fined_v2.2_classaware_rank2_mem8_12/ar-ckpt-last.pth"
VAE_PATH="./model_path/vae_ch160v4096z32.pth"
DATA_PATH="./dataset_v3_patches"
EXP_BASE="fined_v2.2_classaware_rank2_mem8_12"
for SEED in 0 1 2; do
  echo "===== Evaluating ${EXP_BASE}, seed ${SEED} ====="
  python test_var_convmem.py \
    --model_path "${MODEL_PATH}" \
    --vae_path "${VAE_PATH}" \
    --data_path "${DATA_PATH}" \
    --depth 16 \
    --num_samples 8700 \
    --batch_size 32 \
    --num_classes 6 \
    --output_dir "./evaluation_results/${EXP_BASE}_seed${SEED}_cfg4.5" \
    --enable_texture \
    --texture_enable_layers 12_13_14_15 \
    --enable_memory \
    --memory_enable_layers 8_12 \
    --memory_num_patterns 4 \
    --memory_size 4 \
    --mem_class_aware \
    --memory_cat_rank 2 \
    --cfg 4.5 \
    --top_k 900 \
    --top_p 0.96 \
    --seed "${SEED}"
done
echo "All multi-seed evaluations finished."