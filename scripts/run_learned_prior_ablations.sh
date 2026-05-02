#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES=1

MODEL_PATH="./local_output/fined_v2.2_classaware_rank2_mem8_12/ar-ckpt-last.pth"
VAE_PATH="./model_path/vae_ch160v4096z32.pth"
DATA_PATH="./dataset_v3_patches"
REAL_ROOT="./evaluation_results/fined_v2.2_classaware_rank2_mem8_12_cfg4.5/temp_all_real_images"
FAKE_ROOT="./evaluation_results/fined_v2.2_classaware_rank2_mem8_12_cfg4.5/generated_images_for_fid"

DEPTH=16
NUM_SAMPLES=8700
BATCH_SIZE=8
NUM_CLASSES=6
CFG=4.5
TOP_K=900
TOP_P=0.96
SEED=0

PRIOR_IMAGE_SIZE=256
PRIOR_BATCH_SIZE=256
PRIOR_LR=1e-4
PRIOR_WORKERS=4
PRIOR_VAL_RATIO=0.1
PRIOR_MAX_EPOCHS=100
PRIOR_PATIENCE=8
PRIOR_MIN_DELTA=1e-3
PRIOR_LOG_INTERVAL=50

DEFAULT_N=8
DEFAULT_P=64
DEFAULT_K=2
DEFAULT_W=1.0

ABLATION_ROOT="./evaluation_results/learned_prior_ablation_cfg4.5"
PRIOR_ROOT="./local_output/learned_prior_ablation"
LOG_ROOT="./logs/learned_prior_ablation"
SUMMARY_CSV="${ABLATION_ROOT}/summary.csv"
SUMMARY_MD="${ABLATION_ROOT}/summary.md"

mkdir -p "${ABLATION_ROOT}" "${PRIOR_ROOT}" "${LOG_ROOT}"

run_prior_train() {
  local patch_size="$1"
  local out_dir="${PRIOR_ROOT}/patch_local_prior_P${patch_size}"
  local log_path="${LOG_ROOT}/train_prior_P${patch_size}.log"
  local last_ckpt="${out_dir}/patch-local-prior-last.pth"
  local done_flag="${out_dir}/TRAIN_DONE"
  local -a resume_args=()

  mkdir -p "${out_dir}"

  if [[ -f "${done_flag}" ]]; then
    echo "=== Skip prior training for P=${patch_size}: already completed ==="
    return 0
  fi

  if [[ -f "${last_ckpt}" ]]; then
    echo "=== Resume prior training for P=${patch_size} from ${last_ckpt} ==="
    resume_args=(--resume "${last_ckpt}")
  else
    echo "=== Training prior for P=${patch_size} from scratch ==="
  fi

  echo "=== Training prior for P=${patch_size} ==="
  python -u train_patch_local_prior.py \
    --real_root "${REAL_ROOT}" \
    --fake_root "${FAKE_ROOT}" \
    --out_dir "${out_dir}" \
    --patch_size "${patch_size}" \
    --image_size "${PRIOR_IMAGE_SIZE}" \
    --batch_size "${PRIOR_BATCH_SIZE}" \
    --lr "${PRIOR_LR}" \
    --workers "${PRIOR_WORKERS}" \
    --seed "${SEED}" \
    --val_ratio "${PRIOR_VAL_RATIO}" \
    --max_epochs "${PRIOR_MAX_EPOCHS}" \
    --patience "${PRIOR_PATIENCE}" \
    --min_delta "${PRIOR_MIN_DELTA}" \
    --log_interval "${PRIOR_LOG_INTERVAL}" \
    "${resume_args[@]}" | tee "${log_path}"

  touch "${done_flag}"
}

run_eval() {
  local run_name="$1"
  local ckpt_path="$2"
  local n="$3"
  local p="$4"
  local k="$5"
  local w="$6"
  local out_dir="${ABLATION_ROOT}/${run_name}"
  local log_path="${LOG_ROOT}/${run_name}.log"
  local result_txt="${out_dir}/evaluation_results_tex_mem_d16_cfg${CFG}_seed${SEED}.txt"

  if [[ -f "${result_txt}" ]]; then
    echo "=== Skip evaluation ${run_name}: result already exists ==="
    return 0
  fi

  echo "=== Evaluating ${run_name} ==="
  python -u test_var_convmem.py \
    --model_path "${MODEL_PATH}" \
    --vae_path "${VAE_PATH}" \
    --data_path "${DATA_PATH}" \
    --depth "${DEPTH}" \
    --num_samples "${NUM_SAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --num_classes "${NUM_CLASSES}" \
    --fid_splits train_val_test \
    --skip_per_class_fid \
    --output_dir "${out_dir}" \
    --enable_texture \
    --texture_enable_layers 12_13_14_15 \
    --enable_memory \
    --memory_enable_layers 8_12 \
    --memory_num_patterns 4 \
    --memory_size 4 \
    --mem_class_aware \
    --memory_cat_rank 2 \
    --cfg "${CFG}" \
    --top_k "${TOP_K}" \
    --top_p "${TOP_P}" \
    --seed "${SEED}" \
    --enable_learned_local_prior \
    --local_prior_ckpt "${ckpt_path}" \
    --local_prior_weight "${w}" \
    --local_prior_candidates "${k}" \
    --local_prior_patch_size "${p}" \
    --local_prior_num_patches "${n}" | tee "${log_path}"
}

collect_results() {
  python - <<'PY'
import csv
import glob
import os
import re

ablation_root = os.path.join(".", "evaluation_results", "learned_prior_ablation_cfg4.5")
summary_csv = os.path.join(ablation_root, "summary.csv")
summary_md = os.path.join(ablation_root, "summary.md")

rows = []
pattern = re.compile(r"N(?P<N>\d+)_P(?P<P>\d+)_K(?P<K>\d+)_w(?P<W>[0-9.]+)")
fid_pat = re.compile(r"FID Score:\s*([0-9.]+)")
kid_pat = re.compile(r"KID Score:\s*([0-9.]+)")
lpips_pat = re.compile(r"LPIPS:\s*([0-9.]+)")
ssim_pat = re.compile(r"SSIM:\s*([0-9.]+)")

for run_dir in sorted(glob.glob(os.path.join(ablation_root, "learnedprior_*"))):
    run_name = os.path.basename(run_dir)
    match = pattern.search(run_name)
    if not match:
        continue
    txt_candidates = sorted(glob.glob(os.path.join(run_dir, "evaluation_results*.txt")))
    if not txt_candidates:
        continue
    txt_path = txt_candidates[0]
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    def extract(rx):
        m = rx.search(text)
        return m.group(1) if m else ""

    rows.append({
        "run_name": run_name,
        "N": match.group("N"),
        "P": match.group("P"),
        "K": match.group("K"),
        "w": match.group("W"),
        "FID": extract(fid_pat),
        "KID": extract(kid_pat),
        "LPIPS": extract(lpips_pat),
        "SSIM": extract(ssim_pat),
        "result_file": txt_path,
    })

rows.sort(key=lambda x: (int(x["P"]), int(x["N"]), int(x["K"]), float(x["w"])))

with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["run_name", "N", "P", "K", "w", "FID", "KID", "LPIPS", "SSIM", "result_file"])
    writer.writeheader()
    writer.writerows(rows)

with open(summary_md, "w", encoding="utf-8") as f:
    f.write("| Run | N | P | K | w | FID | KID | LPIPS | SSIM |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in rows:
        f.write(
            f"| {row['run_name']} | {row['N']} | {row['P']} | {row['K']} | {row['w']} | "
            f"{row['FID']} | {row['KID']} | {row['LPIPS']} | {row['SSIM']} |\n"
        )

print(f"Saved summary CSV to: {summary_csv}")
print(f"Saved summary Markdown to: {summary_md}")
PY
}

for patch_size in 32 64 96 128; do
  run_prior_train "${patch_size}"
done

run_eval "learnedprior_N1_P64_K2_w1.0_cfg4.5"  "${PRIOR_ROOT}/patch_local_prior_P64/patch-local-prior-best.pth" 1  64 2 1.0
run_eval "learnedprior_N4_P64_K2_w1.0_cfg4.5"  "${PRIOR_ROOT}/patch_local_prior_P64/patch-local-prior-best.pth" 4  64 2 1.0
run_eval "learnedprior_N8_P64_K2_w1.0_cfg4.5"  "${PRIOR_ROOT}/patch_local_prior_P64/patch-local-prior-best.pth" 8  64 2 1.0
run_eval "learnedprior_N16_P64_K2_w1.0_cfg4.5" "${PRIOR_ROOT}/patch_local_prior_P64/patch-local-prior-best.pth" 16 64 2 1.0

run_eval "learnedprior_N8_P32_K2_w1.0_cfg4.5"  "${PRIOR_ROOT}/patch_local_prior_P32/patch-local-prior-best.pth" 8  32 2 1.0
run_eval "learnedprior_N8_P64_K2_w1.0_cfg4.5"  "${PRIOR_ROOT}/patch_local_prior_P64/patch-local-prior-best.pth" 8  64 2 1.0
run_eval "learnedprior_N8_P96_K2_w1.0_cfg4.5"  "${PRIOR_ROOT}/patch_local_prior_P96/patch-local-prior-best.pth" 8  96 2 1.0
run_eval "learnedprior_N8_P128_K2_w1.0_cfg4.5" "${PRIOR_ROOT}/patch_local_prior_P128/patch-local-prior-best.pth" 8 128 2 1.0

run_eval "learnedprior_N8_P64_K1_w1.0_cfg4.5"  "${PRIOR_ROOT}/patch_local_prior_P64/patch-local-prior-best.pth" 8 64 1 1.0
run_eval "learnedprior_N8_P64_K2_w1.0_cfg4.5"  "${PRIOR_ROOT}/patch_local_prior_P64/patch-local-prior-best.pth" 8 64 2 1.0
run_eval "learnedprior_N8_P64_K4_w1.0_cfg4.5"  "${PRIOR_ROOT}/patch_local_prior_P64/patch-local-prior-best.pth" 8 64 4 1.0
run_eval "learnedprior_N8_P64_K8_w1.0_cfg4.5"  "${PRIOR_ROOT}/patch_local_prior_P64/patch-local-prior-best.pth" 8 64 8 1.0

collect_results
echo "All learned prior ablations finished."
