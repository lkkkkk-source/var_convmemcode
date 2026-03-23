#!/bin/bash
#SBATCH -J var_full_model
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:4
#SBATCH -o /share/home/u2415063010/myproject/var_convmemcode/logs/full_model.txt

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate var
export HTTP_PROXY=http://211.67.63.75:3128
export HTTPS_PROXY=http://211.67.63.75:3128
export HF_ENDPOINT=https://hf-mirror.com
python --version
export TORCH_DISTRIBUTED_DEBUG=DETAIL
cd /share/home/u2415063010/myproject/var_convmemcode

torchrun --nproc_per_node=4 --master_port=18556 train.py \
	--data_path=./dataset_v3_patches \
	--exp_name=full_model_d12 \
	--bs=16 \
	--depth=12 \
	--ep=250 \
	--fp16=1 \
	--alng=1e-3 \
	--wpe=0.1 \
	--twde=0.05 \
	--twd=0.08 \
	--hflip=True \
	--vflip=True \
	--rand_rot=True \
	--cyclic_shift=True \
	--ls=0.1 \
	--drop=0.1 \
	--workers=5 \
	--tex=True \
	--tex_layers=6_7_8_9_10_11 \
	--tex_scales=3_5_7_11 \
	--mem=1 \
	--mem_layers=6_9 \
	--mem_class_aware=1 \
	--mem_num_categories=6 \
	--mem_patterns=4 \
	--mem_size=4 \
	--mem_div_weight=0.01 \
	--mem_temp_warmup=50 \
	--aux_cls_weight=0.10 \
	--aux_tap_layer=3 \
	--seam_weight=0.02 \
	--seam_warmup=10 \
	--slot_sep_weight=0.001
