#!/bin/bash
#SBATCH -J var_no_memory
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:4
#SBATCH -o /share/home/u2415063010/myproject/var_convmemcode/logs/no_memory.txt

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate var
export HTTP_PROXY=http://211.67.63.75:3128
export HTTPS_PROXY=http://211.67.63.75:3128
export HF_ENDPOINT=https://hf-mirror.com
python --version
export TORCH_DISTRIBUTED_DEBUG=DETAIL
cd /share/home/u2415063010/myproject/var_convmemcode

torchrun --nproc_per_node=4 --master_port=18558 train.py \
	--data_path=./dataset_v3_patches \
	--exp_name=no_memory_d12 \
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
	--aux_cls_weight=0.10 \
	--aux_tap_layer=3 \
	--seam_weight=0.02 \
	--seam_warmup=10
