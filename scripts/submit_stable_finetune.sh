#!/bin/bash
#SBATCH -J var_stable_ft
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2415063010/myproject/var_convmemcode/logs/stable_finetune.txt

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate var
export HTTP_PROXY=http://211.67.63.75:3128
export HTTPS_PROXY=http://211.67.63.75:3128
export HF_ENDPOINT=https://hf-mirror.com
python --version
export TORCH_DISTRIBUTED_DEBUG=DETAIL
cd /share/home/u2415063010/myproject/var_convmemcode

torchrun --nproc_per_node=1 --master_port=18535 train.py \
	--data_path=./dataset_v3_patches \
	--exp_name=fined_v1.3_stable \
	--pretrained_ckpt=../model_path/var_d16.pth \
	--freeze_layers=0_11 \
	--finetune_lr_scale=0.1 \
	--bs=8 \
	--depth=16 \
	--ep=80 \
	--fp16=1 \
	--alng=1e-3 \
	--wpe=0.02 \
	--twde=0.05 \
	--hflip=True \
	--workers=5 \
	--tex=True \
	--tex_layers=12_13_14_15 \
	--tex_scales=3_5_7_11 \
	--mem=1 \
	--mem_layers=12 \
	--mem_class_aware=0 \
	--mem_num_categories=6 \
	--mem_patterns=4 \
	--mem_size=4 \
	--mem_div_weight=0.005 \
	--mem_temp_warmup=15 \
	--aux_cls_weight=0.02 \
	--aux_tap_layer=12 \
	--seam_weight=0.0 \
	--seam_warmup=20 \
	--slot_sep_weight=0.0002
