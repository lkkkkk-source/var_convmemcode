#!/bin/bash
#SBATCH -J var_finetune
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:4
#SBATCH -o /share/home/u2415063010/myproject/var_convmemcode/logs/finetune.txt

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate var
export HTTP_PROXY=http://211.67.63.75:3128
export HTTPS_PROXY=http://211.67.63.75:3128
export HF_ENDPOINT=https://hf-mirror.com
python --version
export TORCH_DISTRIBUTED_DEBUG=DETAIL
cd /share/home/u2415063010/myproject/var_convmemcode

torchrun --nproc_per_node=4 --master_port=18555 train.py \
	--data_path=./dataset_v3_patches \
	--exp_name=finetune_v1 \
	--pretrained_ckpt=../model_path/var_d16.pth \
	--freeze_layers=0_7 \
	--finetune_lr_scale=0.1 \
	--bs=32 \
	--depth=16 \
	--ep=200 \
	--fp16=1 \
	--tblr=5e-5 \
	--alng=1e-3 \
	--wpe=0.01 \
	--twde=0.1 \
	--hflip=True \
	--workers=5 \
	--tex=True \
	--tex_layers=9_10_11_12_13_14_15 \
	--tex_scales=3_5_7_11 \
	--mem=1 \
	--mem_layers=10_14 \
	--mem_class_aware=1 \
	--mem_num_categories=6 \
	--mem_patterns=4 \
	--mem_size=4 \
	--mem_div_weight=0.05 \
	--mem_temp_warmup=30 \
	--aux_cls_weight=0.10 \
	--aux_tap_layer=9 \
	--seam_weight=0.02 \
	--seam_warmup=10 \
	--slot_sep_weight=0.001
