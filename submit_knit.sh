#!/bin/bash
#SBATCH -J pytorch_var_knit
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:4
#SBATCH -o /share/home/u2415063010/myproject/VAR_convmemcode/var_16_convmem.txt

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate var
export HTTP_PROXY=http://211.67.63.75:3128
export HTTPS_PROXY=http://211.67.63.75:3128
export HF_ENDPOINT=https://hf-mirror.com
python --version
export TORCH_DISTRIBUTED_DEBUG=DETAIL
cd /share/home/u2415063010/myproject/VAR_convmemcode
torchrun --nproc_per_node 4 --master_port 18555   train.py \
	--data_path ../datasets \
	--bs 32 \
	--depth 16\
	--ep 1000 \
	--fp16 1 \
	--alng 1e-3 \
	--wpe=0.1 \
	--twde=0.05 \
	--tex=True \
	--hflip=True \
	--mem=1 \
       	--mem_layers=8_12 \
	--mem_class_aware=1 \
	--mem_num_categories=22 \
	--mem_patterns 4  \
	--mem_size 4 \
