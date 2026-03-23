#!/bin/bash
# Download pretrained VAR-d16 checkpoint from HuggingFace
# The original VAR repo: https://github.com/FoundationVision/VAR
# HuggingFace: https://huggingface.co/FoundationVision/var

export HTTP_PROXY=http://211.67.63.75:3128
export HTTPS_PROXY=http://211.67.63.75:3128
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p ../model_path

# VAR-d16 (depth=16, 310M params, trained on ImageNet 256x256)
echo "Downloading VAR-d16 pretrained checkpoint..."
wget -O ../model_path/var_d16.pth \
	"https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth" \
	|| python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download('FoundationVision/var', 'var_d16.pth', local_dir='../model_path')
print(f'Downloaded to: {path}')
"

echo "Done. Checkpoint saved to ../model_path/var_d16.pth"
