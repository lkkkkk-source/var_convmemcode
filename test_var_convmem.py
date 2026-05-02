################## VAR_convMem Model Test Script
# 测试训练出来的VAR_convMem模型
# 支持命令行参数配置，包括纹理增强和记忆功能
##################

import os
import os.path as osp
import torch, torchvision
import torch.nn.functional as F
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
import shutil
import glob
import argparse
import json
from collections import defaultdict

from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS

# 禁用默认参数初始化以提高加载速度
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

from models import VQVAE, build_vae_var
from models.patch_realism_scorer import PatchRealismScorer

################## 参数解析

parser = argparse.ArgumentParser(description='VAR_convMem Model Test Script')
# 基础参数
parser.add_argument('--model_path', type=str, default='./local_output/ar-ckpt-best.pth',
                    help='Path to the model checkpoint')
parser.add_argument('--vae_path', type=str, default='../model_path/vae_ch160v4096z32.pth',
                    help='Path to the VAE checkpoint')
parser.add_argument('--data_path', type=str, default='/path/to/imagenet',
                    help='Path to the ImageNet dataset')
parser.add_argument('--depth', type=int, default=16,
                    help='Model depth (16, 20, 24, 30, 36)')

# 生成参数
parser.add_argument('--cfg', type=float, default=1.5,
                    help='Classifier-free guidance scale (1.5 for FID evaluation, 5.0 for quality)')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('--num_samples', type=int, default=672,
                    help='Number of samples to generate for FID calculation (default 672 for val+test reference set)')
parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                    help='Output directory for results')
parser.add_argument('--top_k', type=int, default=900,
                    help='Top-k for sampling')
parser.add_argument('--top_p', type=float, default=0.96,
                    help='Top-p for sampling')
parser.add_argument('--fid_splits', type=str, default='val_test',
                    help='Reference splits for FID/KID/per-class FID: val_test, test, val, train, or train_val_test')
parser.add_argument('--num_classes', type=int, default=22,
                    help='Number of classes (22 for knitting dataset, 1000 for ImageNet)')
parser.add_argument('--batch_size', type=int, default=50,
                    help='Batch size for generation')
parser.add_argument('--demo_only', action='store_true',
                    help='Only generate demo images without calculating FID')
parser.add_argument('--skip_per_class_fid', action='store_true',
                    help='Skip per-class FID computation while keeping overall FID/KID evaluation')
parser.add_argument('--real_images_dir', type=str, default='',
                    help='Optional prebuilt real image directory for FID/KID/LPIPS/SSIM; skip rebuilding temp real dirs')
parser.add_argument('--enable_learned_local_prior', action='store_true',
                    help='Enable learned patch realism prior reranking during generation')
parser.add_argument('--local_prior_ckpt', type=str, default='',
                    help='Checkpoint path for learned patch realism scorer')
parser.add_argument('--local_prior_weight', type=float, default=1.0,
                    help='Weight multiplier for learned local prior scores')
parser.add_argument('--local_prior_candidates', type=int, default=1,
                    help='Number of candidate images to generate per label')
parser.add_argument('--local_prior_patch_size', type=int, default=64,
                    help='Patch size used by learned patch realism scorer')
parser.add_argument('--local_prior_num_patches', type=int, default=8,
                    help='Number of random patches scored per generated image')

# VAR_convMem 特定参数 - 纹理增强
parser.add_argument('--enable_texture', action='store_true',
                    help='Enable axial texture enhancement')
parser.add_argument('--texture_scales', type=str, default='3_5_7_11',
                    help='Texture scales (underscore-separated, e.g., 3_5_7_11)')
parser.add_argument('--texture_enable_layers', type=str, default=None,
                    help='Layers to enable texture (underscore-separated, e.g., 0_1_2)')
parser.add_argument('--texture_per_head_kernels', action='store_true',
                    help='Use per-head kernels for texture enhancement')

# VAR_convMem 特定参数 - 记忆功能
parser.add_argument('--enable_memory', action='store_true',
                    help='Enable knitting pattern memory')
parser.add_argument('--memory_num_patterns', type=int, default=16,
                    help='Number of memory patterns')
parser.add_argument('--memory_size', type=int, default=8,
                    help='Size of memory')
parser.add_argument('--memory_enable_layers', type=str, default=None,
                    help='Layers to enable memory (underscore-separated, e.g., 0_1_2)')
parser.add_argument('--mem_class_aware', action='store_true',
                    help='Use class-aware memory (ClassAwareKnittingMemory) instead of basic memory')
parser.add_argument('--memory_num_categories', type=int, default=None,
                    help='Number of categories for class-aware memory (default: use --num_classes)')
parser.add_argument('--memory_cat_rank', type=int, default=4,
                    help='Low-rank residual rank for class-aware memory')

args = parser.parse_args()


def parse_fid_splits(spec):
    alias_to_splits = {
        'train_val_test': ['train', 'val', 'test'],
        'all': ['train', 'val', 'test'],
        'val_test': ['val', 'test'],
        'test': ['test'],
        'val': ['val'],
        'train': ['train'],
    }
    key = str(spec).strip().lower()
    if key in alias_to_splits:
        return alias_to_splits[key]

    split_names = [part.strip().lower() for part in key.replace('+', '_').split('_') if part.strip()]
    valid = [name for name in split_names if name in {'train', 'val', 'test'}]
    deduped = []
    for name in valid:
        if name not in deduped:
            deduped.append(name)
    return deduped or ['val', 'test']


FID_SPLITS = parse_fid_splits(args.fid_splits)

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)

################## 参数验证和调试输出

print("\n" + "="*60)
print("🔍 VAR_convMem 测试配置参数")
print("="*60)

# 模型结构参数
print("\n📐 模型参数:")
print(f"  --depth: {args.depth}")
print(f"  --num_classes: {args.num_classes}")

# VAR_convMem 特性
print("\n🎨 VAR_convMem 特性:")
print(f"  纹理增强:")
print(f"    --enable_texture: {args.enable_texture}")
if args.enable_texture:
    print(f"    --texture_scales: {args.texture_scales}")
    print(f"    --texture_enable_layers: {args.texture_enable_layers}")
    print(f"    --texture_per_head_kernels: {args.texture_per_head_kernels}")
print(f"  记忆功能:")
print(f"    --enable_memory: {args.enable_memory}")
if args.enable_memory:
    print(f"    --memory_num_patterns: {args.memory_num_patterns}")
    print(f"    --memory_size: {args.memory_size}")
    print(f"    --memory_enable_layers: {args.memory_enable_layers}")
    print(f"    --mem_class_aware: {args.mem_class_aware}")
    print(f"    --memory_num_categories: {args.memory_num_categories}")
    print(f"    --memory_cat_rank: {args.memory_cat_rank}")

# 评估参数
print("\n⚙️ 生成参数:")
print(f"  --cfg: {args.cfg}")
print(f"  --seed: {args.seed}")
print(f"  --num_samples: {args.num_samples}")
print(f"  --batch_size: {args.batch_size}")
print(f"  --top_k: {args.top_k}")
print(f"  --top_p: {args.top_p}")
print(f"  --fid_splits: {args.fid_splits} -> {FID_SPLITS}")
print(f"  --skip_per_class_fid: {args.skip_per_class_fid}")
print(f"  --enable_learned_local_prior: {args.enable_learned_local_prior}")
if args.enable_learned_local_prior:
    print(f"  --local_prior_ckpt: {args.local_prior_ckpt}")
    print(f"  --local_prior_weight: {args.local_prior_weight}")
    print(f"  --local_prior_candidates: {args.local_prior_candidates}")
    print(f"  --local_prior_patch_size: {args.local_prior_patch_size}")
    print(f"  --local_prior_num_patches: {args.local_prior_num_patches}")

# 路径参数
print("\n📂 路径参数:")
print(f"  --model_path: {args.model_path}")
print(f"  --vae_path: {args.vae_path}")
print(f"  --data_path: {args.data_path}")
print(f"  --output_dir: {args.output_dir}")

print("="*60 + "\n")

################## 评估函数定义

def calculate_lpips(generated_images, real_images, device='cuda', sample_size=500):
    """计算LPIPS分数"""
    print("🔍 计算LPIPS分数...")

    try:
        import lpips
    except ImportError:
        print("❌ 未安装lpips，请运行: pip install lpips")
        return None, None

    # 初始化LPIPS网络
    lpips_net = lpips.LPIPS(net='alex').to(device)

    from torchvision import transforms
    # 统一图像尺寸的预处理
    target_size = (256, 256)
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])

    # 预处理函数：转换为[-1, 1]范围
    def preprocess_for_lpips(img_tensor):
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0
        return img_tensor * 2.0 - 1.0

    lpips_scores = []

    # 计算生成图像与真实图像之间的LPIPS
    min_samples = min(len(generated_images), len(real_images), sample_size)
    sample_indices = random.sample(range(min_samples), min_samples)

    for i in sample_indices:
        try:
            # 读取生成图像
            gen_img = PImage.open(generated_images[i]).convert('RGB')
            gen_tensor = preprocess(gen_img).unsqueeze(0).to(device)
            gen_tensor = preprocess_for_lpips(gen_tensor)

            # 读取真实图像
            real_img = PImage.open(real_images[i]).convert('RGB')
            real_tensor = preprocess(real_img).unsqueeze(0).to(device)
            real_tensor = preprocess_for_lpips(real_tensor)

            # 计算LPIPS
            with torch.no_grad():
                lpips_score = lpips_net(gen_tensor, real_tensor).item()
                lpips_scores.append(lpips_score)

        except Exception as e:
            continue

    if lpips_scores:
        avg_lpips = np.mean(lpips_scores)
        std_lpips = np.std(lpips_scores)
        print(f"✅ LPIPS计算完成: {avg_lpips:.4f} ± {std_lpips:.4f}")
        return avg_lpips, std_lpips
    else:
        print("❌ LPIPS计算失败")
        return None, None


def calculate_ssim(generated_images, real_images, sample_size=500):
    """计算SSIM分数 (Structural Similarity Index)"""
    print("🔍 计算SSIM分数...")

    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        print("❌ 未安装scikit-image，请运行: pip install scikit-image")
        return None, None

    ssim_scores = []

    # 计算生成图像与真实图像之间的SSIM
    min_samples = min(len(generated_images), len(real_images), sample_size)
    sample_indices = random.sample(range(min_samples), min_samples)

    for i in sample_indices:
        try:
            # 读取生成图像
            gen_img = PImage.open(generated_images[i]).convert('RGB')
            gen_img = gen_img.resize((256, 256), PImage.Resampling.LANCZOS)
            gen_array = np.array(gen_img)

            # 读取真实图像
            real_img = PImage.open(real_images[i]).convert('RGB')
            real_img = real_img.resize((256, 256), PImage.Resampling.LANCZOS)
            real_array = np.array(real_img)

            # 计算SSIM (multichannel=True for RGB images)
            ssim_score = ssim(real_array, gen_array,
                            multichannel=True,
                            channel_axis=2,
                            data_range=255)
            ssim_scores.append(ssim_score)

        except Exception as e:
            continue

    if ssim_scores:
        avg_ssim = np.mean(ssim_scores)
        std_ssim = np.std(ssim_scores)
        print(f"✅ SSIM计算完成: {avg_ssim:.4f} ± {std_ssim:.4f}")
        return avg_ssim, std_ssim
    else:
        print("❌ SSIM计算失败")
        return None, None


def build_local_prior_model(device='cuda'):
    if not args.enable_learned_local_prior:
        return None
    if not args.local_prior_ckpt:
        raise ValueError('--enable_learned_local_prior requires --local_prior_ckpt')

    ckpt = torch.load(args.local_prior_ckpt, map_location=device)
    model = PatchRealismScorer().to(device)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"✅ Learned local prior loaded from: {args.local_prior_ckpt}")
    return model


def score_batch_with_local_prior(model, images_B3HW, patch_size=64, num_patches=8):
    if model is None:
        return torch.zeros(images_B3HW.shape[0], device=images_B3HW.device)

    B, C, H, W = images_B3HW.shape
    scores = []
    for bi in range(B):
        img = images_B3HW[bi]
        patch_scores = []
        for _ in range(num_patches):
            if H <= patch_size or W <= patch_size:
                patch = F.interpolate(img.unsqueeze(0), size=(patch_size, patch_size), mode='bilinear', align_corners=False)
            else:
                top = random.randint(0, H - patch_size)
                left = random.randint(0, W - patch_size)
                patch = img[:, top:top + patch_size, left:left + patch_size].unsqueeze(0)
            with torch.no_grad():
                patch_scores.append(model(patch).squeeze(0))
        scores.append(torch.stack(patch_scores).mean())
    return torch.stack(scores)


def pil_loader_rgb(path):
    with open(path, 'rb') as f:
        img = PImage.open(f).convert('RGB')
    return img


def build_label_to_class_name(data_path, num_classes, split_names=None):
    if split_names is None:
        split_names = ['train', 'val', 'test']
    split_datasets = {}
    split_roots = {
        'train': os.path.join(data_path, 'train'),
        'val': os.path.join(data_path, 'val'),
        'test': os.path.join(data_path, 'test'),
    }

    for split_name in split_names:
        split_root = split_roots[split_name]
        if not os.path.isdir(split_root):
            continue
        split_datasets[split_name] = DatasetFolder(
            root=split_root,
            loader=pil_loader_rgb,
            extensions=IMG_EXTENSIONS,
        )

    class_name_to_label = {}
    if 'train' in split_datasets:
        class_name_to_label.update(split_datasets['train'].class_to_idx)
    else:
        class_names = set()
        for dataset in split_datasets.values():
            class_names.update(dataset.classes)
        for idx, class_name in enumerate(sorted(class_names)):
            class_name_to_label[class_name] = idx

    label_to_class_name = {}
    for class_name, label in class_name_to_label.items():
        if 0 <= label < num_classes:
            label_to_class_name[label] = class_name

    for label in range(num_classes):
        label_to_class_name.setdefault(label, f'class_{label:03d}')

    return label_to_class_name


def sanitize_class_name(class_name):
    sanitized = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(class_name).strip())
    sanitized = '_'.join(part for part in sanitized.split('_') if part)
    return sanitized or 'unknown'


def collect_real_images_by_class(data_path, split_names=None):
    if split_names is None:
        split_names = ['train', 'val', 'test']
    split_datasets = {}
    class_name_to_label = {}
    split_roots = {
        'train': os.path.join(data_path, 'train'),
        'val': os.path.join(data_path, 'val'),
        'test': os.path.join(data_path, 'test'),
    }

    for split_name in split_names:
        split_root = split_roots[split_name]
        if not os.path.isdir(split_root):
            continue
        split_datasets[split_name] = DatasetFolder(
            root=split_root,
            loader=pil_loader_rgb,
            extensions=IMG_EXTENSIONS,
        )

    if 'train' in split_datasets:
        class_name_to_label.update(split_datasets['train'].class_to_idx)
    else:
        class_names = set()
        for dataset in split_datasets.values():
            class_names.update(dataset.classes)
        for idx, class_name in enumerate(sorted(class_names)):
            class_name_to_label[class_name] = idx

    real_images_by_class = defaultdict(list)
    skipped_unknown = 0

    for split_name in split_names:
        dataset = split_datasets.get(split_name)
        if dataset is None:
            continue
        for sample_path, class_idx in dataset.samples:
            class_name = dataset.classes[class_idx]
            mapped_label = class_name_to_label.get(class_name)
            if mapped_label is None:
                skipped_unknown += 1
                continue
            real_images_by_class[mapped_label].append(sample_path)

    if skipped_unknown > 0:
        print(f"⚠️ 跳过 {skipped_unknown} 个无法映射类别的真实样本")

    return real_images_by_class


def compute_per_class_fid(args, generated_images_by_class, real_images_by_class, output_dir, image_size=256):
    try:
        from cleanfid import fid
    except ImportError:
        print("⚠️ clean-fid 未安装，跳过 per-class FID")
        return None

    per_class_results = {}
    temp_root = os.path.join(output_dir, "temp_per_class_fid")
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)
    os.makedirs(temp_root, exist_ok=True)

    rng = random.Random(args.seed)

    for class_id in range(args.num_classes):
        class_key = str(class_id)
        gen_paths = generated_images_by_class.get(class_id, [])
        real_paths = real_images_by_class.get(class_id, [])

        if len(gen_paths) < 2 or len(real_paths) < 2:
            per_class_results[class_key] = {
                'fid': None,
                'reason': f'insufficient samples: generated={len(gen_paths)}, real={len(real_paths)}, requires >=2 for each',
                'num_generated': len(gen_paths),
                'num_real': len(real_paths),
            }
            continue

        if len(real_paths) > len(gen_paths):
            selected_real_paths = rng.sample(real_paths, len(gen_paths))
        else:
            selected_real_paths = real_paths

        class_temp_dir = os.path.join(temp_root, f'class_{class_id:03d}')
        class_gen_dir = os.path.join(class_temp_dir, 'generated')
        class_real_dir = os.path.join(class_temp_dir, 'real')
        os.makedirs(class_gen_dir, exist_ok=True)
        os.makedirs(class_real_dir, exist_ok=True)

        try:
            for idx, src_path in enumerate(gen_paths):
                with PImage.open(src_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = img.resize((image_size, image_size), PImage.Resampling.LANCZOS)
                    img.save(os.path.join(class_gen_dir, f'gen_{idx:05d}.png'), 'PNG')

            for idx, src_path in enumerate(selected_real_paths):
                with PImage.open(src_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = img.resize((image_size, image_size), PImage.Resampling.LANCZOS)
                    img.save(os.path.join(class_real_dir, f'real_{idx:05d}.png'), 'PNG')

            class_fid = fid.compute_fid(class_gen_dir, class_real_dir, mode='clean', num_workers=0)
            per_class_results[class_key] = {
                'fid': float(class_fid),
                'reason': None,
                'num_generated': len(gen_paths),
                'num_real': len(selected_real_paths),
            }
        except Exception as e:
            per_class_results[class_key] = {
                'fid': None,
                'reason': f'per-class FID failed: {e}',
                'num_generated': len(gen_paths),
                'num_real': len(selected_real_paths),
            }

    per_class_fid_path = os.path.join(output_dir, 'per_class_fid.json')
    with open(per_class_fid_path, 'w', encoding='utf-8') as f:
        json.dump(per_class_results, f, ensure_ascii=False, indent=2)

    print(f"💾 每类别FID结果已保存到: {per_class_fid_path}")
    print(f"💾 per-class FID临时目录已保留: {temp_root}")
    return per_class_results


MODEL_DEPTH = args.depth
assert MODEL_DEPTH in {16, 20, 24, 30, 36}

################## 1. 加载检查点和构建模型

# 模型检查点路径
vae_ckpt = args.vae_path
var_ckpt = args.model_path

# 下载VAE检查点（如果不存在）
hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
if not osp.exists(vae_ckpt):
    vae_dir = os.path.dirname(vae_ckpt)
    if vae_dir:
        os.makedirs(vae_dir, exist_ok=True)
    print(f"⬇️ 正在下载VAE检查点...")
    os.system(f'wget {hf_home}/vae_ch160v4096z32.pth -O {vae_ckpt}')

# 检查VAR检查点是否存在
if not osp.exists(var_ckpt):
    print(f"❌ 找不到检查点文件: {var_ckpt}")
    print("请确保已完成模型训练并且检查点存在于指定目录中")
    exit(1)

# 构建VAR_convMem模型
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

print("🔧 构建VAR_convMem模型...")

# 解析纹理和记忆参数
texture_scales = list(map(int, args.texture_scales.replace('-', '_').split('_'))) if args.enable_texture else [3, 5, 7, 11]
texture_enable_layers = list(map(int, args.texture_enable_layers.replace('-', '_').split('_'))) if args.texture_enable_layers else None
memory_enable_layers = list(map(int, args.memory_enable_layers.replace('-', '_').split('_'))) if args.memory_enable_layers else None
memory_num_categories = args.memory_num_categories if args.memory_num_categories is not None else args.num_classes

vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    device=device, patch_nums=patch_nums,
    num_classes=args.num_classes, depth=MODEL_DEPTH, shared_aln=False,
    attn_l2_norm=True, flash_if_available=True, fused_if_available=True,
    # 纹理增强参数
    enable_texture=args.enable_texture,
    texture_scales=texture_scales,
    texture_enable_layers=texture_enable_layers,
    texture_per_head_kernels=args.texture_per_head_kernels,
    # 记忆功能参数
    enable_memory=args.enable_memory,
    memory_num_patterns=args.memory_num_patterns,
    memory_size=args.memory_size,
    memory_enable_layers=memory_enable_layers,
    use_class_aware_memory=args.mem_class_aware,
    num_categories=memory_num_categories,
    cat_rank=args.memory_cat_rank,
)
print("✅ 模型构建完成")
import sys; sys.stdout.flush()

# 加载检查点
print("📥 加载检查点...")
try:
    checkpoint = torch.load(var_ckpt, map_location='cpu')
    print(f"✅ 成功加载检查点: {var_ckpt}")
    print(f"检查点包含的键: {list(checkpoint.keys())}")

    # 提取模型状态字典
    if 'trainer' in checkpoint:
        model_state_dict = checkpoint['trainer']

        # 加载VAE权重
        if 'vae_local' in model_state_dict:
            vae.load_state_dict(model_state_dict['vae_local'], strict=True)
            print("✅ VAE权重加载成功")

        # 加载VAR权重
        if 'var_wo_ddp' in model_state_dict:
            missing_keys, unexpected_keys = var.load_state_dict(
                model_state_dict['var_wo_ddp'], strict=False  # strict=False to allow for new modules
            )
            print(f"✅ VAR权重加载成功")
            if missing_keys:
                print(f"⚠️  缺失键: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️  意外键: {unexpected_keys}")

    else:
        print("❌ 检查点格式不匹配，请检查训练过程")
        exit(1)

except Exception as e:
    print(f"❌ 加载检查点失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("🔧 设置模型为评估模式...")
vae.eval()
var.eval()
for p in vae.parameters():
    p.requires_grad_(False)
for p in var.parameters():
    p.requires_grad_(False)

# 设置memory temperature为训练结束后的值（0.15）
if args.enable_memory:
    for block in var.blocks:
        if hasattr(block.attn, 'knitting_memory'):
            block.attn.knitting_memory.override_temperature = 0.15
    print("✅ Memory temperature 设置为 0.15")

print("✅ 模型准备完成")
local_prior_model = build_local_prior_model(device=device)
label_to_class_name = build_label_to_class_name(args.data_path, args.num_classes, split_names=FID_SPLITS)
print(f"✅ 类别命名映射: {label_to_class_name}")

################## 2. 生成演示图像

# 设置随机种子
seed = args.seed
print(f'🎲 随机种子: {seed}')
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 生成参数
cfg = args.cfg
more_smooth = True

# 优化设置
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

# 生成演示图像 - 每个类别生成一个样本

demo_classes = list(range(min(args.num_classes, 64)))  # 最多生成64个类别
B = len(demo_classes)
label_B = torch.tensor(demo_classes, device=device)

print(f"🎨 开始生成演示图像...")
print(f"   - 批次大小: {B}")
print(f"   - 类别标签: 前{B}个类别")
print(f"   - CFG强度: {cfg}")

# 生成图像
try:
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
            print("🎨 正在生成图像...")
            if local_prior_model is None or args.local_prior_candidates <= 1:
                recon_B3HW = var.autoregressive_infer_cfg(
                    B=B, label_B=label_B,
                    cfg=cfg, top_k=args.top_k, top_p=args.top_p,
                    g_seed=seed, more_smooth=more_smooth
                )
            else:
                cand_imgs = []
                cand_scores = []
                for ci in range(args.local_prior_candidates):
                    cand = var.autoregressive_infer_cfg(
                        B=B, label_B=label_B,
                        cfg=cfg, top_k=args.top_k, top_p=args.top_p,
                        g_seed=seed + ci, more_smooth=more_smooth
                    )
                    score = score_batch_with_local_prior(
                        local_prior_model, cand,
                        patch_size=args.local_prior_patch_size,
                        num_patches=args.local_prior_num_patches,
                    ) * args.local_prior_weight
                    cand_imgs.append(cand)
                    cand_scores.append(score)
                score_stack = torch.stack(cand_scores, dim=0)
                best_idx = score_stack.argmax(dim=0)
                recon_B3HW = torch.stack([cand_imgs[best_idx[i].item()][i] for i in range(B)], dim=0)

    print("✅ 图像生成完成")

    # 创建网格图像
    chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=2, pad_value=1.0)
    chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8)
    chw = PImage.fromarray(chw)

    # 保存结果
    feature_suffix = ""
    if args.enable_texture:
        feature_suffix += "_tex"
    if args.enable_memory:
        feature_suffix += "_mem"

    output_path = os.path.join(args.output_dir, f"var_convmem{feature_suffix}_d{MODEL_DEPTH}_cfg{cfg}_seed{seed}.png")
    chw.save(output_path)
    print(f"💾 演示结果已保存到: {output_path}")

    demo_individual_dir = os.path.join(args.output_dir, f"demo_images{feature_suffix}_d{MODEL_DEPTH}_cfg{cfg}_seed{seed}")
    if os.path.exists(demo_individual_dir):
        shutil.rmtree(demo_individual_dir)
    os.makedirs(demo_individual_dir, exist_ok=True)

    for i, class_id in enumerate(demo_classes):
        class_name = label_to_class_name.get(class_id, f'class_{class_id:03d}')
        class_name_safe = sanitize_class_name(class_name)
        demo_img = recon_B3HW[i].permute(1, 2, 0).mul(255).cpu().numpy().astype(np.uint8)
        demo_img = PImage.fromarray(demo_img)
        demo_path = os.path.join(demo_individual_dir, f"class{class_id:03d}_{class_name_safe}_demo.png")
        demo_img.save(demo_path)
    print(f"💾 各类别演示图已保存到: {demo_individual_dir}")

except Exception as e:
    print(f"❌ 生成过程出错: {e}")
    import traceback
    traceback.print_exc()

if args.demo_only:
    print("\n🎉 演示模式完成！")
    exit(0)

################## 3. FID分数计算

print("\n📈 开始计算FID/KID分数...")

try:
    from cleanfid import fid
except ImportError:
    print("❌ 未安装clean-fid，请运行: pip install clean-fid")
    print("跳过FID/KID计算...")
    exit(0)

# 准备目录
generated_dir = os.path.abspath(os.path.join(args.output_dir, "generated_images_for_fid"))
# 合并 train+val+test 作为真实图像参考集
import pathlib
IMAGE_EXTS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}
real_dir = os.path.abspath(os.path.join(args.output_dir, "temp_all_real_images"))
if args.real_images_dir:
    real_dir = os.path.abspath(args.real_images_dir)
    _real_count = sum(1 for _f in pathlib.Path(real_dir).rglob('*') if _f.suffix[1:].lower() in IMAGE_EXTS)
    print(f"Using prebuilt real image directory: {real_dir}")
else:
    if os.path.exists(real_dir):
        shutil.rmtree(real_dir)
    os.makedirs(real_dir, exist_ok=True)
    _real_count = 0
    for _split in FID_SPLITS:
        _split_dir = pathlib.Path(args.data_path) / _split
        if not _split_dir.exists():
            continue
        for _f in _split_dir.rglob('*'):
            if _f.suffix[1:].lower() in IMAGE_EXTS:
                _dest = os.path.join(real_dir, f"real_{_real_count:06d}{_f.suffix}")
                shutil.copy2(str(_f), _dest)
                _real_count += 1
print(f"📂 真实图像合并完成: {_real_count} 张 (train+val+test)")
print(f"📂 生成图像目录（绝对路径）: {generated_dir}")
print(f"📂 真实图像目录（绝对路径）: {real_dir}")

# 创建生成图像目录
os.makedirs(generated_dir, exist_ok=True)

# 生成样本用于FID计算
num_samples_for_fid = args.num_samples
batch_size = args.batch_size

print(f"🎨 为FID计算生成 {num_samples_for_fid} 个样本...")

# 准备类别标签 - 均匀分布在所有类别上
generation_labels = []
samples_per_class = (num_samples_for_fid + args.num_classes - 1) // args.num_classes

for class_id in range(args.num_classes):
    generation_labels.extend([class_id] * samples_per_class)
generation_labels = generation_labels[:num_samples_for_fid]

# 打乱顺序以避免批次偏差
random.shuffle(generation_labels)
generated_images_by_class = defaultdict(list)
generated_counts_by_class = defaultdict(int)

num_batches = (num_samples_for_fid + batch_size - 1) // batch_size
sample_count = 0
current_seed = seed  # 使用连续递增的种子

existing_generated = sorted(glob.glob(os.path.join(generated_dir, "*.png")))
existing_count = len(existing_generated)
for existing_path in existing_generated:
    base_name = os.path.basename(existing_path)
    if base_name.startswith('class') and len(base_name) >= 8:
        try:
            class_id = int(base_name[5:8])
            generated_images_by_class[class_id].append(existing_path)
            generated_counts_by_class[class_id] += 1
        except ValueError:
            pass

if existing_count >= num_samples_for_fid:
    print(f"Existing generated images already reach target: {existing_count}; skip regeneration.")
else:
    print(f"Existing generated images: {existing_count}; will continue until {num_samples_for_fid}.")

sample_count = existing_count
current_seed = seed + existing_count * max(args.local_prior_candidates, 1)
remaining_to_generate = max(num_samples_for_fid - sample_count, 0)
num_batches = (remaining_to_generate + batch_size - 1) // batch_size if remaining_to_generate > 0 else 0

for batch_idx in range(num_batches):
    current_batch_size = min(batch_size, num_samples_for_fid - sample_count)

    # 使用预定义的类别标签
    batch_labels = generation_labels[sample_count:sample_count + current_batch_size]
    label_batch = torch.tensor(batch_labels, device=device)

    try:
        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
                if local_prior_model is None or args.local_prior_candidates <= 1:
                    generated_batch = var.autoregressive_infer_cfg(
                        B=current_batch_size, label_B=label_batch,
                        cfg=cfg, top_k=args.top_k, top_p=args.top_p,
                        g_seed=current_seed, more_smooth=more_smooth
                    )
                else:
                    cand_imgs = []
                    cand_scores = []
                    for ci in range(args.local_prior_candidates):
                        cand = var.autoregressive_infer_cfg(
                            B=current_batch_size, label_B=label_batch,
                            cfg=cfg, top_k=args.top_k, top_p=args.top_p,
                            g_seed=current_seed + ci, more_smooth=more_smooth
                        )
                        score = score_batch_with_local_prior(
                            local_prior_model, cand,
                            patch_size=args.local_prior_patch_size,
                            num_patches=args.local_prior_num_patches,
                        ) * args.local_prior_weight
                        cand_imgs.append(cand)
                        cand_scores.append(score)
                    score_stack = torch.stack(cand_scores, dim=0)
                    best_idx = score_stack.argmax(dim=0)
                    generated_batch = torch.stack([cand_imgs[best_idx[i].item()][i] for i in range(current_batch_size)], dim=0)

        # 每个批次后种子递增，确保不同且连续
        current_seed += max(args.local_prior_candidates, 1)

        # 保存单个图像
        for i in range(current_batch_size):
            img_tensor = generated_batch[i].clone()
            # 转换为PIL图像
            img_np = img_tensor.permute(1, 2, 0).mul(255).cpu().numpy().astype(np.uint8)
            img_pil = PImage.fromarray(img_np)

            # 保存图像
            class_id = int(batch_labels[i])
            class_name = label_to_class_name.get(class_id, f'class_{class_id:03d}')
            class_name_safe = sanitize_class_name(class_name)
            class_local_count = generated_counts_by_class[class_id]
            img_path = os.path.join(
                generated_dir,
                f"class{class_id:03d}_{class_name_safe}_{class_local_count:05d}_g{sample_count:05d}.png"
            )
            img_pil.save(img_path)
            generated_images_by_class[class_id].append(img_path)
            generated_counts_by_class[class_id] += 1
            sample_count += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"   批次 {batch_idx + 1}/{num_batches} 完成 ({sample_count}/{num_samples_for_fid}), seed: {current_seed}")

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ 批次 {batch_idx + 1} 生成失败: {e}")
        current_seed += 1  # 即使失败也递增种子保持连续性
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        continue

print(f"✅ 共生成 {sample_count} 个样本")

# 检查数据集目录
if not os.path.exists(real_dir):
    print(f"❌ 真实数据集目录不存在: {real_dir}")
    print("请确保数据集目录存在并包含图像文件")
    print(f"提示: 如果使用训练集，请将 --data_path 参数指向包含 train 或 val 子目录的ImageNet根目录")
else:
    # 计算FID分数
    try:
        print(f"\n🧮 计算FID分数...")
        print(f"   生成图像目录: {generated_dir}")
        print(f"   真实图像目录: {real_dir}")

        # 检查生成的图像数量
        generated_images = glob.glob(os.path.join(generated_dir, "*.png"))
        print(f"   生成图像数量: {len(generated_images)}")

        if len(generated_images) == 0:
            print("❌ 没有生成的图像用于FID计算")
        else:
            # 创建临时目录处理中文文件名问题
            print(f"📝 准备真实图像（处理中文文件名）...")
            use_prebuilt_real_dir = bool(args.real_images_dir)
            temp_real_dir = real_dir if use_prebuilt_real_dir else os.path.join(args.output_dir, "temp_real_images_fid")
            if not use_prebuilt_real_dir:
                if os.path.exists(temp_real_dir):
                    shutil.rmtree(temp_real_dir)
                os.makedirs(temp_real_dir, exist_ok=True)

            # 收集所有真实图像
            import pathlib
            IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}
            real_path = pathlib.Path(real_dir)
            all_real_files = sorted([f for f in real_path.rglob('*') if f.suffix[1:].lower() in IMAGE_EXTENSIONS])
            print(f"   找到真实图像: {len(all_real_files)} 张")

            # 复制到临时目录（使用数字命名避免中文编码问题）
            import PIL.Image as Image
            copied_count = len(all_real_files) if use_prebuilt_real_dir else 0
            target_size = (256, 256)

            # 选择与生成图像相同数量的真实图像
            if use_prebuilt_real_dir:
                selected_files = []
            elif len(all_real_files) > len(generated_images):
                import random
                random.seed(args.seed)
                selected_files = random.sample(all_real_files, len(generated_images))
            else:
                selected_files = all_real_files

            for idx, img_path in enumerate(selected_files):
                try:
                    with Image.open(img_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img = img.resize(target_size, Image.Resampling.LANCZOS)
                        dest_path = os.path.join(temp_real_dir, f"real_{copied_count:05d}.png")
                        img.save(dest_path, 'PNG')
                        copied_count += 1
                except Exception as e:
                    print(f"   警告: 处理图像失败: {e}")
                    continue

            print(f"   已复制 {copied_count} 张真实图像到临时目录")

            real_images_by_class = None
            if args.skip_per_class_fid:
                print("⏭️ 已跳过每类别 FID 计算")
            else:
                real_images_by_class = collect_real_images_by_class(args.data_path, split_names=FID_SPLITS)

            # 使用 clean-fid 计算 FID
            print(f"\n🧮 使用 clean-fid 计算 FID...")
            fid_value = fid.compute_fid(generated_dir, temp_real_dir, mode="clean", num_workers=0)
            print(f"\n🎯 FID分数: {fid_value:.4f}")

            # 使用 clean-fid 计算 KID
            print(f"\n🧮 使用 clean-fid 计算 KID...")
            kid_value = fid.compute_kid(generated_dir, temp_real_dir, mode="clean", num_workers=0)
            print(f"🎯 KID分数: {kid_value:.6f}")

            # 清理临时目录
            if not use_prebuilt_real_dir:
                shutil.rmtree(temp_real_dir)

            # 计算每类别FID
            if args.skip_per_class_fid:
                print(f"\n⏭️ 跳过每类别 FID 计算")
            else:
                print(f"\n🧮 计算每类别 FID...")
                compute_per_class_fid(
                    args=args,
                    generated_images_by_class=generated_images_by_class,
                    real_images_by_class=real_images_by_class,
                    output_dir=args.output_dir,
                )

            # 计算额外的评估指标
            print(f"\n📊 计算额外评估指标...")
            evaluation_results = {'FID': fid_value, 'KID': kid_value}
            gen_img_list = sorted(glob.glob(os.path.join(generated_dir, "*.png")))
            real_img_list = []

            # 1. 计算LPIPS
            try:
                # 获取生成图像和真实图像的路径列表
                # 重新收集真实图像（用于LPIPS计算）
                import pathlib
                IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}
                real_path = pathlib.Path(real_dir)
                real_img_list = sorted([str(f) for f in real_path.rglob('*') if f.suffix[1:].lower() in IMAGE_EXTENSIONS])

                # 确保数量匹配
                if len(real_img_list) > len(gen_img_list):
                    random.seed(args.seed)
                    real_img_list = random.sample(real_img_list, len(gen_img_list))

                lpips_mean, lpips_std = calculate_lpips(gen_img_list, real_img_list, device, sample_size=500)
                if lpips_mean is not None:
                    evaluation_results['LPIPS'] = {'mean': lpips_mean, 'std': lpips_std}
            except Exception as e:
                print(f"⚠️ LPIPS计算失败: {e}")

            # 2. 计算SSIM
            try:
                ssim_mean, ssim_std = calculate_ssim(gen_img_list, real_img_list, sample_size=500)
                if ssim_mean is not None:
                    evaluation_results['SSIM'] = {'mean': ssim_mean, 'std': ssim_std}
            except Exception as e:
                print(f"⚠️ SSIM计算失败: {e}")

            # 保存完整评估结果到文件
            feature_suffix = ""
            if args.enable_texture:
                feature_suffix += "_tex"
            if args.enable_memory:
                feature_suffix += "_mem"

            result_file = os.path.join(args.output_dir, f"evaluation_results{feature_suffix}_d{MODEL_DEPTH}_cfg{cfg}_seed{seed}.txt")
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write("=================== VAR_convMem 评估结果汇总 ===================\n")
                f.write(f"Model Depth: {MODEL_DEPTH}\n")
                f.write(f"Seed: {seed}\n")
                f.write(f"Generated Samples: {sample_count}\n")
                f.write(f"CFG Scale: {cfg}\n")
                f.write(f"Top-k: {args.top_k}\n")
                f.write(f"Top-p: {args.top_p}\n")
                f.write(f"Num Classes: {args.num_classes}\n")
                f.write("\n🎨 VAR_convMem 特性:\n")
                f.write(f"  Texture Enhancement: {args.enable_texture}\n")
                if args.enable_texture:
                    f.write(f"    - Scales: {texture_scales}\n")
                    f.write(f"    - Enable Layers: {texture_enable_layers}\n")
                    f.write(f"    - Per-head Kernels: {args.texture_per_head_kernels}\n")
                f.write(f"  Pattern Memory: {args.enable_memory}\n")
                if args.enable_memory:
                    f.write(f"    - Num Patterns: {args.memory_num_patterns}\n")
                    f.write(f"    - Memory Size: {args.memory_size}\n")
                    f.write(f"    - Enable Layers: {memory_enable_layers}\n")
                f.write("\n📊 图像质量指标:\n")
                f.write(f"  FID Score: {fid_value:.4f}\n")
                f.write(f"  KID Score: {kid_value:.6f}\n")

                if 'LPIPS' in evaluation_results:
                    lpips_data = evaluation_results['LPIPS']
                    f.write(f"  LPIPS: {lpips_data['mean']:.4f} ± {lpips_data['std']:.4f}\n")

                if 'SSIM' in evaluation_results:
                    ssim_data = evaluation_results['SSIM']
                    f.write(f"  SSIM: {ssim_data['mean']:.4f} ± {ssim_data['std']:.4f}\n")

                f.write("================================================================\n")

            print(f"\n💾 完整评估结果已保存到: {result_file}")

            # 打印摘要
            print("\n" + "="*60)
            print("📊 VAR_convMem 评估结果摘要")
            print("="*60)
            print(f"FID Score: {fid_value:.4f}")
            print(f"KID Score: {kid_value:.6f}")
            if 'LPIPS' in evaluation_results:
                print(f"LPIPS: {evaluation_results['LPIPS']['mean']:.4f} ± {evaluation_results['LPIPS']['std']:.4f}")
            if 'SSIM' in evaluation_results:
                print(f"SSIM: {evaluation_results['SSIM']['mean']:.4f} ± {evaluation_results['SSIM']['std']:.4f}")
            print("="*60)

    except Exception as e:
        print(f"❌ FID计算失败: {e}")
        import traceback
        traceback.print_exc()
        print("请确保真实数据集目录包含有效的图像文件")

print("\n🎉 测试完成！")
print(f"📊 生成了 {sample_count} 个样本")
print(f"💾 结果保存在: {args.output_dir}")
