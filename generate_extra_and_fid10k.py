"""Generate 5600 additional samples and compute FID with 10000 total."""
import os, sys, glob, shutil, random, pathlib
import numpy as np
import torch
import PIL.Image as PImage

setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

from models import VQVAE, build_vae_var

# === Config (matching fined_v1.3_stable_lr3e4 training) ===
OUTPUT_DIR = './evaluation_results_ined_v1.3_stable_lr3e4'
MODEL_PATH = './local_output/fined_v1.3_stable_lr3e4/ar-ckpt-best.pth'
VAE_PATH = '../model_path/vae_ch160v4096z32.pth'
DATA_PATH = './dataset_v3_patches'
DEPTH = 16
NUM_CLASSES = 6
CFG = 1.5
TOP_K = 900
TOP_P = 0.96
BATCH_SIZE = 32
SEED = 0

# How many we already have and how many more to generate
EXISTING = 4400
EXTRA = 5600
TOTAL = EXISTING + EXTRA

generated_dir = os.path.join(OUTPUT_DIR, 'generated_images_for_fid')
existing_count = len(glob.glob(os.path.join(generated_dir, '*.png')))
print(f"Existing generated images: {existing_count}")

if existing_count < EXISTING:
    print(f"ERROR: Expected {EXISTING} existing images, found {existing_count}")
    sys.exit(1)

# === Build model ===
device = 'cuda'
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

texture_scales = [3, 5, 7, 11]
texture_enable_layers = [12, 13, 14, 15]
memory_enable_layers = [12]

print("Building model...")
vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    device=device, patch_nums=patch_nums,
    num_classes=NUM_CLASSES, depth=DEPTH, shared_aln=False,
    attn_l2_norm=True, flash_if_available=True, fused_if_available=True,
    enable_texture=True, texture_scales=texture_scales,
    texture_enable_layers=texture_enable_layers, texture_per_head_kernels=False,
    enable_memory=True, memory_num_patterns=4, memory_size=4,
    memory_enable_layers=memory_enable_layers, use_class_aware_memory=False,
)

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
model_state = checkpoint['trainer']
vae.load_state_dict(model_state['vae_local'], strict=True)
var.load_state_dict(model_state['var_wo_ddp'], strict=False)

vae.eval(); var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)

for block in var.blocks:
    if hasattr(block.attn, 'knitting_memory'):
        block.attn.knitting_memory.override_temperature = 0.15

print("Model loaded.")

# === Generate 5600 more samples ===
# The original script used seeds 0..137 for 138 batches of 32.
# Continue from seed 138 and index 4400.
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high')

# Prepare labels for extra samples
random.seed(SEED + 1000)  # different seed for label shuffle to avoid correlation
generation_labels = []
samples_per_class = (EXTRA + NUM_CLASSES - 1) // NUM_CLASSES
for c in range(NUM_CLASSES):
    generation_labels.extend([c] * samples_per_class)
generation_labels = generation_labels[:EXTRA]
random.shuffle(generation_labels)

num_batches = (EXTRA + BATCH_SIZE - 1) // BATCH_SIZE
sample_count = 0
current_seed = 138  # continue from where original left off
file_index = existing_count

print(f"Generating {EXTRA} additional samples...")
for batch_idx in range(num_batches):
    current_batch_size = min(BATCH_SIZE, EXTRA - sample_count)
    batch_labels = generation_labels[sample_count:sample_count + current_batch_size]
    label_batch = torch.tensor(batch_labels, device=device)

    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
            generated_batch = var.autoregressive_infer_cfg(
                B=current_batch_size, label_B=label_batch,
                cfg=CFG, top_k=TOP_K, top_p=TOP_P,
                g_seed=current_seed, more_smooth=True
            )
    current_seed += 1

    for i in range(current_batch_size):
        img_np = generated_batch[i].permute(1, 2, 0).mul(255).cpu().numpy().astype(np.uint8)
        img_pil = PImage.fromarray(img_np)
        img_pil.save(os.path.join(generated_dir, f"generated_{file_index:05d}.png"))
        file_index += 1
        sample_count += 1

    if (batch_idx + 1) % 10 == 0:
        print(f"  Batch {batch_idx+1}/{num_batches} ({sample_count}/{EXTRA})")

    torch.cuda.empty_cache()

total_generated = len(glob.glob(os.path.join(generated_dir, '*.png')))
print(f"Total generated images now: {total_generated}")

# === Compute FID/KID with 10000 samples ===
from cleanfid import fid

# Prepare real images (reuse existing or rebuild)
real_dir = os.path.join(OUTPUT_DIR, 'temp_all_real_images')
if not os.path.exists(real_dir):
    print("Rebuilding real image directory...")
    os.makedirs(real_dir, exist_ok=True)
    IMAGE_EXTS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}
    cnt = 0
    for split in ['train', 'val', 'test']:
        split_dir = pathlib.Path(DATA_PATH) / split
        if not split_dir.exists(): continue
        for f in split_dir.rglob('*'):
            if f.suffix[1:].lower() in IMAGE_EXTS:
                shutil.copy2(str(f), os.path.join(real_dir, f"real_{cnt:06d}{f.suffix}"))
                cnt += 1
    print(f"Real images: {cnt}")

# Create temp real dir with matching count
IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}
real_path = pathlib.Path(real_dir)
all_real_files = sorted([f for f in real_path.rglob('*') if f.suffix[1:].lower() in IMAGE_EXTENSIONS])

temp_real_dir = os.path.join(OUTPUT_DIR, 'temp_real_images_fid_10k')
if os.path.exists(temp_real_dir):
    shutil.rmtree(temp_real_dir)
os.makedirs(temp_real_dir, exist_ok=True)

# Sample real images to match generated count
random.seed(SEED)
if len(all_real_files) > total_generated:
    selected = random.sample(all_real_files, total_generated)
else:
    # If not enough, use all (8700 real vs 10000 gen - use all real)
    selected = all_real_files

print(f"Preparing {len(selected)} real images for FID...")
for idx, img_path in enumerate(selected):
    with PImage.open(img_path) as img:
        if img.mode != 'RGB': img = img.convert('RGB')
        img = img.resize((256, 256), PImage.Resampling.LANCZOS)
        img.save(os.path.join(temp_real_dir, f"real_{idx:05d}.png"), 'PNG')

print(f"\nComputing FID (generated: {total_generated}, real: {len(selected)})...")
fid_value = fid.compute_fid(generated_dir, temp_real_dir, mode="clean", num_workers=0)
print(f"\nFID Score (10k): {fid_value:.4f}")

print(f"\nComputing KID...")
kid_value = fid.compute_kid(generated_dir, temp_real_dir, mode="clean", num_workers=0)
print(f"KID Score (10k): {kid_value:.6f}")

# Cleanup
shutil.rmtree(temp_real_dir)

# Save results
result_file = os.path.join(OUTPUT_DIR, 'evaluation_results_10k.txt')
with open(result_file, 'w') as f:
    f.write(f"FID (10k generated vs {len(selected)} real): {fid_value:.4f}\n")
    f.write(f"KID (10k generated vs {len(selected)} real): {kid_value:.6f}\n")
    f.write(f"Total generated: {total_generated}\n")
    f.write(f"Total real used: {len(selected)}\n")
print(f"\nResults saved to {result_file}")
