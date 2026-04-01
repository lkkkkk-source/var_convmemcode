from typing import Tuple, Optional, List
import torch.nn as nn

from .quant import VectorQuantizer2
from .var import VAR
from .vqvae import VQVAE


def build_vae_var(
    # Shared args
    device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
    # VQVAE args
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    # VAR args
    num_classes=1000, depth=16, shared_aln=False, attn_l2_norm=True,
    flash_if_available=True, fused_if_available=True,
    init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,    # init_std < 0: automated
    drop_rate=0., attn_drop_rate=0., drop_path_rate=None,  # dropout / drop path
    # Axial texture enhancement args
    enable_texture: bool = False,
    texture_scales: List[int] = [3, 5, 7, 11],
    texture_enable_layers: Optional[List[int]] = None,
    texture_per_head_kernels: bool = False,
    # Knitting pattern memory args
    enable_memory: bool = False,
    memory_num_patterns: int = 16,
    memory_size: int = 8,
    memory_enable_layers: Optional[List[int]] = None,
    # Class-aware memory args
    use_class_aware_memory: bool = False,
    num_categories: int = 22,
    cat_rank: int = 4,
    # Auxiliary classification head
    aux_cls_tap_layer: int = 4,
) -> Tuple[VQVAE, VAR]:
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth/24
    
    # disable built-in initialization for speed
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)
    
    # build models
    vae_local = VQVAE(vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi, v_patch_nums=patch_nums).to(device)
    var_wo_ddp = VAR(
        vae_local=vae_local,
        num_classes=num_classes, depth=depth, embed_dim=width, num_heads=heads, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=(drop_path_rate if drop_path_rate is not None else dpr),
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        # Axial texture enhancement
        enable_texture=enable_texture,
        texture_scales=texture_scales,
        texture_enable_layers=texture_enable_layers,
        texture_per_head_kernels=texture_per_head_kernels,
        # Knitting pattern memory
        enable_memory=enable_memory,
        memory_num_patterns=memory_num_patterns,
        memory_size=memory_size,
        memory_enable_layers=memory_enable_layers,
        # Class-aware memory
        use_class_aware_memory=use_class_aware_memory,
        num_categories=num_categories,
        cat_rank=cat_rank,
        # Auxiliary classification head
        aux_cls_tap_layer=aux_cls_tap_layer,
    ).to(device)
    var_wo_ddp.init_weights(init_adaln=init_adaln, init_adaln_gamma=init_adaln_gamma, init_head=init_head, init_std=init_std)
    
    return vae_local, var_wo_ddp
