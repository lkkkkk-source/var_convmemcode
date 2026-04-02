import gc
import os
import shutil
import subprocess
import sys
import time
import warnings
from functools import partial
from typing import Dict, List, Set, Tuple

import torch
from torch.utils.data import DataLoader

import dist
from utils import arg_util, misc
from utils.data import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.misc import auto_resume


def _looks_like_state_dict(candidate) -> bool:
    if not isinstance(candidate, dict) or len(candidate) == 0:
        return False
    return all(isinstance(k, str) for k in candidate.keys()) and any(torch.is_tensor(v) for v in candidate.values())


def _extract_var_state_dict(ckpt_obj) -> Dict[str, torch.Tensor]:
    if _looks_like_state_dict(ckpt_obj):
        return ckpt_obj

    if isinstance(ckpt_obj, dict):
        for key in ('var_wo_ddp', 'state_dict', 'model', 'module', 'var'):
            value = ckpt_obj.get(key, None)
            if _looks_like_state_dict(value):
                assert isinstance(value, dict)
                return value

        trainer_state = ckpt_obj.get('trainer', None)
        if isinstance(trainer_state, dict):
            value = trainer_state.get('var_wo_ddp', None)
            if _looks_like_state_dict(value):
                assert isinstance(value, dict)
                return value

    raise RuntimeError('Cannot extract VAR state_dict from checkpoint payload.')


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ('module.', '_orig_mod.', 'var.', 'var_wo_ddp.', 'trainer.var_wo_ddp.')
    normalized = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        normalized[new_key] = value
    return normalized


def _parse_freeze_layers_spec(spec: str, depth: int) -> List[int]:
    if not spec:
        return []
    items = [s.strip() for s in spec.replace('-', '_').split('_') if s.strip()]
    if len(items) == 0:
        return []
    try:
        idxs = [int(x) for x in items]
    except ValueError:
        print(f'[INIT][WARN] Invalid --freeze_layers={spec}, skip freezing.')
        return []

    if len(idxs) == 2 and idxs[0] <= idxs[1]:
        idxs = list(range(idxs[0], idxs[1] + 1))

    valid = sorted({i for i in idxs if 0 <= i < depth})
    return valid


def _freeze_backbone_layers(var_model, freeze_layers: List[int]) -> int:
    if not freeze_layers:
        return 0

    keep_trainable_keywords = (
        'knitting_memory',
        'texture_',
        'row_',
        'col_',
        'diag_',
        'tex_',
        'memory_per_scale',
        'shared_memory',
        'category_memories',
        'category_embedding',
    )

    frozen_count = 0
    freeze_layer_set = set(freeze_layers)
    for name, para in var_model.named_parameters():
        if not name.startswith('blocks.'):
            continue
        parts = name.split('.')
        if len(parts) < 2:
            continue
        try:
            layer_idx = int(parts[1])
        except ValueError:
            continue
        if layer_idx not in freeze_layer_set:
            continue
        if any(k in name for k in keep_trainable_keywords):
            continue
        if para.requires_grad:
            para.requires_grad = False
            frozen_count += 1
    return frozen_count


def _apply_finetune_lr_scale(
    para_groups: List[dict],
    pretrained_param_ids: Set[int],
    finetune_lr_scale: float,
) -> Tuple[List[dict], int]:
    if finetune_lr_scale == 1.0 or len(pretrained_param_ids) == 0:
        return para_groups, 0

    scaled_groups = []
    scaled_param_count = 0
    for group in para_groups:
        pre_params, new_params = [], []
        for para in group['params']:
            if id(para) in pretrained_param_ids:
                pre_params.append(para)
            else:
                new_params.append(para)

        if pre_params:
            g_pre = dict(group)
            g_pre['params'] = pre_params
            g_pre['lr_sc'] = g_pre.get('lr_sc', 1.0) * finetune_lr_scale
            scaled_groups.append(g_pre)
            scaled_param_count += len(pre_params)

        if new_params:
            g_new = dict(group)
            g_new['params'] = new_params
            scaled_groups.append(g_new)

    return scaled_groups, scaled_param_count


def build_everything(args: arg_util.Args):
    # resume
    auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    # create tensorboard logger
    tb_lg: misc.TensorboardLogger
    with_tb_lg = dist.is_master()
    if with_tb_lg:
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(misc.TensorboardLogger(log_dir=args.tb_log_dir_path, filename_suffix=f'__{misc.time_str("%m%d_%H%M")}'), verbose=True)
        tb_lg.flush()
    else:
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(None, verbose=False)
    dist.barrier()
    
    # log args
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')
    
    # build data
    if not args.local_debug:
        print(f'[build PT data] ...\n')
        num_classes, dataset_train, dataset_val = build_dataset(
            args.data_path, final_reso=args.data_load_reso, hflip=args.hflip, mid_reso=args.mid_reso,
            cyclic_shift=args.cyclic_shift,
            vflip=args.vflip,
            rand_rot=args.rand_rot,
            color_jitter=args.color_jitter,
        )
        types = str((type(dataset_train).__name__, type(dataset_val).__name__))
        
        ld_val = DataLoader(
            dataset_val, num_workers=0, pin_memory=True,
            batch_size=round(args.batch_size*1.5), sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
            shuffle=False, drop_last=False,
        )
        del dataset_val
        
        ld_train = DataLoader(
            dataset=dataset_train, num_workers=args.workers, pin_memory=True,
            generator=args.get_different_generator_for_each_rank(), # worker_init_fn=worker_init_fn,
            batch_sampler=DistInfiniteBatchSampler(
                dataset_len=len(dataset_train), glb_batch_size=args.glb_batch_size, same_seed_for_all_ranks=args.same_seed_for_all_ranks,
                shuffle=True, fill_last=True, rank=dist.get_rank(), world_size=dist.get_world_size(), start_ep=start_ep, start_it=start_it,
            ),
        )
        del dataset_train
        
        [print(line) for line in auto_resume_info]
        print(f'[dataloader multi processing] ...', end='', flush=True)
        stt = time.time()
        iters_train = len(ld_train)
        ld_train = iter(ld_train)
        # noinspection PyArgumentList
        print(f'     [dataloader multi processing](*) finished! ({time.time()-stt:.2f}s)', flush=True, clean=True)
        print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}, types(tr, va)={types}')
    
    else:
        num_classes = 1000
        ld_val = ld_train = None
        iters_train = 10
    
    # build models
    from torch.nn.parallel import DistributedDataParallel as DDP
    from models import VAR, VQVAE, build_vae_var
    from trainer import VARTrainer
    from utils.amp_sc import AmpOptimizer
    from utils.lr_control import filter_params
    
    vae_local, var_wo_ddp = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
        device=dist.get_device(), patch_nums=args.patch_nums,
        num_classes=num_classes, depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini,
        drop_rate=args.drop, attn_drop_rate=args.drop, drop_path_rate=args.drop_path,
        # Axial texture enhancement
        enable_texture=args.tex,
        texture_scales=list(map(int, args.tex_scales.replace('-', '_').split('_'))) if args.tex else [3, 5, 7, 11],
        texture_enable_layers=list(map(int, args.tex_layers.replace('-', '_').split('_'))) if args.tex_layers else None,
        texture_per_head_kernels=args.tex_per_head,
        # Knitting pattern memory
        enable_memory=args.mem,
        memory_num_patterns=args.mem_patterns,
        memory_size=args.mem_size,
        memory_enable_layers=list(map(int, args.mem_layers.replace('-', '_').split('_'))) if args.mem_layers else None,
        # Class-aware memory
        use_class_aware_memory=args.mem_class_aware,
        num_categories=args.mem_num_categories,
        cat_rank=args.mem_cat_rank,
        # Auxiliary classification head
        aux_cls_tap_layer=args.aux_tap_layer,
    )
    
    vae_ckpt = './model_path/vae_ch160v4096z32.pth'
    vae_url = 'https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth'
    if dist.is_local_master():
        os.makedirs(os.path.dirname(vae_ckpt), exist_ok=True)
        need_download = (not os.path.exists(vae_ckpt)) or os.path.getsize(vae_ckpt) == 0
        if not need_download:
            try:
                torch.load(vae_ckpt, map_location='cpu')
            except Exception:
                need_download = True
        if need_download:
            try:
                subprocess.run(
                    ['wget', '-q', '-O', vae_ckpt, vae_url],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f'Failed to download VAE checkpoint: {e}') from e
    dist.barrier()
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    
    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    var_wo_ddp: VAR = args.compile_model(var_wo_ddp, args.tfast)

    # Load pretrained VAR checkpoint for finetuning (only when not auto-resuming training state)
    pretrained_loaded_param_ids: Set[int] = set()
    did_load_pretrained = False
    if args.pretrained_ckpt and not (trainer_state is not None and len(trainer_state)):
        if not os.path.exists(args.pretrained_ckpt):
            print(f'[INIT][Finetune][WARN] --pretrained_ckpt not found: {args.pretrained_ckpt}; skip pretrained loading.')
        else:
            raw_ckpt = torch.load(args.pretrained_ckpt, map_location='cpu')
            raw_var_sd = _extract_var_state_dict(raw_ckpt)
            pretrained_sd = _normalize_state_dict_keys(raw_var_sd)

            model_sd = var_wo_ddp.state_dict()
            loadable_sd = {}
            mismatch = []
            for k, v in pretrained_sd.items():
                if k in model_sd:
                    if model_sd[k].shape == v.shape:
                        loadable_sd[k] = v
                    else:
                        mismatch.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

            missing = [k for k in model_sd.keys() if k not in loadable_sd]
            unexpected = [k for k in pretrained_sd.keys() if k not in model_sd]
            var_wo_ddp.load_state_dict(loadable_sd, strict=False)

            param_names = {name for name, _ in var_wo_ddp.named_parameters()}
            pretrained_loaded_param_ids = {
                id(param)
                for name, param in var_wo_ddp.named_parameters()
                if name in loadable_sd
            }
            loaded_param_names = {name for name in loadable_sd.keys() if name in param_names}
            did_load_pretrained = len(loaded_param_names) > 0

            print(
                f'[INIT][Finetune] Loaded pretrained VAR from {args.pretrained_ckpt}\n'
                f'  loaded_keys={len(loadable_sd)}, missing_keys={len(missing)}, '
                f'unexpected_keys={len(unexpected)}, shape_mismatch={len(mismatch)}\n'
                f'  loaded_parameters={len(loaded_param_names)}'
            )
    elif args.pretrained_ckpt:
        print('[INIT][Finetune] Auto-resume detected: skip loading --pretrained_ckpt (resume state has priority).')

    # Freeze selected backbone blocks while keeping newly-added modules trainable
    freeze_layers = _parse_freeze_layers_spec(args.freeze_layers, args.depth)
    should_apply_freeze = bool(freeze_layers) and (did_load_pretrained or (trainer_state is not None and len(trainer_state)))
    if should_apply_freeze:
        frozen_count = _freeze_backbone_layers(var_wo_ddp, freeze_layers)
        print(f'[INIT][Finetune] Frozen backbone params in blocks {freeze_layers}: {frozen_count} parameters.')
    elif freeze_layers:
        print('[INIT][Finetune][WARN] freeze_layers is set but pretrained weights are not loaded; skip freezing.')

    # Note: find_unused_parameters=True is needed because:
    # 1. Texture enhancement is only enabled in some layers (second half by default)
    # 2. Memory bank is only enabled in specific layers [0, depth//4, depth//2, 3*depth//4]
    # This incurs ~10-20% overhead but is necessary for proper gradient synchronization
    var: DDP = (DDP if dist.initialized() else NullDDP)(var_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=True, broadcast_buffers=False)
    
    print(f'[INIT] VAR model = {var_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters())/1e6:.2f}'
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAE', vae_local), ('VAE.enc', vae_local.encoder), ('VAE.dec', vae_local.decoder), ('VAE.quant', vae_local.quantize))]))
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAR', var_wo_ddp),)]) + '\n\n')
    
    # build optimizer
    names, paras, para_groups = filter_params(var_wo_ddp, nowd_keys={
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
        'gate_logit',  # 所有gate_logit（texture + memory）都不做weight decay
    })

    # Memory-specific parameter grouping (if memory enabled)
    if args.mem:
        print(f'[INIT] Memory enabled: refining parameter groups for memory bank...')

        # Build name->parameter mapping for quick lookup
        name_to_param = dict(var_wo_ddp.named_parameters())

        # Identify memory-related parameters
        memory_slot_names = set()
        memory_proj_names = set()
        residual_scale_names = set()

        for name in name_to_param.keys():
            if 'memory_per_scale' in name or 'shared_memory' in name or 'category_memories' in name or 'category_embedding' in name:
                memory_slot_names.add(name)
            elif 'knitting_memory' in name and ('proj' in name or 'out_proj' in name or 'Wk_mem' in name or 'Wv_mem' in name):
                memory_proj_names.add(name)
            elif 'residual_scale' in name:
                residual_scale_names.add(name)

        # Refine existing para_groups: split ND and D groups based on memory params
        final_para_groups = []

        for group in para_groups:
            group_params = group['params']
            wd_sc = group.get('wd_sc', 1.0)
            lr_sc = group.get('lr_sc', 1.0)

            # Split this group into: memory_slots, memory_projs, residual_scales, and regular params
            regular_params = []
            local_memory_slots = []
            local_memory_projs = []
            local_residual_scales = []

            for param in group_params:
                # Find parameter name (linear search, inefficient but simple)
                param_name = None
                for name, p in var_wo_ddp.named_parameters():
                    if p is param:
                        param_name = name
                        break

                if param_name in memory_slot_names:
                    local_memory_slots.append(param)
                elif param_name in memory_proj_names:
                    local_memory_projs.append(param)
                elif param_name in residual_scale_names:
                    local_residual_scales.append(param)
                else:
                    regular_params.append(param)

            # Add refined groups (only if non-empty)
            if len(regular_params) > 0:
                final_para_groups.append({
                    'params': regular_params,
                    'wd_sc': wd_sc,
                    'lr_sc': lr_sc,
                })

            # Memory slots: zero weight decay, inherit lr_sc
            if len(local_memory_slots) > 0:
                final_para_groups.append({
                    'params': local_memory_slots,
                    'wd_sc': 0.0,
                    'lr_sc': lr_sc,
                })

            # Memory projections: 50% weight decay, inherit lr_sc
            if len(local_memory_projs) > 0:
                final_para_groups.append({
                    'params': local_memory_projs,
                    'wd_sc': 0.5,
                    'lr_sc': lr_sc,
                })

            # Residual scales: zero wd, 1.0x lr (need higher lr to grow from 0)
            if len(local_residual_scales) > 0:
                final_para_groups.append({
                    'params': local_residual_scales,
                    'wd_sc': 0.0,
                    'lr_sc': 1.0,
                })

        total_memory_slots = sum(1 for n in memory_slot_names)
        total_memory_projs = sum(1 for n in memory_proj_names)
        total_residual_scales = sum(1 for n in residual_scale_names)

        print(f'[INIT] Memory param refinement: '
              f'slots={total_memory_slots} (wd_sc=0.0), '
              f'projections={total_memory_projs} (wd_sc=0.5), '
              f'residual_scales={total_residual_scales} (lr_sc=1.0, wd_sc=0.0)')
    else:
        # Normal parameter grouping (no memory)
        final_para_groups = para_groups

    # Fine-tuning LR scaling for pretrained/backbone parameters
    final_para_groups, scaled_param_count = _apply_finetune_lr_scale(
        para_groups=final_para_groups,
        pretrained_param_ids=pretrained_loaded_param_ids,
        finetune_lr_scale=args.finetune_lr_scale,
    )
    if scaled_param_count > 0:
        print(
            f'[INIT][Finetune] Applied finetune_lr_scale={args.finetune_lr_scale:g} '
            f'to {scaled_param_count} pretrained parameters.'
        )

    # Build optimizer
    opt_clz = {
        'adam':  partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
    }[args.opt.lower().strip()]

    print(f'[INIT] optim={opt_clz}, base_lr={args.tlr}, base_wd={args.twd}\n')

    var_optim = AmpOptimizer(
        mixed_precision=args.fp16, optimizer=opt_clz(params=final_para_groups, lr=args.tlr, weight_decay=args.twd),
        names=names, paras=paras,
        grad_clip=args.tclip, n_gradient_accumulation=args.ac
    )
    del names, paras, para_groups
    
    # build trainer
    trainer = VARTrainer(
        device=args.device, patch_nums=args.patch_nums, resos=args.resos,
        vae_local=vae_local, var_wo_ddp=var_wo_ddp, var=var,
        var_opt=var_optim, label_smooth=args.ls,
    )
    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True) # don't load vae again
    del vae_local, var_wo_ddp, var, var_optim
    
    if args.local_debug:
        rng = torch.Generator('cpu')
        rng.manual_seed(0)
        B = 4
        inp = torch.rand(B, 3, args.data_load_reso, args.data_load_reso)
        label = torch.ones(B, dtype=torch.long)
        
        me = misc.MetricLogger(delimiter='  ')
        trainer.train_step(
            it=0, g_it=0, stepping=True, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, label_B=label, prog_si=args.pg0, prog_wp_it=20,
        )
        trainer.load_state_dict(trainer.state_dict())
        trainer.train_step(
            it=99, g_it=599, stepping=True, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, label_B=label, prog_si=-1, prog_wp_it=20,
        )
        print({k: meter.global_avg for k, meter in me.meters.items()})
        
        args.dump_log(); tb_lg.flush(); tb_lg.close()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
        exit(0)
    
    dist.barrier()
    return (
        tb_lg, trainer, start_ep, start_it,
        iters_train, ld_train, ld_val
    )


def _get_memory_modules(var_model):
    """获取所有memory模块，避免硬编码属性名散布在各处。"""
    modules = []
    if hasattr(var_model, 'blocks'):
        for block in var_model.blocks:
            if hasattr(block.attn, 'knitting_memory'):
                modules.append(block.attn.knitting_memory)
    return modules


def main_training():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    if args.local_debug:
        torch.autograd.set_detect_anomaly(True)
    
    (
        tb_lg, trainer,
        start_ep, start_it,
        iters_train, ld_train, ld_val
    ) = build_everything(args)
    
    # train
    start_time = time.time()
    best_L_mean, best_L_tail, best_acc_mean, best_acc_tail = 999., 999., -1., -1.
    best_val_loss_mean, best_val_loss_tail, best_val_acc_mean, best_val_acc_tail = 999, 999, -1, -1

    # Create memory scheduler if memory is enabled
    memory_scheduler = None
    if args.mem:
        from utils.memory_scheduler import MemoryTrainingScheduler
        memory_scheduler = MemoryTrainingScheduler(
            total_epochs=args.ep,
            warmup_epochs=args.mem_temp_warmup,
            temp_init=0.5,
            temp_final=0.20,  # 避免过早硬化，保持注意力分布
            div_weight_init=0.0,
            div_weight_final=args.mem_div_weight,
        )
        print(f"\n[Memory Scheduler] Warmup epochs: {args.mem_temp_warmup}")
        print(f"  Temperature: 0.5 -> 0.20")
        print(f"  Diversity weight: 0.0 -> {args.mem_div_weight}\n")

    L_mean, L_tail = -1, -1
    for ep in range(start_ep, args.ep):
        # ========== Memory Scheduler ==========
        if memory_scheduler is not None:
            current_temp = memory_scheduler.get_temperature(ep)
            current_div_weight = memory_scheduler.get_diversity_weight(ep)

            # Set temperature to all memory modules
            var_model = trainer.var.module if hasattr(trainer.var, 'module') else trainer.var
            mem_modules = _get_memory_modules(var_model)
            for mem in mem_modules:
                mem.override_temperature = current_temp

            # Freeze learnable temperature after warmup
            if memory_scheduler.should_freeze_temperature(ep):
                print(f"\n[Epoch {ep}] Freezing learnable temperature...")
                for mem in mem_modules:
                    mem.freeze_learnable_temperature()
                print(f"[Epoch {ep}] Temperature fixed at {current_temp:.4f}\n")

            # Pass diversity weight to trainer
            trainer.current_diversity_weight = current_div_weight

            if ep % 10 == 0:
                print(f"[Epoch {ep}] Memory: temp={current_temp:.4f}, div_weight={current_div_weight:.6f}")
        else:
            trainer.current_diversity_weight = 0.0

        # ========== Seam Loss Warmup ==========
        if args.seam_warmup > 0 and ep < args.seam_warmup:
            trainer.current_seam_weight = args.seam_weight * (ep / args.seam_warmup)
        else:
            trainer.current_seam_weight = args.seam_weight

        # ========== Aux cls + slot sep weights ==========
        trainer.current_aux_cls_weight = args.aux_cls_weight
        trainer.current_slot_sep_weight = args.slot_sep_weight

        if ep % 10 == 0:
            print(f"[Epoch {ep}] Losses: seam_w={trainer.current_seam_weight:.4f}, "
                  f"aux_cls_w={trainer.current_aux_cls_weight:.4f}, "
                  f"slot_sep_w={trainer.current_slot_sep_weight:.6f}")

        if hasattr(ld_train, 'sampler') and hasattr(ld_train.sampler, 'set_epoch'):
            ld_train.sampler.set_epoch(ep)
            if ep < 3:
                # noinspection PyArgumentList
                print(f'[{type(ld_train).__name__}] [ld_train.sampler.set_epoch({ep})]', flush=True, force=True)
        tb_lg.set_step(ep * iters_train)
        
        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep, ep == start_ep, start_it if ep == start_ep else 0, args, tb_lg, ld_train, iters_train, trainer
        )
        
        L_mean, L_tail, acc_mean, acc_tail, grad_norm = stats['Lm'], stats['Lt'], stats['Accm'], stats['Acct'], stats['tnm']
        best_L_mean, best_acc_mean = min(best_L_mean, L_mean), max(best_acc_mean, acc_mean)
        if L_tail != -1: best_L_tail, best_acc_tail = min(best_L_tail, L_tail), max(best_acc_tail, acc_tail)
        args.L_mean, args.L_tail, args.acc_mean, args.acc_tail, args.grad_norm = L_mean, L_tail, acc_mean, acc_tail, grad_norm
        args.cur_ep = f'{ep+1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time
        
        AR_ep_loss = dict(L_mean=L_mean, L_tail=L_tail, acc_mean=acc_mean, acc_tail=acc_tail)
        is_val_and_also_saving = (ep + 1) % 10 == 0 or (ep + 1) == args.ep
        if is_val_and_also_saving:
            val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail, tot, cost = trainer.eval_ep(ld_val)
            best_updated = best_val_loss_tail > val_loss_tail
            best_val_loss_mean, best_val_loss_tail = min(best_val_loss_mean, val_loss_mean), min(best_val_loss_tail, val_loss_tail)
            best_val_acc_mean, best_val_acc_tail = max(best_val_acc_mean, val_acc_mean), max(best_val_acc_tail, val_acc_tail)
            AR_ep_loss.update(vL_mean=val_loss_mean, vL_tail=val_loss_tail, vacc_mean=val_acc_mean, vacc_tail=val_acc_tail)
            args.vL_mean, args.vL_tail, args.vacc_mean, args.vacc_tail = val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail
            print(f' [*] [ep{ep}]  (val {tot})  Lm: {val_loss_mean:.4f}, Lt: {val_loss_tail:.4f}, Acc m&t: {val_acc_mean:.2f} {val_acc_tail:.2f},  Val cost: {cost:.2f}s')
            
            if dist.is_local_master():
                current_epoch = ep + 1
                local_out_ckpt = os.path.join(args.local_out_dir_path, 'ar-ckpt-last.pth')
                local_out_ckpt_best = os.path.join(args.local_out_dir_path, 'ar-ckpt-best.pth')
                snapshot_epochs = {30, 40, 50, 60}
                print(f'[saving ckpt] ...', end='', flush=True)
                torch.save({
                    'epoch':    current_epoch,
                    'iter':     0,
                    'trainer':  trainer.state_dict(),
                    'args':     args.state_dict(),
                }, local_out_ckpt)
                if best_updated:
                    shutil.copy(local_out_ckpt, local_out_ckpt_best)
                if current_epoch in snapshot_epochs:
                    snapshot_ckpt = os.path.join(args.local_out_dir_path, f'ar-ckpt-ep{current_epoch}.pth')
                    shutil.copy(local_out_ckpt, snapshot_ckpt)
                print(f'     [saving ckpt](*) finished!  @ {local_out_ckpt}', flush=True, clean=True)
            dist.barrier()
        
        print(    f'     [ep{ep}]  (training )  Lm: {best_L_mean:.3f} ({L_mean:.3f}), Lt: {best_L_tail:.3f} ({L_tail:.3f}),  Acc m&t: {best_acc_mean:.2f} {best_acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}', flush=True)
        tb_lg.update(head='AR_ep_loss', step=ep+1, **AR_ep_loss)
        tb_lg.update(head='AR_z_burnout', step=ep+1, rest_hours=round(sec / 60 / 60, 2))
        args.dump_log(); tb_lg.flush()
    
    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(f'  [*] [PT finished]  Total cost: {total_time},   Lm: {best_L_mean:.3f} ({L_mean}),   Lt: {best_L_tail:.3f} ({L_tail})')
    print('\n\n')
    
    del stats
    del iters_train, ld_train
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    
    args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    print(f'final args:\n\n{str(args)}')
    args.dump_log(); tb_lg.flush(); tb_lg.close()
    dist.barrier()


def train_one_ep(ep: int, is_first_ep: bool, start_it: int, args: arg_util.Args, tb_lg: misc.TensorboardLogger, ld_or_itrt, iters_train: int, trainer):
    # import heavy packages after Dataloader object creation
    from trainer import VARTrainer
    from utils.lr_control import lr_wd_annealing
    trainer: VARTrainer
    
    step_cnt = 0
    me = misc.MetricLogger(delimiter='  ')
    me.add_meter('tlr', misc.SmoothedValue(window_size=1, fmt='{value:.2g}'))
    me.add_meter('tnm', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]
    header = f'[Ep]: [{ep:4d}/{args.ep}]'
    
    if is_first_ep:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
    g_it, max_it = ep * iters_train, args.ep * iters_train
    
    for it, (inp, label) in me.log_every(start_it, iters_train, ld_or_itrt, 30 if iters_train > 8000 else 5, header):
        g_it = ep * iters_train + it
        if it < start_it: continue
        if is_first_ep and it == start_it: warnings.resetwarnings()
        
        inp = inp.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True)
        
        args.cur_it = f'{it+1}/{iters_train}'
        
        wp_it = args.wp * iters_train
        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(args.sche, trainer.var_opt.optimizer, args.tlr, args.twd, args.twde, g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe)
        args.cur_lr, args.cur_wd = max_tlr, max_twd
        
        if args.pg: # default: args.pg == 0.0, means no progressive training, won't get into this
            if g_it <= wp_it: prog_si = args.pg0
            elif g_it >= max_it*args.pg: prog_si = len(args.patch_nums) - 1
            else:
                delta = len(args.patch_nums) - 1 - args.pg0
                progress = min(max((g_it - wp_it) / (max_it*args.pg - wp_it), 0), 1) # from 0 to 1
                prog_si = args.pg0 + round(progress * delta)    # from args.pg0 to len(args.patch_nums)-1
        else:
            prog_si = -1
        
        stepping = (g_it + 1) % args.ac == 0
        step_cnt += int(stepping)
        
        grad_norm, scale_log2 = trainer.train_step(
            it=it, g_it=g_it, stepping=stepping, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, label_B=label, prog_si=prog_si, prog_wp_it=args.pgwp * iters_train,
        )
        
        me.update(tlr=max_tlr)
        tb_lg.set_step(step=g_it)
        tb_lg.update(head='AR_opt_lr/lr_min', sche_tlr=min_tlr)
        tb_lg.update(head='AR_opt_lr/lr_max', sche_tlr=max_tlr)
        tb_lg.update(head='AR_opt_wd/wd_max', sche_twd=max_twd)
        tb_lg.update(head='AR_opt_wd/wd_min', sche_twd=min_twd)
        tb_lg.update(head='AR_opt_grad/fp16', scale_log2=scale_log2)
        
        if args.tclip > 0:
            tb_lg.update(head='AR_opt_grad/grad', grad_norm=grad_norm)
            tb_lg.update(head='AR_opt_grad/grad', grad_clip=args.tclip)
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)  # +15: other cost


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


if __name__ == '__main__':
    try: main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
