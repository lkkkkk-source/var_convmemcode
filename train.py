import gc
import os
import shutil
import subprocess
import sys
import time
import warnings
from functools import partial

import torch
from torch.utils.data import DataLoader

import dist
from utils import arg_util, misc
from utils.data import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.misc import auto_resume


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
        # Auxiliary classification head
        aux_cls_tap_layer=args.aux_tap_layer,
    )
    
    vae_ckpt = './model_path/vae_ch160v4096z32.pth'
    if dist.is_local_master():
        if not os.path.exists(vae_ckpt):
            try:
                subprocess.run(
                    ['wget', '-q', f'https://huggingface.co/FoundationVision/var/resolve/main/{vae_ckpt}'],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f'Failed to download VAE checkpoint: {e}') from e
    dist.barrier()
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    
    # ========== Load pretrained VAR weights (e.g., from ImageNet) ==========
    has_auto_resumed = len(auto_resume_info) > 0 and 'success' in auto_resume_info[-1]
    if args.pretrained_ckpt and has_auto_resumed:
        # Skip if auto_resume already loaded a checkpoint (continuing training)
        print(f'[pretrained] Skipping pretrained loading: auto_resume already loaded a checkpoint')
    elif args.pretrained_ckpt:
        print(f'[pretrained] Loading pretrained VAR from {args.pretrained_ckpt} ...')
        pretrained_state = torch.load(args.pretrained_ckpt, map_location='cpu')

        # Handle checkpoint format: could be raw state_dict or wrapped in 'trainer'/'var_wo_ddp'
        if 'trainer' in pretrained_state and 'var_wo_ddp' in pretrained_state['trainer']:
            pt_sd = pretrained_state['trainer']['var_wo_ddp']
        elif 'var_wo_ddp' in pretrained_state:
            pt_sd = pretrained_state['var_wo_ddp']
        elif 'state_dict' in pretrained_state:
            pt_sd = pretrained_state['state_dict']
        else:
            # Assume it's a raw state_dict
            pt_sd = pretrained_state

        model_sd = var_wo_ddp.state_dict()

        # Filter: skip mismatched shapes (e.g., class_emb, aux_cls_head, new modules)
        loaded_keys = []
        skipped_keys = []
        for k, v in pt_sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k] = v
                loaded_keys.append(k)
            else:
                skipped_keys.append(k)

        # Find keys in model but not in pretrained (new modules)
        new_keys = [k for k in model_sd if k not in pt_sd]

        var_wo_ddp.load_state_dict(model_sd, strict=True)
        print(f'[pretrained] Loaded {len(loaded_keys)}/{len(pt_sd)} keys from pretrained checkpoint')
        if skipped_keys:
            print(f'[pretrained] Skipped (shape mismatch): {skipped_keys[:20]}{"..." if len(skipped_keys) > 20 else ""}')
        if new_keys:
            print(f'[pretrained] New (randomly init): {new_keys[:20]}{"..." if len(new_keys) > 20 else ""}')
        del pretrained_state, pt_sd

    # ========== Freeze layers for fine-tuning ==========
    if args.freeze_layers:
        parts = list(map(int, args.freeze_layers.replace('-', '_').split('_')))
        if len(parts) == 2:
            freeze_start, freeze_end = parts[0], parts[1]
        elif len(parts) == 1:
            freeze_start, freeze_end = 0, parts[0]
        else:
            raise ValueError(f'--freeze_layers format: start_end (e.g., 0_7) or single number (e.g., 7)')

        frozen_count = 0
        for i in range(freeze_start, min(freeze_end + 1, len(var_wo_ddp.blocks))):
            for param in var_wo_ddp.blocks[i].parameters():
                param.requires_grad = False
                frozen_count += 1

        # Also freeze shared embeddings if freezing from layer 0
        if freeze_start == 0:
            for param in var_wo_ddp.word_embed.parameters():
                param.requires_grad = False
                frozen_count += 1
            for param in [var_wo_ddp.pos_1LC, var_wo_ddp.pos_start]:
                param.requires_grad = False
                frozen_count += 1
            for param in var_wo_ddp.lvl_embed.parameters():
                param.requires_grad = False
                frozen_count += 1

        trainable = sum(p.numel() for p in var_wo_ddp.parameters() if p.requires_grad)
        total = sum(p.numel() for p in var_wo_ddp.parameters())
        print(f'[freeze] Froze layers {freeze_start}-{freeze_end} ({frozen_count} params)')
        print(f'[freeze] Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.1f}%)')

    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    var_wo_ddp: VAR = args.compile_model(var_wo_ddp, args.tfast)
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
            if 'memory_per_scale' in name or 'shared_memory' in name or 'category_memories' in name or 'category_embedding' in name or 'cat_A' in name or 'cat_B' in name:
                memory_slot_names.add(name)
            elif 'knitting_memory' in name and ('proj' in name or 'out_proj' in name):
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

    # Apply fine-tuning lr scale to backbone parameters
    if args.pretrained_ckpt and args.finetune_lr_scale != 1.0:
        # Identify new module parameters (texture, memory, class_emb, aux_cls)
        new_param_ids = set()
        for name, param in var_wo_ddp.named_parameters():
            if any(kw in name for kw in ('knitting_memory', 'gabor_texture', 'texture_conv',
                                          'class_emb', 'aux_cls_head', 'Wk_tex', 'Wv_tex', 'Wq_tex',
                                          'Wk_mem', 'Wv_mem', 'alpha_mlp', 'gate_logit')):
                new_param_ids.add(id(param))

        for group in final_para_groups:
            backbone_params = []
            new_params = []
            for p in group['params']:
                if id(p) in new_param_ids:
                    new_params.append(p)
                else:
                    backbone_params.append(p)

            if backbone_params and new_params:
                # Split group: backbone gets scaled lr, new modules get full lr
                group['params'] = new_params  # keep original lr for new modules
                final_para_groups.append({
                    'params': backbone_params,
                    'lr_sc': group.get('lr_sc', 1.0) * args.finetune_lr_scale,
                    'wd_sc': group.get('wd_sc', 1.0),
                })
            elif backbone_params:
                # All backbone params
                group['lr_sc'] = group.get('lr_sc', 1.0) * args.finetune_lr_scale

        print(f'[finetune] Backbone lr scale: {args.finetune_lr_scale}x')

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
            temp_final=0.35,  # 小数据先用较高终温，避免 slot 塌缩
            div_weight_init=0.0,
            div_weight_final=args.mem_div_weight,
        )
        print(f"\n[Memory Scheduler] Warmup epochs: {args.mem_temp_warmup}")
        print(f"  Temperature: 0.5 -> 0.35")
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
        is_val_and_also_saving = (ep + 1) % 2 == 0 or (ep + 1) == args.ep
        if is_val_and_also_saving:
            val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail, tot, cost = trainer.eval_ep(ld_val)
            best_updated = best_val_loss_tail > val_loss_tail
            best_val_loss_mean, best_val_loss_tail = min(best_val_loss_mean, val_loss_mean), min(best_val_loss_tail, val_loss_tail)
            best_val_acc_mean, best_val_acc_tail = max(best_val_acc_mean, val_acc_mean), max(best_val_acc_tail, val_acc_tail)
            AR_ep_loss.update(vL_mean=val_loss_mean, vL_tail=val_loss_tail, vacc_mean=val_acc_mean, vacc_tail=val_acc_tail)
            args.vL_mean, args.vL_tail, args.vacc_mean, args.vacc_tail = val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail
            print(f' [*] [ep{ep}]  (val {tot})  Lm: {L_mean:.4f}, Lt: {L_tail:.4f}, Acc m&t: {acc_mean:.2f} {acc_tail:.2f},  Val cost: {cost:.2f}s')
            
            if dist.is_local_master():
                local_out_ckpt = os.path.join(args.local_out_dir_path, 'ar-ckpt-last.pth')
                local_out_ckpt_best = os.path.join(args.local_out_dir_path, 'ar-ckpt-best.pth')
                print(f'[saving ckpt] ...', end='', flush=True)
                torch.save({
                    'epoch':    ep+1,
                    'iter':     0,
                    'trainer':  trainer.state_dict(),
                    'args':     args.state_dict(),
                }, local_out_ckpt)
                if best_updated:
                    shutil.copy(local_out_ckpt, local_out_ckpt_best)
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
