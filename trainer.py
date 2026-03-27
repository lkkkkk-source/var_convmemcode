import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dist
from models import VAR, VQVAE, VectorQuantizer2
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger
from utils.memory_entropy_monitor import MemoryEntropyMonitor

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class VARTrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, var_wo_ddp: VAR, var: DDP,
        var_opt: AmpOptimizer, label_smooth: float,
    ):
        super(VARTrainer, self).__init__()
        
        self.var, self.vae_local, self.quantize_local = var, vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer2
        self.var_wo_ddp: VAR = var_wo_ddp  # after torch.compile
        self.var_opt = var_opt
        
        del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)
        
        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]

        # Scale-weighted CE: early (coarse) scales weighted higher for structural correctness
        scale_lambdas = [3.0, 3.0, 2.5, 2.0, 1.8, 1.5, 1.2, 1.0, 1.0, 1.0]
        lw = torch.zeros(1, self.L, device=device)
        cur = 0
        for i, pn in enumerate(patch_nums):
            w = scale_lambdas[i] if i < len(scale_lambdas) else 1.0
            lw[0, cur:cur + pn * pn] = w
            cur += pn * pn
        self.loss_weight = lw / lw.sum()  # normalize so total weight sums to 1
        
        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn*pn

        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True

        # Memory training (set by external scheduler)
        self.current_diversity_weight = 0.0
        self.current_slot_sep_weight = 0.001
        self.current_seam_weight = 0.0  # warmed up externally
        self.current_aux_cls_weight = 0.10

        # Memory entropy monitor (for tracking attention distribution)
        self.entropy_monitor = None
        self._captured_hidden_states = {}  # layer_idx -> tensor, captured via hooks
        self._hooks = []
        self._capture_enabled = False  # Only enable near logging steps to avoid per-step detach overhead
        if hasattr(var_wo_ddp, 'blocks'):
            has_memory = any(hasattr(b.attn, 'knitting_memory') for b in var_wo_ddp.blocks)
            if has_memory:
                self.entropy_monitor = MemoryEntropyMonitor(var_wo_ddp, patch_nums)
                # Register forward hooks to capture hidden states entering memory layers
                for idx, block in enumerate(var_wo_ddp.blocks):
                    if hasattr(block.attn, 'knitting_memory'):
                        hook = block.attn.register_forward_hook(
                            self._make_capture_hook(idx)
                        )
                        self._hooks.append(hook)

    def _make_capture_hook(self, layer_idx):
        """Create a forward hook that captures the input to a SelfAttention layer."""
        def hook_fn(module, input, output):
            if not self._capture_enabled:
                return
            # input[0] is x (the normalized hidden state passed to SelfAttention.forward)
            if isinstance(input, tuple) and len(input) > 0:
                self._captured_hidden_states[layer_idx] = input[0].detach()
        return hook_fn
    
    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        for inp_B3HW, label_B in ld_val:
            B, V = label_B.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            label_B = label_B.to(dist.get_device(), non_blocking=True)
            
            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            
            # NOTE: category_ids=label_B assumes num_classes == num_categories.
            # If your dataset has different class/category counts, add a mapping here.
            logits_BLV, _aux = self.var_wo_ddp(label_B, x_BLCv_wo_first_l, category_ids=label_B)
            L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)) * B
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            acc_tail += (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (100 / self.last_l)
            tot += B
        self.var_wo_ddp.train(training)
        
        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt
    
    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], prog_si: int, prog_wp_it: float,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        # if progressive training
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog: prog_wp = 1    # no prog warmup at first prog stage, as it's already solved in wp
        if prog_si == len(self.patch_nums) - 1: prog_si = -1    # max prog, as if no prog
        
        # forward
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping
        
        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
        
        with self.var_opt.amp_ctx:
            # Enable entropy capture only near tensorboard logging steps (every 500)
            self._capture_enabled = (self.entropy_monitor is not None and
                                     (g_it == 0 or (g_it + 1) % 500 == 0))
            # NOTE: category_ids=label_B assumes num_classes == num_categories.
            # If your dataset has different class/category counts, add a mapping here.
            logits_BLV, aux_cls_loss = self.var(label_B, x_BLCv_wo_first_l, category_ids=label_B)
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            if prog_si >= 0:    # in progressive training
                bg, ed = self.begin_ends[prog_si]
                assert logits_BLV.shape[1] == gt_BL.shape[1] == ed
                lw = self.loss_weight[:, :ed].clone()
                lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
            else:               # not in progressive training
                lw = self.loss_weight
            loss = loss.mul(lw).sum(dim=-1).mean()

            # ========== Collect diversity + slot separation losses from memory modules ==========
            diversity_loss = 0.0
            slot_sep_loss = 0.0
            num_memory_layers = 0

            var_model = self.var.module if hasattr(self.var, 'module') else self.var
            for block in var_model.blocks:
                if hasattr(block.attn, 'knitting_memory'):
                    div_loss = block.attn.knitting_memory.last_diversity_loss
                    sep_loss = block.attn.knitting_memory.last_slot_sep_loss
                    # 确保设备一致（防止forward前初始值在CPU上）
                    if div_loss.device != loss.device:
                        div_loss = div_loss.to(loss.device)
                    if sep_loss.device != loss.device:
                        sep_loss = sep_loss.to(loss.device)
                    diversity_loss = diversity_loss + div_loss
                    slot_sep_loss = slot_sep_loss + sep_loss
                    num_memory_layers += 1

            if num_memory_layers > 0:
                diversity_loss = diversity_loss / num_memory_layers
                slot_sep_loss = slot_sep_loss / num_memory_layers
                loss = loss + self.current_diversity_weight * diversity_loss
                loss = loss + self.current_slot_sep_weight * slot_sep_loss

            # ========== Seam loss ==========
            seam_loss = getattr(var_model, 'last_seam_loss', None)
            if seam_loss is not None and self.current_seam_weight > 0:
                if seam_loss.device != loss.device:
                    seam_loss = seam_loss.to(loss.device)
                loss = loss + self.current_seam_weight * seam_loss

            # ========== Auxiliary classification loss (returned from forward) ==========
            if aux_cls_loss is not None and self.current_aux_cls_weight > 0:
                loss = loss + self.current_aux_cls_weight * aux_cls_loss
        
        # backward
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)
        
        # log
        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            if prog_si >= 0:    # in progressive training
                Ltail = acc_tail = -1
            else:               # not in progressive training
                Ltail = self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)).item()
                acc_tail = (pred_BL[:, -self.last_l:] == gt_BL[:, -self.last_l:]).float().mean().item() * 100
            grad_norm = grad_norm.item() if grad_norm is not None else 0.0
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)
        
        # log to tensorboard
        if g_it == 0 or (g_it + 1) % 500 == 0:
            prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float()
            dist.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100
            if dist.is_master():
                if g_it == 0:
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-10000)
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-1000)
                kw = dict(z_voc_usage=cluster_usage)
                for si, (bg, ed) in enumerate(self.begin_ends):
                    if 0 <= prog_si < si: break
                    pred, tar = logits_BLV.data[:, bg:ed].reshape(-1, V), gt_BL[:, bg:ed].reshape(-1)
                    acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                    ce = self.val_loss(pred, tar).item()
                    kw[f'acc_{self.resos[si]}'] = acc
                    kw[f'L_{self.resos[si]}'] = ce
                tb_lg.update(head='AR_iter_loss', **kw, step=g_it)
                tb_lg.update(head='AR_iter_schedule', prog_a_reso=self.resos[prog_si], prog_si=prog_si, prog_wp=prog_wp, step=g_it)

                # Log memory diversity loss if enabled
                if num_memory_layers > 0:
                    div_val = diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else diversity_loss
                    sep_val = slot_sep_loss.item() if isinstance(slot_sep_loss, torch.Tensor) else slot_sep_loss
                    tb_lg.update(head='Memory/diversity_loss', div_loss=div_val, step=g_it)
                    tb_lg.update(head='Memory/slot_sep_loss', sep_loss=sep_val, step=g_it)
                    tb_lg.update(head='Memory/div_weight', div_weight=self.current_diversity_weight, step=g_it)

                    # Log per-layer memory metrics
                    var_model = self.var.module if hasattr(self.var, 'module') else self.var
                    for block_idx, block in enumerate(var_model.blocks):
                        if hasattr(block.attn, 'knitting_memory'):
                            mem = block.attn.knitting_memory
                            temp_val = mem.get_current_temperature()
                            if isinstance(temp_val, torch.Tensor):
                                temp_val = temp_val.item()
                            tb_lg.update(head=f'Memory/gk_weight_layer{block_idx}',
                                        value=torch.sigmoid(mem.gk_logit).item(), step=g_it)
                            tb_lg.update(head=f'Memory/gv_weight_layer{block_idx}',
                                        value=torch.sigmoid(mem.gv_logit).item(), step=g_it)
                            tb_lg.update(head=f'Memory/temperature_layer{block_idx}',
                                        value=temp_val, step=g_it)
                            if hasattr(mem, '_last_collapse_ratio'):
                                tb_lg.update(head=f'Memory/collapse_ratio_layer{block_idx}',
                                            value=mem._last_collapse_ratio, step=g_it)

                    # Log entropy ratio (attention distribution quality)
                    if self.entropy_monitor is not None and self._captured_hidden_states:
                        with torch.no_grad():
                            for layer_idx in self.entropy_monitor.memory_layers:
                                if layer_idx in self._captured_hidden_states:
                                    x_real = self._captured_hidden_states[layer_idx]
                                    metrics = self.entropy_monitor.compute_entropy_ratio(
                                        x_real, layer_idx
                                    )
                                    if metrics:
                                        tb_lg.update(head=f'Memory/entropy_ratio_layer{layer_idx}',
                                                    value=metrics['entropy_ratio'], step=g_it)
                                        tb_lg.update(head=f'Memory/max_attn_layer{layer_idx}',
                                                    value=metrics['max_attn'], step=g_it)
                            self._captured_hidden_states.clear()

                # Log seam loss
                if seam_loss is not None:
                    seam_val = seam_loss.item() if isinstance(seam_loss, torch.Tensor) else seam_loss
                    tb_lg.update(head='Loss/seam_loss', seam=seam_val, step=g_it)
                    tb_lg.update(head='Loss/seam_weight', seam_w=self.current_seam_weight, step=g_it)

                # Log aux cls loss
                if aux_cls_loss is not None:
                    aux_val = aux_cls_loss.item() if isinstance(aux_cls_loss, torch.Tensor) else aux_cls_loss
                    tb_lg.update(head='Loss/aux_cls_loss', aux_cls=aux_val, step=g_it)
        
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return grad_norm, scale_log2
    
    def get_config(self):
        return {
            'patch_nums':   self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }
    
    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[VARTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)
