import argparse
import os
import os.path as osp
import random
import time
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

from models.patch_realism_scorer import PatchRealismScorer


IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}


def list_images(root: str) -> List[str]:
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if osp.splitext(name)[1].lower() in IMAGE_EXTS:
                paths.append(osp.join(dirpath, name))
    return sorted(paths)


class PatchBinaryDataset(Dataset):
    def __init__(self, real_paths: List[str], fake_paths: List[str], patch_size: int = 64, image_size: int = 256):
        self.samples: List[Tuple[str, int]] = [(p, 1) for p in real_paths] + [(p, 0) for p in fake_paths]
        self.patch_size = patch_size
        self.resize = transforms.Resize((image_size, image_size))
        self.to_tensor = transforms.PILToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        x = self.to_tensor(self.resize(img)).float().div_(255.0)
        _, h, w = x.shape
        if h < self.patch_size or w < self.patch_size:
            x = F.interpolate(x.unsqueeze(0), size=(max(h, self.patch_size), max(w, self.patch_size)), mode='bilinear', align_corners=False).squeeze(0)
            _, h, w = x.shape

        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        patch = x[:, top:top + self.patch_size, left:left + self.patch_size]
        return patch, torch.tensor(label, dtype=torch.float32)


def split_paths(paths: Sequence[str], val_ratio: float, rng: random.Random) -> Tuple[List[str], List[str]]:
    paths = list(paths)
    rng.shuffle(paths)
    if len(paths) < 2 or val_ratio <= 0:
        return paths, []

    val_count = max(1, int(round(len(paths) * val_ratio)))
    val_count = min(val_count, len(paths) - 1)
    train_count = len(paths) - val_count
    return paths[:train_count], paths[train_count:]


class TorchRandomSampler(Sampler[int]):
    def __init__(self, data_source: Dataset, seed: int = 0):
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_source), generator=generator).tolist()
        self.epoch += 1
        return iter(indices)

    def __len__(self) -> int:
        return len(self.data_source)


def run_one_epoch(model, loader, criterion, device, optimizer=None, epoch_idx=None, max_epochs=None, split_name='train', log_interval=0):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total = 0
    correct = 0

    num_batches = len(loader)
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        pred = (torch.sigmoid(logits) > 0.5).float()
        correct += (pred == y).sum().item()

        if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == num_batches):
            elapsed = time.time() - start_time
            avg_loss_so_far = total_loss / max(total, 1)
            acc_so_far = correct / max(total, 1)
            speed = total / max(elapsed, 1e-6)
            prefix = f'[LocalPrior][ep {epoch_idx}/{max_epochs}]' if epoch_idx is not None and max_epochs is not None else '[LocalPrior]'
            print(
                f'{prefix}[{split_name}][batch {batch_idx}/{num_batches}] '
                f'loss={avg_loss_so_far:.4f}, acc={acc_so_far:.4f}, '
                f'samples={total}, speed={speed:.1f} samples/s'
            )

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    elapsed = time.time() - start_time
    return avg_loss, acc, elapsed


def main():
    parser = argparse.ArgumentParser(description='Train learned patch-level local prior')
    parser.add_argument('--real_root', type=str, required=True)
    parser.add_argument('--fake_root', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--min_delta', type=float, default=1e-3)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--resume', type=str, default='')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    real_paths = list_images(args.real_root)
    fake_paths = list_images(args.fake_root)
    if len(real_paths) == 0 or len(fake_paths) == 0:
        raise RuntimeError(f'Empty dataset: real={len(real_paths)}, fake={len(fake_paths)}')

    print(f'[LocalPrior] real patches source: {len(real_paths)} images')
    print(f'[LocalPrior] fake patches source: {len(fake_paths)} images')

    split_rng = random.Random(args.seed)
    train_real, val_real = split_paths(real_paths, args.val_ratio, split_rng)
    train_fake, val_fake = split_paths(fake_paths, args.val_ratio, split_rng)

    train_dataset = PatchBinaryDataset(train_real, train_fake, patch_size=args.patch_size, image_size=args.image_size)
    val_dataset = PatchBinaryDataset(val_real, val_fake, patch_size=args.patch_size, image_size=args.image_size) if val_real and val_fake else None
    train_sampler = TorchRandomSampler(train_dataset, seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)

    print(f'[LocalPrior] train split: real={len(train_real)}, fake={len(train_fake)}')
    print(f'[LocalPrior] val split: real={len(val_real)}, fake={len(val_fake)}')

    model = PatchRealismScorer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float('inf')
    best_epoch = 0
    bad_epochs = 0
    stop_reason = 'max_epochs_reached'
    start_epoch = 0
    best_path = osp.join(args.out_dir, 'patch-local-prior-best.pth')
    last_path = osp.join(args.out_dir, 'patch-local-prior-last.pth')

    if args.resume:
        if not osp.isfile(args.resume):
            raise FileNotFoundError(f'Resume checkpoint not found: {args.resume}')
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt.get('model', ckpt)
        model.load_state_dict(state, strict=True)
        opt_state = ckpt.get('optimizer')
        if opt_state is not None:
            opt.load_state_dict(opt_state)
        best_loss = ckpt.get('best_loss', best_loss)
        best_epoch = ckpt.get('best_epoch', best_epoch)
        bad_epochs = ckpt.get('bad_epochs', bad_epochs)
        start_epoch = int(ckpt.get('epoch', 0))
        print(
            f'[LocalPrior] resumed from: {args.resume} '
            f'(start_epoch={start_epoch}, best_loss={best_loss:.4f}, '
            f'best_epoch={best_epoch}, bad_epochs={bad_epochs})'
        )

    for ep in range(start_epoch, args.max_epochs):
        train_loss, train_acc, train_time = run_one_epoch(
            model, train_loader, criterion, device, optimizer=opt,
            epoch_idx=ep + 1, max_epochs=args.max_epochs, split_name='train', log_interval=args.log_interval
        )
        metric_name = 'train_loss'
        metric_value = train_loss
        val_loss = None
        val_acc = None
        val_time = 0.0

        if val_loader is not None:
            with torch.inference_mode():
                val_loss, val_acc, val_time = run_one_epoch(
                    model, val_loader, criterion, device, optimizer=None,
                    epoch_idx=ep + 1, max_epochs=args.max_epochs, split_name='val', log_interval=args.log_interval
                )
            metric_name = 'val_loss'
            metric_value = val_loss
            print(
                f'[LocalPrior][ep {ep+1}/{args.max_epochs}] '
                f'train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, '
                f'val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, '
                f'train_time={train_time:.1f}s, val_time={val_time:.1f}s'
            )
        else:
            print(
                f'[LocalPrior][ep {ep+1}/{args.max_epochs}] '
                f'train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, train_time={train_time:.1f}s'
            )

        checkpoint = {
            'epoch': ep + 1,
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'args': vars(args),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_loss': best_loss,
            'best_epoch': best_epoch,
            'bad_epochs': bad_epochs,
        }

        improved = metric_value < (best_loss - args.min_delta)
        if improved:
            best_loss = metric_value
            best_epoch = ep + 1
            bad_epochs = 0
            checkpoint['best_loss'] = best_loss
            checkpoint['best_epoch'] = best_epoch
            checkpoint['bad_epochs'] = bad_epochs
            torch.save(checkpoint, best_path)
            print(f'[LocalPrior] new best {metric_name}={best_loss:.4f} at epoch {best_epoch}')
        else:
            bad_epochs += 1
            checkpoint['best_loss'] = best_loss
            checkpoint['best_epoch'] = best_epoch
            checkpoint['bad_epochs'] = bad_epochs
            if val_loader is not None:
                print(f'[LocalPrior] no improvement: best_{metric_name}={best_loss:.4f} at epoch {best_epoch}, patience_left={args.patience - bad_epochs}')
            if val_loader is not None and bad_epochs >= args.patience:
                stop_reason = f'early_stopping(patience={args.patience}, best_epoch={best_epoch})'
                torch.save(checkpoint, last_path)
                print(f'[LocalPrior] Early stopping triggered at epoch {ep+1}; best {metric_name}={best_loss:.4f} at epoch {best_epoch}')
                break

        torch.save(checkpoint, last_path)

    print(f'[LocalPrior] best saved to: {best_path}')
    print(f'[LocalPrior] last saved to: {last_path}')
    print(f'[LocalPrior] stop reason: {stop_reason}')


if __name__ == '__main__':
    main()
