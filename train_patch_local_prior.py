import argparse
import os
import os.path as osp
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
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
    def __init__(
        self,
        real_paths: List[str],
        fake_paths: List[str],
        patch_size: int = 64,
        image_size: int = 256,
        random_crop: bool = True,
    ):
        self.samples: List[Tuple[str, int]] = [(p, 1) for p in real_paths] + [(p, 0) for p in fake_paths]
        self.patch_size = patch_size
        self.random_crop = random_crop
        self.pre = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        x = self.pre(img)
        _, h, w = x.shape
        if h < self.patch_size or w < self.patch_size:
            x = F.interpolate(x.unsqueeze(0), size=(max(h, self.patch_size), max(w, self.patch_size)), mode='bilinear', align_corners=False).squeeze(0)
            _, h, w = x.shape

        if self.random_crop:
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
        else:
            top = (h - self.patch_size) // 2
            left = (w - self.patch_size) // 2
        patch = x[:, top:top + self.patch_size, left:left + self.patch_size]
        return patch, torch.tensor(label, dtype=torch.float32)


def split_train_val(paths: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    paths = list(paths)
    rng = random.Random(seed)
    rng.shuffle(paths)
    if val_ratio <= 0:
        return paths, []
    n_val = int(round(len(paths) * val_ratio))
    n_val = max(1, min(n_val, len(paths) - 1))
    return paths[n_val:], paths[:n_val]


def binary_auc_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    scores = logits.detach().float().cpu()
    labels = targets.detach().float().cpu()
    pos = labels > 0.5
    neg = ~pos
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=scores.dtype)
    pos_rank_sum = ranks[pos].sum().item()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    all_logits = []
    all_targets = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        pred = (torch.sigmoid(logits) > 0.5).float()
        correct += (pred == y).sum().item()
        all_logits.append(logits.detach())
        all_targets.append(y.detach())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    auc = binary_auc_from_logits(torch.cat(all_logits), torch.cat(all_targets)) if all_logits else 0.5
    return avg_loss, acc, auc


def main():
    parser = argparse.ArgumentParser(description='Train learned patch-level local prior')
    parser.add_argument('--real_root', type=str, required=True)
    parser.add_argument('--fake_root', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--best_metric', type=str, default='val_auc', choices=['val_auc', 'val_loss'])
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

    real_train, real_val = split_train_val(real_paths, args.val_ratio, args.seed)
    fake_train, fake_val = split_train_val(fake_paths, args.val_ratio, args.seed + 1)
    print(f'[LocalPrior] train split: real={len(real_train)}, fake={len(fake_train)}')
    print(f'[LocalPrior] val split: real={len(real_val)}, fake={len(fake_val)}')

    train_dataset = PatchBinaryDataset(real_train, fake_train, patch_size=args.patch_size, image_size=args.image_size, random_crop=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)

    val_loader = None
    if len(real_val) > 0 and len(fake_val) > 0:
        val_dataset = PatchBinaryDataset(real_val, fake_val, patch_size=args.patch_size, image_size=args.image_size, random_crop=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)

    model = PatchRealismScorer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_score = -float('inf')
    best_path = osp.join(args.out_dir, 'patch-local-prior-best.pth')
    last_path = osp.join(args.out_dir, 'patch-local-prior-last.pth')

    for ep in range(args.epochs):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            pred = (torch.sigmoid(logits) > 0.5).float()
            correct += (pred == y).sum().item()

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)

        if val_loader is not None:
            val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
            monitor_score = val_auc if args.best_metric == 'val_auc' else -val_loss
            print(
                f'[LocalPrior][ep {ep+1}/{args.epochs}] '
                f'train_loss={avg_loss:.4f}, train_acc={acc:.4f}, '
                f'val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_auc={val_auc:.4f}'
            )
        else:
            val_loss, val_acc, val_auc = None, None, None
            monitor_score = -avg_loss
            print(f'[LocalPrior][ep {ep+1}/{args.epochs}] train_loss={avg_loss:.4f}, train_acc={acc:.4f}')

        ckpt = {
            'epoch': ep + 1,
            'model': model.state_dict(),
            'args': vars(args),
            'train_loss': avg_loss,
            'train_acc': acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': val_auc,
        }
        torch.save(ckpt, last_path)
        if monitor_score > best_score:
            best_score = monitor_score
            torch.save(ckpt, best_path)

    print(f'[LocalPrior] best saved to: {best_path}')
    print(f'[LocalPrior] last saved to: {last_path}')


if __name__ == '__main__':
    main()
