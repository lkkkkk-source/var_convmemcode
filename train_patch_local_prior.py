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
    def __init__(self, real_paths: List[str], fake_paths: List[str], patch_size: int = 64, image_size: int = 256):
        self.samples: List[Tuple[str, int]] = [(p, 1) for p in real_paths] + [(p, 0) for p in fake_paths]
        self.patch_size = patch_size
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

        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        patch = x[:, top:top + self.patch_size, left:left + self.patch_size]
        return patch, torch.tensor(label, dtype=torch.float32)


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

    dataset = PatchBinaryDataset(real_paths, fake_paths, patch_size=args.patch_size, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)

    model = PatchRealismScorer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float('inf')
    best_path = osp.join(args.out_dir, 'patch-local-prior-best.pth')
    last_path = osp.join(args.out_dir, 'patch-local-prior-last.pth')

    for ep in range(args.epochs):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for x, y in loader:
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
        print(f'[LocalPrior][ep {ep+1}/{args.epochs}] loss={avg_loss:.4f}, acc={acc:.4f}')

        torch.save({'epoch': ep + 1, 'model': model.state_dict(), 'args': vars(args)}, last_path)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'epoch': ep + 1, 'model': model.state_dict(), 'args': vars(args)}, best_path)

    print(f'[LocalPrior] best saved to: {best_path}')
    print(f'[LocalPrior] last saved to: {last_path}')


if __name__ == '__main__':
    main()
