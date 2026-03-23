import os.path as osp
import random

import torch
import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


class CyclicShiftTransform:
    """Random cyclic shift augmentation for periodic patterns (e.g., knitting).
    Applies torch.roll with random offsets along H and W dimensions."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_tensor):
        if random.random() < self.p:
            H, W = img_tensor.shape[-2:]
            dy = random.randint(0, H - 1)
            dx = random.randint(0, W - 1)
            img_tensor = torch.roll(img_tensor, shifts=(dy, dx), dims=(-2, -1))
        return img_tensor

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'


def build_dataset(
    data_path: str, final_reso: int,
    hflip=False, mid_reso=1.125,
    cyclic_shift=False,
    vflip=False,
    rand_rot=False,
    color_jitter=0.0,
):
    # build augmentations
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    train_aug, val_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.RandomCrop((final_reso, final_reso)),
        transforms.ToTensor(),
    ], [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ]
    # Cyclic shift for periodic patterns (applied after ToTensor, before normalize)
    if cyclic_shift:
        train_aug.append(CyclicShiftTransform(p=0.5))
    train_aug.append(normalize_01_into_pm1)
    if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())
    if vflip: train_aug.insert(0, transforms.RandomVerticalFlip())
    if rand_rot: train_aug.insert(0, transforms.RandomChoice([
        transforms.Lambda(lambda x: x),                                          # 0°
        transforms.Lambda(lambda x: x.transpose(PImage.Transpose.ROTATE_90)),    # 90°
        transforms.Lambda(lambda x: x.transpose(PImage.Transpose.ROTATE_180)),   # 180°
        transforms.Lambda(lambda x: x.transpose(PImage.Transpose.ROTATE_270)),   # 270°
    ]))
    if color_jitter > 0:
        train_aug.insert(0, transforms.ColorJitter(
            brightness=color_jitter, contrast=color_jitter,
            saturation=color_jitter * 0.5,
        ))
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    
    # build dataset
    train_set = DatasetFolder(root=osp.join(data_path, 'train'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=train_aug)
    val_set = DatasetFolder(root=osp.join(data_path, 'val'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=val_aug)

    # Automatically get num_classes from dataset
    num_classes = len(train_set.classes)
    print(f'[Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}')
    print(f'[Dataset] classes: {train_set.classes[:5]}{"..." if num_classes > 5 else ""}')
    print_aug(train_aug, '[train]')
    print_aug(val_aug, '[val]')

    return num_classes, train_set, val_set


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')
