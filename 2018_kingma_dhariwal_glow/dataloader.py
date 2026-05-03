import argparse
import csv
from pathlib import Path
from typing import Iterator, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

CIFAR10_ROOT = Path("/root/autodl-tmp/cifar10")
CELEBA_ROOT  = Path("/root/autodl-tmp/celeba")
STL10_ROOT   = Path("/root/autodl-tmp/stl10")


# ─── shared ──────────────────────────────────────────────────────────────────

def infinite_loader(loader: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


# ─── CIFAR-10 ─────────────────────────────────────────────────────────────────

def build_cifar10_loader(args: argparse.Namespace) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    train_folder = CIFAR10_ROOT / "train"
    if not train_folder.is_dir():
        raise RuntimeError(f"CIFAR-10 train split not found at {train_folder}")
    dataset = datasets.ImageFolder(str(train_folder), transform=transform)
    if len(dataset.classes) != 10:
        raise RuntimeError(f"Expected 10 classes, found {len(dataset.classes)}: {dataset.classes}")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    return infinite_loader(loader)


def build_cifar10_test_loader(args: argparse.Namespace) -> DataLoader:
    transform = transforms.Compose(
        [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]
    )
    dataset = datasets.ImageFolder(str(CIFAR10_ROOT / "test"), transform=transform)
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=torch.cuda.is_available(),
    )


# ─── CelebA ──────────────────────────────────────────────────────────────────

class FlatImageDataset(Dataset):
    """Images in a single flat directory; no class subdirs."""

    def __init__(self, img_dir: Path, img_names: List[str], transform=None):
        self.img_dir = img_dir
        self.img_names = img_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.img_dir / self.img_names[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


def _celeba_names(partition: int) -> List[str]:
    """Return image filenames for partition: 0=train / 1=val / 2=test."""
    csv_path = CELEBA_ROOT / "list_eval_partition.csv"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["image_id"] for r in rows if int(r["partition"]) == partition]


def build_celeba_loader(args: argparse.Namespace, image_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    # Aligned CelebA is 178×218; center-crop to 178×178 then resize.
    transform = transforms.Compose(
        [
            transforms.CenterCrop(178),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    img_dir = CELEBA_ROOT / "img_align_celeba" / "img_align_celeba"
    dataset = FlatImageDataset(img_dir, _celeba_names(0), transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    return infinite_loader(loader)


def build_celeba_test_loader(args: argparse.Namespace, image_size: int) -> DataLoader:
    transform = transforms.Compose(
        [transforms.CenterCrop(178), transforms.Resize(image_size), transforms.ToTensor()]
    )
    img_dir = CELEBA_ROOT / "img_align_celeba" / "img_align_celeba"
    dataset = FlatImageDataset(img_dir, _celeba_names(2), transform)
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=torch.cuda.is_available(),
    )


# ─── STL-10 ──────────────────────────────────────────────────────────────────

def _stl10_names(subdir: str) -> List[str]:
    return sorted(f"{subdir}/{f.name}" for f in (STL10_ROOT / subdir).iterdir() if f.suffix == ".png")


def build_stl10_loader(args: argparse.Namespace, image_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    # STL-10 is 96×96; use train + unlabeled for generative training (100k images)
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    names = _stl10_names("train_images") + _stl10_names("unlabeled_images")
    dataset = FlatImageDataset(STL10_ROOT, names, transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    return infinite_loader(loader)


def build_stl10_test_loader(args: argparse.Namespace, image_size: int) -> DataLoader:
    transform = transforms.Compose(
        [transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor()]
    )
    dataset = FlatImageDataset(STL10_ROOT, _stl10_names("test_images"), transform)
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=torch.cuda.is_available(),
    )
