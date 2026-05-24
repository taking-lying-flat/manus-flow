import argparse
from pathlib import Path
from typing import Iterator, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

STL10_ROOT = Path("/root/autodl-tmp/STL-10")
CIFAR10_ROOT = Path("/root/autodl-tmp/cifar-10")
FASHION_ROOT = Path("/root/autodl-tmp/fashion-mnist")

IMAGE_SIZE = {
    "stl-10": 96,
    "cifar-10": 32,
    "fashion-mnist": 28,
}


def infinite_loader(loader: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


class FlatImageDataset(Dataset):
    def __init__(self, paths: List[Path], transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


class GrayImageDataset(Dataset):
    def __init__(self, paths: List[Path], transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.paths[idx]).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


def _list_class_paths(root: Path) -> List[Path]:
    paths: List[Path] = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        paths.extend(sorted(cls_dir.glob("*.png")))
    return paths


def _list_flat_paths(root: Path) -> List[Path]:
    return sorted(root.glob("*.png"))


def _rgb_transform(image_size: int, hflip: bool = True) -> transforms.Compose:
    ops = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ]
    if hflip:
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def _gray_transform(image_size: int = 28) -> transforms.Compose:
    ops = []
    if image_size != 28:
        ops.append(transforms.Resize(image_size))
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def _make_loader(dataset: Dataset, batch_size: int, num_workers: int = 4) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    return infinite_loader(loader)


def build_stl10_loader(args: argparse.Namespace, image_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    transform = _rgb_transform(image_size, hflip=True)
    paths = _list_flat_paths(STL10_ROOT)
    return _make_loader(FlatImageDataset(paths, transform), args.batch_size)


def build_cifar10_loader(args: argparse.Namespace, image_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    transform = _rgb_transform(image_size, hflip=True)
    paths = _list_class_paths(CIFAR10_ROOT)
    return _make_loader(FlatImageDataset(paths, transform), args.batch_size)


def build_fashion_mnist_loader(args: argparse.Namespace, image_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    transform = _gray_transform(image_size)
    paths = _list_class_paths(FASHION_ROOT)
    return _make_loader(GrayImageDataset(paths, transform), args.batch_size)


def build_loader(args: argparse.Namespace, image_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    if args.dataset == "stl-10":
        return build_stl10_loader(args, image_size)
    if args.dataset == "cifar-10":
        return build_cifar10_loader(args, image_size)
    if args.dataset == "fashion-mnist":
        return build_fashion_mnist_loader(args, image_size)
    raise ValueError(f"unknown dataset: {args.dataset}")
