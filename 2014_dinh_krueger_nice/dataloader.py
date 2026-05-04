import argparse
import csv
import struct
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CELEBA_ROOT = Path("/root/autodl-tmp/celeba")
CIFAR10_ROOT = Path("/root/autodl-tmp/cifar10")
FASHION_ROOT = Path("/root/autodl-tmp/fashion-mnist")

CELEBA_IMG_DIR = CELEBA_ROOT / "img_align_celeba" / "img_align_celeba"
CIFAR10_TRAIN_DIR = CIFAR10_ROOT / "train"
CIFAR10_TEST_DIR = CIFAR10_ROOT / "test"

NUM_WORKERS = 4

IMAGE_SIZE = {
    "celeba": 32,
    "cifar10": 32,
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


class IdxImageDataset(Dataset):
    def __init__(self, images_path: Path, labels_path: Path, transform=None):
        self.images = self._read_images(images_path)
        self.labels = self._read_labels(labels_path)
        self.transform = transform

    @staticmethod
    def _read_images(path: Path) -> np.ndarray:
        with open(path, "rb") as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            assert magic == 2051, f"bad magic {magic} in {path}"
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        return data

    @staticmethod
    def _read_labels(path: Path) -> np.ndarray:
        with open(path, "rb") as f:
            magic, num = struct.unpack(">II", f.read(8))
            assert magic == 2049, f"bad magic {magic} in {path}"
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.fromarray(self.images[idx], mode="L")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


def _celeba_names(partition: int) -> List[str]:
    csv_path = CELEBA_ROOT / "list_eval_partition.csv"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["image_id"] for r in rows if int(r["partition"]) == partition]


def _list_cifar10_paths(root: Path) -> List[Path]:
    paths: List[Path] = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        paths.extend(sorted(cls_dir.glob("*.png")))
    return paths


def _train_transform_rgb(image_size: int, center_crop: Optional[int] = None,
                         hflip: bool = True) -> transforms.Compose:
    ops = []
    if center_crop is not None:
        ops.append(transforms.CenterCrop(center_crop))
    ops += [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ]
    if hflip:
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def _test_transform_rgb(image_size: int, center_crop: Optional[int] = None) -> transforms.Compose:
    ops = []
    if center_crop is not None:
        ops.append(transforms.CenterCrop(center_crop))
    ops += [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    return transforms.Compose(ops)


def _gray_transform(image_size: int = 28) -> transforms.Compose:
    ops = []
    if image_size != 28:
        ops.append(transforms.Resize(image_size))
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def _make_train_loader(dataset: Dataset, batch_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    return infinite_loader(loader)


def _make_test_loader(dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def build_celeba_loader(args: argparse.Namespace, image_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    transform = _train_transform_rgb(image_size, center_crop=178)
    paths = [CELEBA_IMG_DIR / n for n in _celeba_names(0)]
    return _make_train_loader(FlatImageDataset(paths, transform), args.batch_size)


def build_celeba_test_loader(args: argparse.Namespace, image_size: int) -> DataLoader:
    transform = _test_transform_rgb(image_size, center_crop=178)
    paths = [CELEBA_IMG_DIR / n for n in _celeba_names(2)]
    return _make_test_loader(FlatImageDataset(paths, transform), args.batch_size)


def build_cifar10_loader(args: argparse.Namespace, image_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    transform = _train_transform_rgb(image_size, hflip=True)
    paths = _list_cifar10_paths(CIFAR10_TRAIN_DIR)
    return _make_train_loader(FlatImageDataset(paths, transform), args.batch_size)


def build_cifar10_test_loader(args: argparse.Namespace, image_size: int) -> DataLoader:
    transform = _test_transform_rgb(image_size)
    paths = _list_cifar10_paths(CIFAR10_TEST_DIR)
    return _make_test_loader(FlatImageDataset(paths, transform), args.batch_size)


def build_fashion_mnist_loader(args: argparse.Namespace, image_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    transform = _gray_transform(image_size)
    dataset = IdxImageDataset(
        FASHION_ROOT / "train-images-idx3-ubyte",
        FASHION_ROOT / "train-labels-idx1-ubyte",
        transform,
    )
    return _make_train_loader(dataset, args.batch_size)


def build_fashion_mnist_test_loader(args: argparse.Namespace, image_size: int) -> DataLoader:
    transform = _gray_transform(image_size)
    dataset = IdxImageDataset(
        FASHION_ROOT / "t10k-images-idx3-ubyte",
        FASHION_ROOT / "t10k-labels-idx1-ubyte",
        transform,
    )
    return _make_test_loader(dataset, args.batch_size)


def build_loaders(args: argparse.Namespace, image_size: int):
    if args.dataset == "celeba":
        return build_celeba_loader(args, image_size), build_celeba_test_loader(args, image_size)
    if args.dataset == "cifar10":
        return build_cifar10_loader(args, image_size), build_cifar10_test_loader(args, image_size)
    if args.dataset == "fashion-mnist":
        return build_fashion_mnist_loader(args, image_size), build_fashion_mnist_test_loader(args, image_size)
    raise ValueError(f"unknown dataset: {args.dataset}")
