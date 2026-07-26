from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CIFAR10_ROOT = Path("/root/autodl-tmp/cifar10")
FASHION_MNIST_ROOT = Path("/root/autodl-tmp/fashion-mnist")


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    image_size: int
    in_channels: int


class FlatImageDataset(Dataset[Tensor]):
    def __init__(self, root: Path, image_size: int, image_mode: str, random_flip: bool) -> None:
        self.paths = sorted(
            p for p in root.expanduser().resolve().rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        steps = [transforms.Resize((image_size, image_size))]
        if random_flip:
            steps.append(transforms.RandomHorizontalFlip(p=0.5))
        steps.append(transforms.ToTensor())
        self.transform = transforms.Compose(steps)
        self.image_mode = image_mode

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tensor:
        with Image.open(self.paths[idx]) as img:
            return self.transform(img.convert(self.image_mode))


def infinite_loader(loader: DataLoader) -> Iterator[Tensor]:
    while True:
        yield from loader


def build_loaders(dataset: str, batch_size: int) -> tuple[DataLoader, DatasetSpec]:
    dataset = dataset.lower()
    workers = min(8, os.cpu_count() or 1)
    pin = torch.cuda.is_available()

    if dataset == "cifar10":
        spec = DatasetSpec(image_size=32, in_channels=3)
        ds = FlatImageDataset(CIFAR10_ROOT, spec.image_size, "RGB", random_flip=True)
    elif dataset == "fashion-mnist":
        spec = DatasetSpec(image_size=28, in_channels=1)
        ds = FlatImageDataset(FASHION_MNIST_ROOT, spec.image_size, "L", random_flip=False)
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=workers, pin_memory=pin,
        persistent_workers=True, prefetch_factor=2,
    )
    return loader, spec
