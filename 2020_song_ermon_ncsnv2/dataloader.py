from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from _impl.config import DataConfig

CIFAR10_ROOT = Path("/root/autodl-tmp/cifar10")
CELEBA_ROOT = Path("/root/autodl-tmp/celeba_64")
DATA_ROOTS: dict[tuple[str, str | None], Path] = {
    ("cifar10", None): CIFAR10_ROOT,
    ("celeba", None): CELEBA_ROOT,
}


class FlatImageDataset(Dataset[Tensor]):
    def __init__(self, root: Path, config: DataConfig) -> None:
        self.paths = sorted(
            p for p in root.expanduser().resolve().rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        steps = [transforms.Resize((config.image_size, config.image_size))]
        if config.random_flip:
            steps.append(transforms.RandomHorizontalFlip(p=0.5))
        steps.append(transforms.ToTensor())
        self.transform = transforms.Compose(steps)
        self.config = config

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tensor:
        with Image.open(self.paths[idx]) as img:
            return preprocess_image(self.transform(img.convert("RGB")), self.config)


def dequantize_8bit(x: Tensor) -> Tensor:
    return (x * 255.0 + torch.rand_like(x)) / 256.0


def preprocess_image(x: Tensor, config: DataConfig) -> Tensor:
    if config.uniform_dequantization:
        x = dequantize_8bit(x)

    if config.gaussian_dequantization:
        x = (x + 0.01 * torch.randn_like(x)).clamp(0.0, 1.0)

    return x


def infinite_loader(loader: DataLoader) -> Iterator[Tensor]:
    while True:
        yield from loader


def build_loaders(data_config: DataConfig, batch_size: int) -> DataLoader:
    category = data_config.category.lower() if data_config.category is not None else None
    dataset_key = (data_config.dataset.lower(), category)

    root = DATA_ROOTS[dataset_key]
    workers = data_config.num_workers
    pin = torch.cuda.is_available()

    ds = FlatImageDataset(
        root=root,
        config=data_config,
    )

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=workers, pin_memory=pin,
        persistent_workers=True, prefetch_factor=2,
    )
    return loader
