"""Shared CIFAR-10 input pipeline for every image model in this repository."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

DATA_ROOT = Path(__file__).resolve().parent / "data"
DATASET = "cifar10"
IMAGE_SIZE = 32
CHANNELS = 3


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    image_size: int = IMAGE_SIZE
    in_channels: int = CHANNELS


CIFAR10_SPEC = DatasetSpec()


def _uniform_dequantize(x: Tensor) -> Tensor:
    return (x * 255.0 + torch.rand_like(x)) / 256.0


def build_cifar10_loader(
    batch_size: int,
    *,
    value_range: Literal["zero_one", "minus_one_one"] = "zero_one",
    dequantization: Literal["uniform"] | None = None,
    num_workers: int | None = None,
    drop_last: bool = True,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    steps = [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    if dequantization == "uniform":
        steps.append(transforms.Lambda(_uniform_dequantize))
    elif dequantization is not None:
        raise ValueError(f"unsupported dequantization: {dequantization}")

    if value_range == "minus_one_one":
        steps.append(transforms.Normalize((0.5,) * 3, (0.5,) * 3))
    elif value_range != "zero_one":
        raise ValueError(f"unsupported value range: {value_range}")

    workers = min(8, os.cpu_count() or 1) if num_workers is None else num_workers
    dataset = CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transforms.Compose(steps),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def infinite_batches(loader: DataLoader) -> Iterator[tuple[Tensor, Tensor]]:
    while True:
        yield from loader


def infinite_images(loader: DataLoader) -> Iterator[Tensor]:
    for images, _ in infinite_batches(loader):
        yield images
