import argparse
import csv
from pathlib import Path
from typing import Iterator, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CELEBA_ROOT  = Path("/root/autodl-tmp/celeba")
CHURCH_ROOT  = Path("/root/autodl-tmp/church_outdoor")
BEDROOM_ROOT = Path("/root/autodl-tmp/lsun_bedroom")

CELEBA_IMG_DIR  = CELEBA_ROOT
CHURCH_IMG_DIR  = CHURCH_ROOT
BEDROOM_IMG_DIR = BEDROOM_ROOT

NUM_WORKERS = 8
TEST_HOLDOUT = 5000


def infinite_loader(loader: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


class FlatImageDataset(Dataset):
    def __init__(self, img_dir: Path, img_names: List[str], transform=None):
        self.img_dir = img_dir
        self.img_names = img_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.img_dir / self.img_names[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


def _celeba_names(partition: int) -> List[str]:
    csv_path = CELEBA_ROOT / "list_eval_partition.csv"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["image_id"] for r in rows if int(r["partition"]) == partition]


def _flat_names(img_dir: Path, train: bool, holdout: int = TEST_HOLDOUT) -> List[str]:
    names = sorted(p.name for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if train:
        return names[:-holdout]
    return names[-holdout:]


def _train_transform(image_size: int, center_crop: int = None) -> transforms.Compose:
    ops = []
    if center_crop is not None:
        ops.append(transforms.CenterCrop(center_crop))
    ops += [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    return transforms.Compose(ops)


def _test_transform(image_size: int, center_crop: int = None) -> transforms.Compose:
    ops = []
    if center_crop is not None:
        ops.append(transforms.CenterCrop(center_crop))
    ops += [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    return transforms.Compose(ops)


def _make_train_loader(dataset: Dataset, batch_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    return infinite_loader(loader)


def _make_test_loader(dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
    )


def build_celeba_loader(args: argparse.Namespace, image_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    transform = _train_transform(image_size, center_crop=178)
    dataset = FlatImageDataset(CELEBA_IMG_DIR, _celeba_names(0), transform)
    return _make_train_loader(dataset, args.batch_size)


def build_celeba_test_loader(args: argparse.Namespace, image_size: int) -> DataLoader:
    transform = _test_transform(image_size, center_crop=178)
    dataset = FlatImageDataset(CELEBA_IMG_DIR, _celeba_names(2), transform)
    return _make_test_loader(dataset, args.batch_size)


def build_church_loader(args: argparse.Namespace, image_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    transform = _train_transform(image_size)
    dataset = FlatImageDataset(CHURCH_IMG_DIR, _flat_names(CHURCH_IMG_DIR, train=True), transform)
    return _make_train_loader(dataset, args.batch_size)


def build_church_test_loader(args: argparse.Namespace, image_size: int) -> DataLoader:
    transform = _test_transform(image_size)
    dataset = FlatImageDataset(CHURCH_IMG_DIR, _flat_names(CHURCH_IMG_DIR, train=False), transform)
    return _make_test_loader(dataset, args.batch_size)


def build_bedroom_loader(args: argparse.Namespace, image_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    transform = _train_transform(image_size)
    dataset = FlatImageDataset(BEDROOM_IMG_DIR, _flat_names(BEDROOM_IMG_DIR, train=True), transform)
    return _make_train_loader(dataset, args.batch_size)


def build_bedroom_test_loader(args: argparse.Namespace, image_size: int) -> DataLoader:
    transform = _test_transform(image_size)
    dataset = FlatImageDataset(BEDROOM_IMG_DIR, _flat_names(BEDROOM_IMG_DIR, train=False), transform)
    return _make_test_loader(dataset, args.batch_size)


def build_loaders(args: argparse.Namespace, image_size: int):
    if args.dataset == "celeba":
        return build_celeba_loader(args, image_size), build_celeba_test_loader(args, image_size)
    if args.dataset == "church":
        return build_church_loader(args, image_size), build_church_test_loader(args, image_size)
    if args.dataset == "bedroom":
        return build_bedroom_loader(args, image_size), build_bedroom_test_loader(args, image_size)
    raise ValueError(f"unknown dataset: {args.dataset}")
