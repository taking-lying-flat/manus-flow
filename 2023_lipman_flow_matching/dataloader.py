from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

BEDROOM_ROOT = Path("/root/autodl-tmp/lsun_bedroom")
CHURCH_ROOT = Path("/root/autodl-tmp/church_outdoor") / "img_align_celeba" / "img_align_celeba"
STL10_ROOT = Path("/root/autodl-tmp/stl10")

TEST_HOLDOUT = 5000

IMAGE_SIZE = {
    "bedroom": 64,
    "church":  64,
    "stl-10":  96,
}

CHANNELS = {
    "bedroom": 3,
    "church":  3,
    "stl-10":  3,
}


@dataclass
class DatasetSpec:
    image_size: int
    in_channels: int


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


def _list_image_paths(root: Path, recursive: bool = False) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    if recursive:
        return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)
    return sorted(p for p in root.iterdir() if p.suffix.lower() in exts)


def _split_paths(paths: List[Path], train: bool, holdout: int = TEST_HOLDOUT) -> List[Path]:
    return paths[:-holdout] if train else paths[-holdout:]


def _center_crop_min_resize(image_size: int) -> List[transforms.Compose]:
    """CenterCrop the shorter side to make it square, then resize to image_size."""
    return [
        transforms.Lambda(lambda im: transforms.functional.center_crop(im, min(im.size))),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ]


def _train_transform(
    image_size: int,
    center_crop_min: bool = False,
    resize: bool = False,
    hflip: bool = True,
) -> transforms.Compose:
    ops: List = []
    if center_crop_min:
        ops += _center_crop_min_resize(image_size)
    elif resize:
        ops += [transforms.Resize(image_size), transforms.CenterCrop(image_size)]
    if hflip:
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    return transforms.Compose(ops)


def _make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=16,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=True,
    )


def build_bedroom_loader(batch_size: int) -> DataLoader:
    image_size = IMAGE_SIZE["bedroom"]
    paths = _list_image_paths(BEDROOM_ROOT)
    train_tf = _train_transform(image_size, center_crop_min=False, resize=False, hflip=True)
    train = FlatImageDataset(_split_paths(paths, train=True), train_tf)
    return _make_loader(train, batch_size, shuffle=True, drop_last=True)


def build_church_loader(batch_size: int) -> DataLoader:
    image_size = IMAGE_SIZE["church"]
    paths = _list_image_paths(CHURCH_ROOT)
    train_tf = _train_transform(image_size, center_crop_min=True, hflip=True)
    train = FlatImageDataset(_split_paths(paths, train=True), train_tf)
    return _make_loader(train, batch_size, shuffle=True, drop_last=True)


def build_stl10_loader(batch_size: int) -> DataLoader:
    image_size = IMAGE_SIZE["stl-10"]
    train_paths = (
        _list_image_paths(STL10_ROOT / "train_images")
        + _list_image_paths(STL10_ROOT / "unlabeled_images")
    )
    train_tf = _train_transform(image_size, center_crop_min=False, resize=False, hflip=True)
    train = FlatImageDataset(train_paths, train_tf)
    return _make_loader(train, batch_size, shuffle=True, drop_last=True)


def build_loaders(dataset: str, batch_size: int) -> Tuple[DataLoader, DatasetSpec]:
    if dataset == "bedroom":
        train = build_bedroom_loader(batch_size)
    elif dataset == "church":
        train = build_church_loader(batch_size)
    elif dataset == "stl-10":
        train = build_stl10_loader(batch_size)
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    spec = DatasetSpec(image_size=IMAGE_SIZE[dataset], in_channels=CHANNELS[dataset])
    return train, spec


def get_dataloader(
    dataset: str, batch_size: int, data_root: Optional[str] = None
) -> Tuple[DataLoader, DatasetSpec]:
    return build_loaders(dataset, batch_size)
