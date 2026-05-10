from functools import partial
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from ._impl.utils import exists


class ImageDataset(Dataset):
    def __init__(
        self,
        folder: str | Path,
        image_size: int,
        exts: List[str] | None = None,
        augment_horizontal_flip: bool = False,
        convert_image_to: Optional[str] = None,
    ):
        super().__init__()
        if exts is None:
            exts = ["jpg", "jpeg", "png", "tiff"]
        if isinstance(folder, str):
            folder = Path(folder)
        assert folder.is_dir()

        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in folder.glob(f"**/*.{ext}")]

        def convert_image_to_fn(img_type, image):
            if image.mode == img_type:
                return image
            return image.convert(img_type)

        maybe_convert_fn = (
            partial(convert_image_to_fn, convert_image_to)
            if exists(convert_image_to)
            else nn.Identity()
        )

        self.transform = T.Compose(
            [
                T.Lambda(maybe_convert_fn),
                T.Resize(image_size),
                T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


def build_loader(
    folder: str,
    image_size: int,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    dataset = ImageDataset(folder=folder, image_size=image_size, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=True,
    )
