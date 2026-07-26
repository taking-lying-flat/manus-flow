"""Small utilities shared by the CIFAR-10 training entry points."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
from torch import Tensor
from torchvision.utils import save_image


def setup_logger(
    log_dir: Path | str | None, log_file: str = "train.log", name: str = "train"
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def geometric_sigmas(
    sigma_begin: float,
    sigma_end: float,
    steps: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if sigma_begin <= 0 or sigma_end <= 0:
        raise ValueError("sigma bounds must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    return torch.exp(
        torch.linspace(
            math.log(sigma_begin),
            math.log(sigma_end),
            steps=steps,
            device=device,
            dtype=dtype,
        )
    )


def save_image_grid(images: Tensor, out_path: Path) -> None:
    if images.ndim != 4 or images.shape[0] == 0:
        raise ValueError("images must have shape [batch, channels, height, width]")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(
        images.detach().cpu(),
        out_path,
        nrow=max(1, math.isqrt(images.shape[0])),
    )
