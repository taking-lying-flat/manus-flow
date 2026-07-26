"""Small utilities shared by the CIFAR-10 training entry points."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor
from torchvision.utils import save_image


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
