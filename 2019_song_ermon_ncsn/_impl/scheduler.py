from __future__ import annotations

import math

import torch
from torch import Tensor


def get_sigmas(
    sigma_begin: float,
    sigma_end: float,
    num_classes: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if sigma_begin <= 0 or sigma_end <= 0:
        raise ValueError("sigma_begin and sigma_end must be positive")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    return torch.exp(
        torch.linspace(
            math.log(sigma_begin),
            math.log(sigma_end),
            steps=num_classes,
            device=device,
            dtype=dtype,
        )
    )
