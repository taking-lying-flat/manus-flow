from __future__ import annotations

import torch
from torch import Tensor

from training_utils import geometric_sigmas


def get_sigmas(
    sigma_begin: float,
    sigma_end: float,
    num_classes: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    return geometric_sigmas(
        sigma_begin,
        sigma_end,
        num_classes,
        device=device,
        dtype=dtype,
    )
