from __future__ import annotations

import math

import torch
from torch import Tensor

from _impl.config import RuntimeConfig


def get_sigmas(config: RuntimeConfig) -> Tensor:
    return torch.logspace(
        start=math.log10(float(config.model.sigma_begin)),
        end=math.log10(float(config.model.sigma_end)),
        steps=int(config.model.num_noise_levels),
        device=config.device,
        dtype=torch.float32,
    )
