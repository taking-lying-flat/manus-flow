from __future__ import annotations

import torch
from torch import Tensor

from _impl.config import RuntimeConfig
from training_utils import geometric_sigmas


def get_sigmas(config: RuntimeConfig) -> Tensor:
    return geometric_sigmas(
        config.model.sigma_begin,
        config.model.sigma_end,
        config.model.num_noise_levels,
        device=config.device,
        dtype=torch.float32,
    )
