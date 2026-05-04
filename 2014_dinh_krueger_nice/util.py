from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn

LOGIT_ALPHA = 0.05


def preprocess(x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    noise = torch.rand_like(x) if training else torch.full_like(x, 0.5)
    x = (x * 255.0 + noise) / 256.0
    x_scaled = LOGIT_ALPHA + (1.0 - 2.0 * LOGIT_ALPHA) * x
    x_logit = torch.log(x_scaled) - torch.log1p(-x_scaled)
    log_det = (
        math.log(1.0 - 2.0 * LOGIT_ALPHA)
        - torch.log(x_scaled)
        - torch.log1p(-x_scaled)
    )
    log_det = log_det.reshape(x.size(0), -1).sum(dim=1)
    return x_logit, log_det


def preprocess_inverse(x_logit: torch.Tensor) -> torch.Tensor:
    x_scaled = torch.sigmoid(x_logit)
    x = (x_scaled - LOGIT_ALPHA) / (1.0 - 2.0 * LOGIT_ALPHA)
    return x.clamp(0.0, 1.0)


def bits_per_dim(x: torch.Tensor, avg_nll: float, num_bits: int = 8) -> float:
    d = x.size(1) if x.dim() == 2 else x[0].numel()
    return float(avg_nll / (math.log(2) * d)) + num_bits


def get_param_groups(model: nn.Module, weight_decay: float) -> Tuple[dict, ...]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() == 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return (
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    )
