from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from _impl.config import RuntimeConfig

Normalization = Callable[[int], nn.Module]


def get_normalization(config: RuntimeConfig) -> Normalization:
    match config.model.normalization:
        case None | "None" | "NoneNorm":
            return lambda _: nn.Identity()
        case "BatchNorm":
            return nn.BatchNorm2d
        case "InstanceNorm":
            return lambda num_features: nn.InstanceNorm2d(
                num_features,
                affine=False,
                track_running_stats=False,
            )
        case "InstanceNorm++":
            return InstanceNorm2dPlus
        case "VarianceNorm":
            return VarianceNorm2d
        case name:
            raise NotImplementedError(f"normalization does not exist: {name}")


class VarianceNorm2d(nn.Module):
    def __init__(self, num_features: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.empty(num_features))
        nn.init.normal_(self.scale, mean=1.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        variance = x.var(dim=(2, 3), keepdim=True, correction=0)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.scale[None, :, None, None]


class InstanceNorm2dPlus(nn.Module):
    def __init__(self, num_features: int, *, bias: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.instance_norm = nn.InstanceNorm2d(
            num_features,
            affine=False,
            track_running_stats=False,
        )
        self.alpha = nn.Parameter(torch.empty(num_features))
        self.gamma = nn.Parameter(torch.empty(num_features))
        if bias:
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("beta", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.alpha, mean=1.0, std=0.02)
        nn.init.normal_(self.gamma, mean=1.0, std=0.02)
        if self.beta is not None:
            nn.init.zeros_(self.beta)

    def forward(self, x: Tensor) -> Tensor:
        channel_means = x.mean(dim=(2, 3))
        centered_means = channel_means - channel_means.mean(dim=-1, keepdim=True)
        mean_variance = channel_means.var(dim=-1, keepdim=True, correction=0)
        normalized_means = centered_means * torch.rsqrt(mean_variance + self.eps)

        h = self.instance_norm(x)
        h = h + normalized_means[:, :, None, None] * self.alpha[None, :, None, None]
        h = h * self.gamma[None, :, None, None]
        if self.beta is not None:
            h = h + self.beta[None, :, None, None]
        return h
