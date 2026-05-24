import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution


class CouplingLayer(nn.Module):
    def __init__(self, data_dim: int, hidden_dim: int, mask: torch.Tensor, num_layers: int = 4):
        super().__init__()
        self.register_buffer("mask", mask.bool())

        d1 = int(mask.sum().item())
        d2 = data_dim - d1

        layers = [nn.Linear(d1, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(hidden_dim, d2))

        self.m = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, logdet, invert: bool = False):
        x1 = x[:, self.mask]
        x2 = x[:, ~self.mask]

        shift = self.m(x1)
        x2_new = x2 - shift if invert else x2 + shift

        y = x.clone()
        y[:, ~self.mask] = x2_new
        return y, logdet


class ScalingLayer(nn.Module):
    def __init__(self, data_dim: int):
        super().__init__()
        self.log_scale_vector = nn.Parameter(torch.zeros(1, data_dim))

    def forward(self, x: torch.Tensor, logdet, invert: bool = False):
        log_det_j = self.log_scale_vector.sum()
        if invert:
            return torch.exp(-self.log_scale_vector) * x, logdet - log_det_j
        return torch.exp(self.log_scale_vector) * x, logdet + log_det_j


class LogisticDistribution(Distribution):
    def __init__(self):
        super().__init__(validate_args=False)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -(F.softplus(x) + F.softplus(-x))

    def sample(self, shape, device=None):
        if device is None:
            device = torch.device("cpu")
        u = torch.rand(shape, device=device, dtype=torch.float32).clamp_(1e-6, 1.0 - 1e-6)
        return torch.log(u) - torch.log1p(-u)
