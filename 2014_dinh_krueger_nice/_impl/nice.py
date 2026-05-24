import torch
import torch.nn as nn

from .modules import CouplingLayer, LogisticDistribution, ScalingLayer


def _make_mask(data_dim: int, even_pass: bool) -> torch.Tensor:
    mask = torch.zeros(data_dim, dtype=torch.bool)
    mask[::2] = True
    if not even_pass:
        mask = ~mask
    return mask


class NICE(nn.Module):
    def __init__(
        self,
        data_dim: int,
        num_coupling_layers: int = 4,
        hidden_dim: int = 1000,
        num_net_layers: int = 6,
    ):
        super().__init__()
        self.data_dim = data_dim

        self.coupling_layers = nn.ModuleList(
            [
                CouplingLayer(
                    data_dim=data_dim,
                    hidden_dim=hidden_dim,
                    mask=_make_mask(data_dim, even_pass=(i % 2 == 0)),
                    num_layers=num_net_layers,
                )
                for i in range(num_coupling_layers)
            ]
        )
        self.scaling_layer = ScalingLayer(data_dim=data_dim)
        self.prior = LogisticDistribution()

    def forward(self, x: torch.Tensor, invert: bool = False):
        if not invert:
            z, log_det = self._encode(x)
            log_likelihood = self.prior.log_prob(z).sum(dim=1) + log_det
            return z, log_likelihood
        return self._decode(x)

    def sample(self, num_samples: int) -> torch.Tensor:
        dev = next(self.parameters()).device
        z = self.prior.sample((num_samples, self.data_dim), device=dev)
        return self._decode(z)

    def _encode(self, x: torch.Tensor):
        z = x
        logdet = x.new_zeros(1)
        for layer in self.coupling_layers:
            z, logdet = layer(z, logdet)
        z, logdet = self.scaling_layer(z, logdet)
        return z, logdet

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        logdet = z.new_zeros(1)
        x, logdet = self.scaling_layer(x, logdet, invert=True)
        for layer in reversed(self.coupling_layers):
            x, logdet = layer(x, logdet, invert=True)
        return x
