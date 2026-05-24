import torch
from torch import nn


class TemperedFreeEnergyBound(nn.Module):
    def __init__(
        self,
        log_density,
        beta_start: float = 0.01,
        beta_warmup_steps: int = 10_000,
    ):
        super().__init__()
        self.log_density = log_density
        self.beta_start = beta_start
        self.beta_warmup_steps = beta_warmup_steps

    def beta(self, step: int, device=None) -> torch.Tensor:
        beta_t = min(1.0, self.beta_start + step / self.beta_warmup_steps)
        return torch.tensor(beta_t, device=device)

    def forward(
        self,
        z0: torch.Tensor,
        zk: torch.Tensor,
        log_det_sum: torch.Tensor,
        base_dist: torch.distributions.Distribution,
        step: int,
    ) -> torch.Tensor:
        beta_t = self.beta(step, device=zk.device)
        log_q0 = base_dist.log_prob(z0).reshape(-1)
        log_p_zk = self.log_density(zk).reshape(-1)
        log_det_sum = log_det_sum.reshape(-1)
        return (log_q0 - beta_t * log_p_zk - log_det_sum).mean()
