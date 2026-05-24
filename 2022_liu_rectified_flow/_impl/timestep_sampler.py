from abc import ABC, abstractmethod

import torch
from torch import Tensor, pi


class TimestepSampler(ABC):
    @abstractmethod
    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        ...


class UniformSampler(TimestepSampler):
    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.rand(batch_size, device=device)


class LogitNormalSampler(TimestepSampler):
    """Sample t via logit-normal distribution.

    u ~ N(m, s), t = sigmoid(u) = 1 / (1 + exp(-u))

    m < 0 biases toward data (t → 0), m > 0 biases toward noise (t → 1).
    s controls spread — larger s = more concentrated at extremes.
    """

    def __init__(self, m: float = 0.0, s: float = 1.0):
        self.m = m
        self.s = s

    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        u = torch.randn(batch_size, device=device) * self.s + self.m
        return torch.sigmoid(u)


class CosMapSampler(TimestepSampler):
    """Map uniform u → t such that the log-SNR matches the cosine schedule.

    t = 1 - 1 / (tan(pi/2 * u) + 1),  u ~ U(0,1)
    π(t) = 2 / (π - 2πt + 2πt²)
    """

    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        u = torch.rand(batch_size, device=device)
        return 1.0 - 1.0 / (torch.tan(pi / 2 * u) + 1)


class ModeSampler(TimestepSampler):
    """Mode sampling with heavy tails (SD3 Eq. 20).

    f_mode(u; s) = 1 - u - s * (cos²(π/2 * u) - 1 + u)

    For -1 ≤ s ≤ 2/(π-2) ≈ 1.75, f is monotonic decreasing on [0,1].
    s > 0 favors midpoint, s < 0 favors endpoints, s = 0 is uniform.

    Uses bisection to invert f_mode.
    """

    def __init__(self, s: float = 1.0, n_bisect_iters: int = 50):
        max_s = 2.0 / (pi - 2)
        if not (-1.0 <= s <= max_s + 1e-6):
            raise ValueError(
                f"s={s} is outside the monotonic range [-1, {max_s:.3f}]"
            )
        self.s = s
        self.n_bisect_iters = n_bisect_iters

    def _f(self, u: Tensor) -> Tensor:
        s = self.s
        return 1.0 - u - s * (torch.cos(pi / 2 * u) ** 2 - 1 + u)

    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        t_target = torch.rand(batch_size, device=device)
        lo = torch.zeros(batch_size, device=device)
        hi = torch.ones(batch_size, device=device)

        for _ in range(self.n_bisect_iters):
            mid = (lo + hi) / 2
            f_mid = self._f(mid)
            lo = torch.where(f_mid > t_target, mid, lo)
            hi = torch.where(f_mid > t_target, hi, mid)

        return (lo + hi) / 2


def build_timestep_sampler(name: str, **kwargs) -> TimestepSampler:
    if name == "uniform":
        return UniformSampler()
    if name == "logit_normal":
        return LogitNormalSampler(
            m=kwargs.get("m", 0.0),
            s=kwargs.get("s", 1.0),
        )
    if name == "cosmap":
        return CosMapSampler()
    if name == "mode":
        return ModeSampler(
            s=kwargs.get("mode_s", 1.0),
            n_bisect_iters=kwargs.get("n_bisect_iters", 50),
        )
    raise ValueError(f"Unknown timestep sampler: {name}")
