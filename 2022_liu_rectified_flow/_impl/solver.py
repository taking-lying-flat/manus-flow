from typing import Callable, Optional

import torch
from torch import Tensor
from torch import nn
from torchdiffeq import odeint


class ODESolver(nn.Module):
    """Wraps torchdiffeq.odeint for rectified flow sampling."""

    def __init__(self, odeint_kwargs: Optional[dict] = None):
        super().__init__()
        self.odeint_kwargs = odeint_kwargs or dict(
            atol=1e-5, rtol=1e-5, method="midpoint"
        )

    def sample(
        self,
        ode_fn: Callable[[Tensor, Tensor], Tensor],
        x_init: Tensor,
        steps: int,
    ) -> Tensor:
        times = torch.linspace(0.0, 1.0, steps, device=x_init.device)
        trajectory = odeint(ode_fn, x_init, times, **self.odeint_kwargs)
        return trajectory[-1]
