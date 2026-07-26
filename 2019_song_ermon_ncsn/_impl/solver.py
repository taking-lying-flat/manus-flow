from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor
from tqdm.auto import tqdm

ScoreNet = Callable[..., Tensor]


def _validate_sampling_args(
    n_steps: int,
    step_lr: float,
    record_every: int | None,
) -> None:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if step_lr <= 0:
        raise ValueError("step_lr must be positive")
    if record_every is not None and record_every <= 0:
        raise ValueError("record_every must be positive or None")


def _snapshot(x: Tensor, clamp: bool) -> Tensor:
    x = x.clamp(0.0, 1.0) if clamp else x
    return x.detach().cpu()


@torch.inference_mode()
def anneal_langevin_dynamics(
    scorenet: ScoreNet,
    x_mod: Tensor,
    sigmas: Tensor,
    n_steps_each: int = 100,
    step_lr: float = 0.00002,
    record_every: int | None = None,
    *,
    clamp_output: bool = True,
    progress: bool = True,
) -> Tensor | list[Tensor]:
    _validate_sampling_args(n_steps_each, step_lr, record_every)
    if sigmas.ndim != 1 or sigmas.numel() == 0:
        raise ValueError("sigmas must be a non-empty 1D tensor")
    if torch.any(sigmas <= 0):
        raise ValueError("all sigmas must be positive")

    history: list[Tensor] = []
    global_step = 0
    sigma_min = sigmas[-1]
    batch_size = x_mod.shape[0]
    device = x_mod.device

    sigma_iter = tqdm(
        enumerate(sigmas),
        total=sigmas.numel(),
        desc="annealed Langevin",
        disable=not progress,
    )

    for class_idx, sigma in sigma_iter:
        labels = torch.full(
            (batch_size,), class_idx, device=device, dtype=torch.long
        )
        step_size = step_lr * (sigma / sigma_min).square()
        noise_scale = (2.0 * step_size) ** 0.5

        for _ in range(n_steps_each):
            x_mod = (
                x_mod
                + step_size * scorenet(x_mod, labels)
                + noise_scale * torch.randn_like(x_mod)
            )
            global_step += 1
            if record_every is not None and global_step % record_every == 0:
                history.append(_snapshot(x_mod, clamp_output))

    if record_every is None:
        return x_mod.clamp(0.0, 1.0) if clamp_output else x_mod

    if global_step % record_every != 0:
        history.append(_snapshot(x_mod, clamp_output))
    return history
