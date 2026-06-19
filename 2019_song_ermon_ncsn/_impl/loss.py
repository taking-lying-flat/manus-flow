from __future__ import annotations

from collections.abc import Callable

import torch
import torch.autograd as autograd
from torch import Tensor


def _per_sample_sum(x: Tensor) -> Tensor:
    return x.flatten(1).sum(dim=1)


def _per_sample_squared_norm(x: Tensor) -> Tensor:
    x = x.flatten(1)
    return (x * x).sum(dim=1)


def _validate_sigma(sigma: float) -> None:
    if sigma <= 0:
        raise ValueError("sigma must be positive")


def dsm(
    energy_net: Callable[[Tensor], Tensor],
    samples: Tensor,
    sigma: float = 1.0,
) -> Tensor:
    _validate_sigma(sigma)
    x = samples.detach().requires_grad_(True)
    noise = torch.randn_like(x)
    perturbed = x + sigma * noise
    logp = -energy_net(perturbed)
    scaled_score = sigma**2 * autograd.grad(
        logp.sum(),
        perturbed,
        create_graph=True,
    )[0]
    return 0.5 * _per_sample_squared_norm(scaled_score + sigma * noise).mean()


def dsm_score_estimation(
    scorenet: Callable[[Tensor], Tensor],
    samples: Tensor,
    sigma: float = 0.01,
) -> Tensor:
    _validate_sigma(sigma)
    noise = torch.randn_like(samples)
    perturbed = samples + sigma * noise
    target = -noise / sigma
    residual = (scorenet(perturbed) - target).float()
    return 0.5 * _per_sample_squared_norm(residual).mean()


def anneal_dsm_score_estimation(
    scorenet: Callable[[Tensor, Tensor], Tensor],
    samples: Tensor,
    labels: Tensor,
    sigmas: Tensor,
    anneal_power: float = 2.0,
) -> Tensor:
    if samples.ndim < 2:
        raise ValueError("samples must include batch and feature dimensions")
    if labels.ndim != 1 or labels.shape[0] != samples.shape[0]:
        raise ValueError("labels must have shape [batch_size]")
    if sigmas.ndim != 1 or sigmas.numel() == 0:
        raise ValueError("sigmas must be a non-empty 1D tensor")
    if torch.any(sigmas <= 0):
        raise ValueError("all sigmas must be positive")

    labels = labels.long()
    used = sigmas[labels].reshape(samples.shape[0], *([1] * (samples.ndim - 1)))
    noise = torch.randn_like(samples)
    perturbed = samples + noise * used
    target = -noise / used

    residual = (scorenet(perturbed, labels) - target).float()
    per_sample_loss = 0.5 * _per_sample_squared_norm(residual)
    weights = used.reshape(samples.shape[0]).float().pow(anneal_power)
    return (per_sample_loss * weights).mean()


