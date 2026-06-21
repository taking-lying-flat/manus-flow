from __future__ import annotations

import torch
from torch import Tensor


def anneal_dsm_score_estimation(
    scorenet: torch.nn.Module,
    samples: Tensor,
    sigmas: Tensor,
    labels: Tensor | None = None,
    anneal_power: float = 2.0,
) -> Tensor:
    if labels is None:
        labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)

    used_sigmas = sigmas[labels].reshape(samples.shape[0], *([1] * len(samples.shape[1:])))
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise
    target = -noise / (used_sigmas ** 2)

    scores = scorenet(perturbed_samples, labels)
    target = target.reshape(target.shape[0], -1)
    scores = scores.reshape(scores.shape[0], -1)

    weights = used_sigmas.reshape(samples.shape[0]) ** anneal_power
    loss = 0.5 * ((scores - target) ** 2).sum(dim=-1) * weights

    return loss.mean(dim=0)
