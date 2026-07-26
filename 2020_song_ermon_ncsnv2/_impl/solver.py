from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

Sigmas = Tensor | Sequence[float]


def _as_sigma_tensor(sigmas: Sigmas, x: Tensor) -> Tensor:
    return torch.as_tensor(sigmas, device=x.device, dtype=x.dtype)


@torch.inference_mode()
def anneal_Langevin_dynamics(
    x_mod: Tensor,
    scorenet: nn.Module,
    sigmas: Sigmas,
    n_steps_each: int = 200,
    step_lr: float = 0.000008,
    final_only: bool = False,
    denoise: bool = True,
) -> list[Tensor]:
    images: list[Tensor] = []
    sigmas = _as_sigma_tensor(sigmas, x_mod)

    for sigma_idx, sigma in enumerate(sigmas):
        labels = torch.full((x_mod.shape[0],), sigma_idx, device=x_mod.device, dtype=torch.long)
        step_size = step_lr * (sigma / sigmas[-1]).square()
        noise_scale = torch.sqrt(step_size * 2.0)

        for _ in range(n_steps_each):
            grad = scorenet(x_mod, labels)
            x_mod = x_mod + step_size * grad + noise_scale * torch.randn_like(x_mod)

            if not final_only:
                images.append(x_mod.cpu())

    if denoise:
        labels = torch.full(
            (x_mod.shape[0],),
            len(sigmas) - 1,
            device=x_mod.device,
            dtype=torch.long,
        )
        x_mod = x_mod + sigmas[-1].square() * scorenet(x_mod, labels)
        if not final_only:
            images.append(x_mod.cpu())

    return [x_mod.cpu()] if final_only else images


@torch.inference_mode()
def anneal_Langevin_dynamics_inpainting(
    x_mod: Tensor,
    refer_image: Tensor,
    scorenet: nn.Module,
    sigmas: Sigmas,
    image_size: int,
    n_steps_each: int = 100,
    step_lr: float = 0.000008,
) -> list[Tensor]:
    images: list[Tensor] = []
    sigmas = _as_sigma_tensor(sigmas, x_mod)
    channels = x_mod.shape[-3]
    cols = image_size // 2

    refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
    refer_image = refer_image.contiguous().reshape(-1, channels, image_size, image_size)
    x_mod = x_mod.reshape(-1, channels, image_size, image_size)
    half_refer_image = refer_image[..., :cols]

    for sigma_idx, sigma in enumerate(sigmas):
        labels = torch.full((x_mod.shape[0],), sigma_idx, device=x_mod.device, dtype=torch.long)
        step_size = step_lr * (sigma / sigmas[-1]).square()
        noise_scale = torch.sqrt(step_size * 2.0)

        for _ in range(n_steps_each):
            images.append(x_mod.cpu())
            x_mod[:, :, :, :cols] = half_refer_image + sigma * torch.randn_like(half_refer_image)

            grad = scorenet(x_mod, labels)
            x_mod = x_mod + step_size * grad + noise_scale * torch.randn_like(x_mod)

    return images


@torch.inference_mode()
def anneal_Langevin_dynamics_interpolation(
    x_mod: Tensor,
    scorenet: nn.Module,
    sigmas: Sigmas,
    n_interpolations: int,
    n_steps_each: int = 200,
    step_lr: float = 0.000008,
    final_only: bool = False,
) -> list[Tensor]:
    images: list[Tensor] = []
    sigmas = _as_sigma_tensor(sigmas, x_mod)
    n_rows = x_mod.shape[0]

    x_mod = x_mod[:, None, ...].repeat(1, n_interpolations, 1, 1, 1)
    x_mod = x_mod.reshape(-1, *x_mod.shape[2:])

    angles = torch.linspace(0.0, math.pi / 2.0, n_interpolations, device=x_mod.device)
    cos_angles = torch.cos(angles)[None, :, None, None, None]
    sin_angles = torch.sin(angles)[None, :, None, None, None]

    for sigma_idx, sigma in enumerate(sigmas):
        labels = torch.full((x_mod.shape[0],), sigma_idx, device=x_mod.device, dtype=torch.long)
        step_size = step_lr * (sigma / sigmas[-1]).square()
        noise_scale = torch.sqrt(step_size * 2.0)

        for _ in range(n_steps_each):
            grad = scorenet(x_mod, labels)
            noise_p = torch.randn(n_rows, *x_mod.shape[1:], device=x_mod.device, dtype=x_mod.dtype)
            noise_q = torch.randn(n_rows, *x_mod.shape[1:], device=x_mod.device, dtype=x_mod.dtype)
            noise = noise_p[:, None, ...] * cos_angles + noise_q[:, None, ...] * sin_angles
            noise = noise.reshape(-1, *noise.shape[2:])

            x_mod = x_mod + step_size * grad + noise_scale * noise

            if not final_only:
                images.append(x_mod.cpu())

    return [x_mod.cpu()] if final_only else images
