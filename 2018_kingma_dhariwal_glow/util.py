import math
from typing import List, Tuple

import torch


def preprocess_image(image: torch.Tensor, n_bits: int) -> Tuple[torch.Tensor, float]:
    """Quantize to n_bits and shift to [-0.5, 0.5)."""
    n_bins = 2.0 ** n_bits
    image = image * 255
    if n_bits < 8:
        image = torch.floor(image / 2 ** (8 - n_bits))
    image = image / n_bins - 0.5
    return image, n_bins


def calc_z_shapes(
    n_channel: int,
    input_size: int,
    n_block: int,
) -> List[Tuple[int, int, int]]:
    """Return latent tensor shapes for each Glow block."""
    z_shapes = []
    for _ in range(n_block - 1):
        input_size //= 2
        n_channel *= 2
        z_shapes.append((n_channel, input_size, input_size))
    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))
    return z_shapes


def calc_loss(
    log_p: torch.Tensor,
    logdet: torch.Tensor,
    image_size: int,
    n_bins: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert log-likelihood to bits-per-dim."""
    n_pixel = image_size * image_size * 3
    loss = -math.log(n_bins) * n_pixel + logdet + log_p
    bits_per_dim = -loss / (math.log(2) * n_pixel)
    log_p_bpd = log_p / (math.log(2) * n_pixel)
    logdet_bpd = logdet / (math.log(2) * n_pixel)
    return bits_per_dim.mean(), log_p_bpd.mean(), logdet_bpd.mean()
