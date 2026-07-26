"""Train the Sohl-Dickstein 2015 diffusion model on local image datasets.

Reference:
    Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli.
    "Deep Unsupervised Learning using Nonequilibrium Thermodynamics", ICML 2015.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torchvision import utils as tv_utils

from _impl import diffusion
from dataloader import build_loaders, infinite_loader

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    trajectory_length: int
    n_temporal_basis: int
    n_t_per_minibatch: int
    n_hidden_conv: int
    n_hidden_dense_lower: int
    n_layers_dense_lower: int
    n_hidden_dense_upper: int
    n_layers_dense_upper: int


DEFAULT_MODEL = ModelConfig(
    trajectory_length=1000,
    n_temporal_basis=10,
    n_t_per_minibatch=4,
    n_hidden_conv=64,
    n_hidden_dense_lower=500,
    n_layers_dense_lower=4,
    n_hidden_dense_upper=20,
    n_layers_dense_upper=2,
)


def compute_normalization(loader: DataLoader, device: torch.device, n_batches: int = 10) -> tuple[float, float]:
    """Estimate (scale, shift) so that scale * x + shift ~ N(0, 1)."""
    total, total_sq, count = 0.0, 0.0, 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        x = batch.float()
        total += x.sum().item()
        total_sq += (x ** 2).sum().item()
        count += x.numel()
    mean = total / count
    std = (total_sq / count - mean ** 2) ** 0.5
    return 1.0 / std, -mean / std


@torch.inference_mode()
def generate_samples(
    model: torch.nn.Module,
    scale: float,
    shift: float,
    out_path: Path,
    *,
    num_samples: int = 64,
) -> None:
    samples = model.sample(batch_size=num_samples)
    # scale * raw + shift = normalized → raw = (normalized - shift) / scale
    samples = ((samples - shift) / scale).clamp(0.0, 1.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nrow = max(1, int(math.sqrt(num_samples)))
    tv_utils.save_image(samples.cpu(), out_path, nrow=nrow)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(__file__).resolve().parent / "output" / args.dataset
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    loader, spec = build_loaders(args.dataset, args.batch_size)
    batches = infinite_loader(loader)

    # ---- normalization ----
    scale, shift = compute_normalization(loader, device)
    uniform_noise = (1.0 / 255.0) / scale

    # ---- model ----
    model = diffusion.DiffusionModel(
        spatial_width=spec.image_size,
        channels=spec.in_channels,
        uniform_noise=uniform_noise,
        **asdict(DEFAULT_MODEL),
    ).to(device)

    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=1e-5)

    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    torch.backends.cudnn.benchmark = True

    step = 0
    LOGGER.info("dataset=%s  samples=%d  device=%s  save_dir=%s", args.dataset, len(loader.dataset), device, save_dir)
    LOGGER.info("lr=%.6g  batch_size=%d  max_steps=%d  amp=%s  scale=%.4f  shift=%.4f",
                args.lr, args.batch_size, args.max_steps, amp_enabled, scale, shift)

    while step < args.max_steps:
        images = next(batches)
        images = images.float().to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        # Normalize to zero-mean unit-variance (paper §3.4).
        images = images * scale + shift

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            loss = model(images)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step + 1}: {loss.item()}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if torch.isfinite(grad_norm):
            scaler.step(optimizer)
            scheduler.step()
        scaler.update()
        step += 1

        if step == 1 or step % 50 == 0:
            LOGGER.info("step=%d  loss=%.6f  grad_norm=%.4f  lr=%.3e",
                         step, loss.item(), float(grad_norm), scheduler.get_last_lr()[0])

        if step % 1000 == 0:
            generate_samples(model, scale, shift, save_dir / f"samples_step_{step}.png")

    generate_samples(model, scale, shift, save_dir / "samples_final.png")
    LOGGER.info("done  final_loss=%.6f  out=%s", loss.item(), save_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diffusion model training (Sohl-Dickstein 2015)")
    parser.add_argument("--dataset", choices=["cifar10", "fashion-mnist"], default="cifar10")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
