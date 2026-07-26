from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR.parent))

import torch
from _impl.cond_refinenet_dilated import CondRefineNetDilated
from _impl.loss import anneal_dsm_score_estimation
from _impl.scheduler import get_sigmas
from _impl.solver import anneal_langevin_dynamics
from torch import Tensor

from dataloader import (
    CIFAR10_SPEC,
    DATASET,
    DatasetSpec,
    build_cifar10_loader,
    infinite_images,
)
from training_utils import save_image_grid

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    ngf: int
    num_classes: int
    sigma_begin: float
    sigma_end: float
    anneal_power: float
    image_size: int
    channels: int


MODEL_CONFIG = ModelConfig(
    ngf=128,
    num_classes=10,
    sigma_begin=1.0,
    sigma_end=0.01,
    anneal_power=2.0,
    image_size=32,
    channels=3,
)


@torch.inference_mode()
def generate_samples(
    model: torch.nn.Module,
    spec: DatasetSpec,
    device: torch.device,
    sigmas: Tensor,
    out_path: Path,
    *,
    num_samples: int = 64,
    n_steps_each: int = 100,
    step_lr: float = 0.00002,
) -> None:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    was_training = model.training
    model.eval()
    x_init = torch.rand(
        num_samples,
        spec.in_channels,
        spec.image_size,
        spec.image_size,
        device=device,
    )
    samples = anneal_langevin_dynamics(
        model,
        x_init,
        sigmas,
        n_steps_each=n_steps_each,
        step_lr=step_lr,
        record_every=None,
        progress=True,
    )
    if not isinstance(samples, Tensor):
        raise TypeError("sampler unexpectedly returned a trajectory")

    save_image_grid(samples, out_path)
    if was_training:
        model.train()


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MODEL_CONFIG
    save_dir = PROJECT_DIR / "runs" / DATASET
    save_dir.mkdir(parents=True, exist_ok=True)

    loader = build_cifar10_loader(
        args.batch_size,
        dequantization="uniform",
    )
    spec = CIFAR10_SPEC
    batches = infinite_images(loader)

    sigmas = get_sigmas(
        config.sigma_begin, config.sigma_end, config.num_classes, device=device
    )
    raw_model = CondRefineNetDilated(
        image_size=config.image_size,
        channels=config.channels,
        ngf=config.ngf,
        num_classes=config.num_classes,
    ).to(device)

    if device.type == "cuda":
        raw_model = raw_model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(
        raw_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=1e-5
    )

    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    step = 0
    model: torch.nn.Module = raw_model
    torch.backends.cudnn.benchmark = True

    LOGGER.info(
        "dataset=%s samples=%d device=%s save_dir=%s",
        DATASET,
        len(loader.dataset),
        device,
        save_dir,
    )
    LOGGER.info(
        "lr=%.6g batch_size=%d max_steps=%d amp=%s",
        args.lr,
        args.batch_size,
        args.max_steps,
        amp_enabled,
    )

    model.train()
    while step < args.max_steps:
        images = next(batches)
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        labels = torch.randint(config.num_classes, (images.shape[0],), device=device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            loss = anneal_dsm_score_estimation(
                model, images, labels, sigmas, config.anneal_power
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss at step {step + 1}: {loss.item()}"
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(grad_norm):
            optimizer.step()
        scaler.update()
        step += 1
        scheduler.step()

        if step == 1 or step % 50 == 0:
            LOGGER.info(
                "step=%d loss=%.6f grad_norm=%.4f lr=%.3e",
                step,
                loss.item(),
                float(grad_norm),
                scheduler.get_last_lr()[0],
            )

        if step % 1000 == 0:
            generate_samples(
                model, spec, device, sigmas, save_dir / f"samples_step_{step}.png"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NCSN training")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
