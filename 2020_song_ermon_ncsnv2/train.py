from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from _impl.config import EXPERIMENT_CONFIG, DataConfig, build_runtime_config
from _impl.ema import EMAHelper
from _impl.loss import anneal_dsm_score_estimation
from _impl.model import NCSNv2
from _impl.scheduler import get_sigmas
from _impl.solver import anneal_Langevin_dynamics
from torch import Tensor, nn
from torchvision import utils as tv_utils

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR.parent))

from dataloader import DATASET, build_cifar10_loader, infinite_images

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainArgs:
    variant: str | None
    ema: bool | None
    batch_size: int | None
    max_steps: int | None
    lr: float


@torch.inference_mode()
def generate_samples(
    *,
    model: nn.Module,
    data: DataConfig,
    device: torch.device,
    sigmas: Tensor,
    out_path: Path,
    num_samples: int = 64,
    n_steps_each: int = 5,
    step_lr: float = 0.00002,
) -> None:
    x_init = torch.rand(
        num_samples,
        data.channels,
        data.image_size,
        data.image_size,
        device=device,
    )

    samples = anneal_Langevin_dynamics(
        x_mod=x_init,
        scorenet=model,
        sigmas=sigmas,
        n_steps_each=n_steps_each,
        step_lr=step_lr,
        final_only=True,
        denoise=True,
    )[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tv_utils.save_image(
        samples.clamp(0.0, 1.0).cpu(),
        out_path,
        nrow=max(1, int(math.sqrt(num_samples))),
    )


def train(args: TrainArgs) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment = EXPERIMENT_CONFIG
    batch_size = args.batch_size or experiment.training.batch_size
    max_steps = args.max_steps or experiment.training.n_iters

    model_config = build_runtime_config(experiment, device=device, variant=args.variant)
    use_ema = model_config.model.ema if args.ema is None else args.ema

    run_dir = PROJECT_DIR / "runs" / DATASET
    run_dir.mkdir(parents=True, exist_ok=True)

    loader = build_cifar10_loader(
        batch_size,
        dequantization="uniform",
        num_workers=experiment.data.num_workers,
    )
    batches = infinite_images(loader)

    sigmas = get_sigmas(model_config)
    model = NCSNv2(model_config).to(device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        model = model.to(memory_format=torch.channels_last)

    ema = EMAHelper(mu=model_config.model.ema_rate) if use_ema else None
    if ema is not None:
        ema.register(model)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=max_steps,
        eta_min=1e-5,
    )

    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)

    LOGGER.info(
        "dataset=%s variant=%s ema=%s samples=%d device=%s",
        DATASET,
        model_config.model.variant,
        use_ema,
        len(loader.dataset),
        device,
    )
    LOGGER.info(
        "lr=%.6g batch_size=%d max_steps=%d amp=%s",
        args.lr,
        batch_size,
        max_steps,
        amp_enabled,
    )

    model.train()
    for step in range(1, max_steps + 1):
        images = next(batches).to(device=device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            loss = anneal_dsm_score_estimation(
                scorenet=model,
                samples=images,
                sigmas=sigmas,
                anneal_power=experiment.training.anneal_power,
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_is_finite = torch.isfinite(grad_norm).item()

        if grad_is_finite:
            scaler.step(optimizer)

        scaler.update()

        if grad_is_finite:
            scheduler.step()

        if ema is not None and grad_is_finite:
            ema.update(model)

        if step == 1 or step % 50 == 0:
            LOGGER.info(
                "step=%d loss=%.6f grad_norm=%.4f lr=%.3e",
                step,
                loss.item(),
                float(grad_norm),
                scheduler.get_last_lr()[0],
            )

        if step % 1000 == 0:
            sample_model = ema.ema_copy(model) if ema is not None else model
            generate_samples(
                model=sample_model,
                data=experiment.data,
                device=device,
                sigmas=sigmas,
                out_path=run_dir / f"samples_step_{step}.png",
                num_samples=16,
                n_steps_each=experiment.training.sample_steps_each,
            )
            del sample_model


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="NCSNv2 training")
    parser.add_argument(
        "--variant",
        choices=("base", "deeper", "deepest"),
        default=None,
        help="model depth; defaults to the selected experiment config",
    )

    parser.add_argument(
        "--ema",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override EMA setting from the selected experiment config",
    )
    parser.add_argument(
        "--batch-size", "--batch_size", dest="batch_size", type=int, default=None
    )
    parser.add_argument(
        "--max-steps", "--max_steps", dest="max_steps", type=int, default=None
    )
    parser.add_argument("--lr", type=float, default=1e-4)

    namespace = parser.parse_args()

    return TrainArgs(
        variant=namespace.variant,
        ema=namespace.ema,
        batch_size=namespace.batch_size,
        max_steps=namespace.max_steps,
        lr=namespace.lr,
    )


if __name__ == "__main__":
    train(parse_args())
