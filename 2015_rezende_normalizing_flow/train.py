import argparse
import sys
from pathlib import Path

import torch
from _impl import DLGMNF, beta_schedule
from torch import optim
from torchvision import utils

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR.parent))

from dataloader import (
    CHANNELS,
    DATASET,
    IMAGE_SIZE,
    build_cifar10_loader,
    infinite_batches,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DLGM with Normalizing Flow.")
    parser.add_argument("--batch-size", default=100, type=int)
    parser.add_argument("--iterations", default=50000, type=int)
    parser.add_argument("--hidden-dim", default=512, type=int)
    parser.add_argument("--latent-dim", default=40, type=int)
    parser.add_argument("--flow-length", default=12, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--beta-start", default=0.01, type=float)
    parser.add_argument("--beta-warmup-steps", default=10000, type=int)
    parser.add_argument(
        "--log-interval", default=100, type=int, help="steps between loss prints"
    )
    parser.add_argument(
        "--sample-interval",
        default=5000,
        type=int,
        help="steps between generated sample grids",
    )
    return parser.parse_args()


def preprocess(x: torch.Tensor, training: bool) -> torch.Tensor:
    return torch.bernoulli(x) if training else (x > 0.5).float()


@torch.no_grad()
def save_samples(model: DLGMNF, output_dir: Path, step: int) -> None:
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    samples = model.sample(64, device=device).cpu()
    path = sample_dir / f"sample_grid_{step:06d}.png"
    utils.save_image(samples, path, nrow=8)


def train(args: argparse.Namespace) -> None:
    output_dir = PROJECT_DIR / "runs" / DATASET
    output_dir.mkdir(parents=True, exist_ok=True)
    image_size = IMAGE_SIZE
    channels = CHANNELS
    image_shape = (channels, image_size, image_size)
    input_dim = channels * image_size * image_size

    train_loader = infinite_batches(build_cifar10_loader(args.batch_size))

    model = DLGMNF(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        flow_length=args.flow_length,
        image_shape=image_shape,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for step in range(1, args.iterations + 1):
        x, _ = next(train_loader)
        x = x.to(device, non_blocking=True)
        x = preprocess(x, training=True)

        beta = beta_schedule(step, args.beta_start, args.beta_warmup_steps)
        out = model(x, beta=beta)

        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        optimizer.step()

        if step % args.log_interval == 0 or step == args.iterations:
            print(
                f"step={step}/{args.iterations}"
                f"  beta={beta:.4f}"
                f"  loss={out.loss.item():.4f}"
                f"  recon_nll={out.recon_nll.item():.4f}"
                f"  kl={out.kl_like.item():.4f}"
                f"  log_det={out.log_det_sum.mean().item():.4f}"
            )

        if step % args.sample_interval == 0 or step == args.iterations:
            save_samples(model, output_dir, step)
            print(f"Step {step}: saved samples")


def main() -> None:
    args = parse_args()
    print(
        f"Training DLGM K={args.flow_length} on {DATASET}"
        f" ({CHANNELS}x{IMAGE_SIZE}x{IMAGE_SIZE})  device={device}"
    )
    train(args)


if __name__ == "__main__":
    main()
