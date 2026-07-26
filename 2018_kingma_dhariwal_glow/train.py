import argparse
import sys
from pathlib import Path

import torch
from _impl import Glow, calc_loss, calc_z_shapes, preprocess_image
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
    parser = argparse.ArgumentParser(description="Train Glow.")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--iterations", default=100000, type=int)
    parser.add_argument("--n-flow", default=32, type=int)
    parser.add_argument("--n-block", default=3, type=int)
    parser.add_argument("--n-bits", default=5, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--temp", default=0.7, type=float)
    parser.add_argument("--n-sample", default=20, type=int)
    parser.add_argument("--affine", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--conv-lu", default=True, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--warmup",
        default=10000,
        type=int,
        help="Linear LR warmup steps (0 = no warmup)",
    )
    parser.add_argument(
        "--log-interval", default=100, type=int, help="steps between loss prints"
    )
    parser.add_argument(
        "--ckpt-interval",
        default=2500,
        type=int,
        help="steps between checkpoint saves (overwrites)",
    )
    return parser.parse_args()


def build_model(args: argparse.Namespace, in_channels: int) -> Glow:
    return Glow(
        in_channels, args.n_flow, args.n_block, affine=args.affine, conv_lu=args.conv_lu
    )


@torch.no_grad()
def make_z_samples(
    args: argparse.Namespace, image_size: int, in_channels: int
) -> list[torch.Tensor]:
    z_shapes = calc_z_shapes(in_channels, image_size, args.n_block)
    return [torch.randn(args.n_sample, *s, device=device) * args.temp for s in z_shapes]


@torch.no_grad()
def save_samples(
    model: Glow,
    args: argparse.Namespace,
    image_size: int,
    in_channels: int,
    step: int,
    output_dir: Path,
) -> None:
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    images = model.reverse(make_z_samples(args, image_size, in_channels)).cpu()
    model.train()

    grid_path = sample_dir / f"sample_grid_{step:06d}.jpg"
    utils.save_image(
        images, grid_path, normalize=True, nrow=10, value_range=(-0.5, 0.5)
    )


def train(args: argparse.Namespace) -> None:
    output_dir = PROJECT_DIR / "runs" / DATASET
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoint.pt"

    image_size = IMAGE_SIZE
    in_channels = CHANNELS
    train_loader = infinite_batches(build_cifar10_loader(args.batch_size))

    model = build_model(args, in_channels).to(device)
    optimizer = optim.Adamax(model.parameters(), lr=args.lr)

    model.train()
    for step in range(1, args.iterations + 1):
        image, _ = next(train_loader)
        image, n_bins = preprocess_image(image.to(device), args.n_bits)

        if step == 1:
            with torch.no_grad():
                model(image + torch.rand_like(image) / n_bins)
            print(f"ActNorm initialized ({DATASET}, {image_size}x{image_size}).")
            continue

        log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)
        logdet = logdet.mean()
        loss, _, _ = calc_loss(log_p, logdet, image_size, n_bins)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.warmup > 0:
            optimizer.param_groups[0]["lr"] = args.lr * min(1.0, step / args.warmup)
        optimizer.step()

        if step % args.log_interval == 0 or step == args.iterations:
            print(f"step={step}  loss={loss.item():.5f}  bpd={loss.item():.5f}")

        if step % args.ckpt_interval == 0 or step == args.iterations:
            torch.save(model.state_dict(), ckpt_path)
            save_samples(model, args, image_size, in_channels, step, output_dir)
            print(f"Step {step}: saved checkpoint + samples")


def main() -> None:
    args = parse_args()
    print(
        f"Training Glow on {DATASET} ({CHANNELS}x{IMAGE_SIZE}x{IMAGE_SIZE})  device={device}"
    )
    train(args)


if __name__ == "__main__":
    main()
