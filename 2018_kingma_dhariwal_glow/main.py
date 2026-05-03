import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import utils

from dataloader import (
    build_celeba_loader,
    build_celeba_test_loader,
    build_cifar10_loader,
    build_cifar10_test_loader,
    build_stl10_loader,
    build_stl10_test_loader,
)
from glow import Glow
from util import calc_loss, calc_z_shapes, preprocess_image


PROJECT_DIR = Path(__file__).resolve().parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── args ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Glow.")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "celeba", "stl10"])
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--iterations", default=10000, type=int)
    parser.add_argument("--n-flow", default=32, type=int)
    parser.add_argument("--n-block", default=3, type=int)
    parser.add_argument("--n-bits", default=5, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--temp", default=0.7, type=float)
    parser.add_argument("--n-sample", default=20, type=int)
    parser.add_argument("--affine",  default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--conv-lu", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--warmup", default=10000, type=int,
                        help="Linear LR warmup steps (0 = no warmup)")
    return parser.parse_args()


def image_size_of(args: argparse.Namespace) -> int:
    return 32 if args.dataset == "cifar10" else 64


# ─── model / sampling ────────────────────────────────────────────────────────

def build_model(args: argparse.Namespace) -> Glow:
    return Glow(
        3, args.n_flow, args.n_block,
        affine=args.affine,
        conv_lu=args.conv_lu,
    )


def make_z_samples(args: argparse.Namespace, image_size: int) -> List[torch.Tensor]:
    z_shapes = calc_z_shapes(3, image_size, args.n_block)
    return [torch.randn(args.n_sample, *s, device=device) * args.temp for s in z_shapes]


def save_samples(
    model: Glow, args: argparse.Namespace, step: int, output_dir: Path, image_size: int
) -> Path:
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        images = model.reverse(make_z_samples(args, image_size)).cpu()
    model.train()

    grid_path = sample_dir / f"sample_grid_{step:06d}.jpg"
    utils.save_image(images, grid_path, normalize=True, nrow=10, value_range=(-0.5, 0.5))
    return grid_path


# ─── evaluate ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: Glow, args: argparse.Namespace, test_loader: DataLoader, image_size: int) -> float:
    """Average bits-per-dim over a test DataLoader."""
    model.eval()
    total_bpd, n_batches = 0.0, 0
    for image, _ in test_loader:
        image, n_bins = preprocess_image(image.to(device), args.n_bits)
        log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)
        bpd, _, _ = calc_loss(log_p, logdet.mean(), image_size, n_bins)
        total_bpd += bpd.item()
        n_batches += 1
    model.train()
    return total_bpd / n_batches


# ─── train ───────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    image_size   = image_size_of(args)
    output_dir   = PROJECT_DIR / f"runs/{args.dataset}"
    log_interval = 100
    sample_interval = 1000
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "cifar10":
        loader      = build_cifar10_loader(args)
        test_loader = build_cifar10_test_loader(args)
    elif args.dataset == "celeba":
        loader      = build_celeba_loader(args, image_size)
        test_loader = build_celeba_test_loader(args, image_size)
    else:
        loader      = build_stl10_loader(args, image_size)
        test_loader = build_stl10_test_loader(args, image_size)

    model_single = build_model(args).to(device)
    model        = nn.DataParallel(model_single).to(device)
    optimizer    = optim.Adamax(model.parameters(), lr=args.lr)

    for step in range(1, args.iterations + 1):
        image, _ = next(loader)
        image, n_bins = preprocess_image(image.to(device), args.n_bits)

        if step == 1:
            with torch.no_grad():
                model_single(image + torch.rand_like(image) / n_bins)
            print(f"Initialized ActNorm ({args.dataset}, {image_size}×{image_size}).")
            continue

        log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)
        logdet = logdet.mean()
        loss, log_p_bpd, logdet_bpd = calc_loss(log_p, logdet, image_size, n_bins)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # linear LR warmup (matches OpenAI glow)
        if args.warmup > 0:
            optimizer.param_groups[0]["lr"] = args.lr * min(1.0, step / args.warmup)
        optimizer.step()

        if step % log_interval == 0 or step == args.iterations:
            print(
                f"step={step} loss={loss.item():.5f} "
                f"log_p={log_p_bpd.item():.5f} logdet={logdet_bpd.item():.5f}"
            )

        if step % sample_interval == 0 or step == args.iterations:
            grid_path = save_samples(model_single, args, step, output_dir, image_size)
            print(f"Saved samples to {grid_path}")

    test_bpd = evaluate(model_single, args, test_loader, image_size)
    print(f"Test set bits-per-dim ({args.dataset}): {test_bpd:.4f}")


# ─── entry ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    print(f"Training Glow on {args.dataset}  device={device}.")
    train(args)


if __name__ == "__main__":
    main()
