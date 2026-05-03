import argparse
import math
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import utils

import util
from dataloader import build_loaders
from real_nvp import RealNVP, RealNVPLoss

PROJECT_DIR = Path(__file__).resolve().parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RealNVP.")
    parser.add_argument("--dataset", default="celeba", choices=["celeba", "church", "bedroom"])
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--iterations", default=100000, type=int)
    parser.add_argument("--num-scales", default=None, type=int,
                        help="Number of RealNVP scales (default: log2(image_size/4), last block at 4x4xC).")
    parser.add_argument("--num-blocks", default=2, type=int,
                        help="Residual blocks per coupling: 2 for 64x64, 4 for 32x32, 8 for CIFAR-10.")
    parser.add_argument("--mid-channels", default=64, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--max-grad-norm", default=100.0, type=float)
    parser.add_argument("--weight-decay", default=5e-5, type=float)
    parser.add_argument("--num-samples", default=64, type=int)
    return parser.parse_args()


def default_num_scales(image_size: int) -> int:
    return int(math.log2(image_size)) - 1


IMAGE_SIZE = {"celeba": 64, "church": 64, "bedroom": 64}


def build_model(args: argparse.Namespace) -> RealNVP:
    image_size = IMAGE_SIZE[args.dataset]
    num_scales = args.num_scales or default_num_scales(image_size)
    return RealNVP(
        num_scales=num_scales,
        in_channels=3,
        mid_channels=args.mid_channels,
        num_blocks=args.num_blocks,
    )


@torch.no_grad()
def sample(net: nn.Module, batch_size: int, image_size: int) -> torch.Tensor:
    z = torch.randn((batch_size, 3, image_size, image_size), dtype=torch.float32, device=device)
    x, _ = net(z, reverse=True)
    return x


def save_samples(net: nn.Module, args: argparse.Namespace, image_size: int, step: int, output_dir: Path) -> Path:
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    net.eval()
    images = sample(net, args.num_samples, image_size).cpu()
    net.train()

    grid_path = sample_dir / f"sample_grid_{step:06d}.png"
    nrow = int(args.num_samples ** 0.5)
    grid = utils.make_grid(images, nrow=nrow, padding=2, pad_value=255)
    utils.save_image(grid, grid_path)
    return grid_path


@torch.no_grad()
def evaluate(net: nn.Module, loss_fn: nn.Module, test_loader: DataLoader) -> float:
    net.train()
    total_loss, total_n = 0.0, 0
    for x, _ in test_loader:
        x = x.to(device)
        z, sldj = net(x, reverse=False)
        loss = loss_fn(z, sldj)
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
    bpd = util.bits_per_dim(x, total_loss / total_n)
    return float(bpd)


def train(args: argparse.Namespace) -> None:
    output_dir = PROJECT_DIR / f"runs/{args.dataset}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_interval    = 100
    sample_interval = 1000
    eval_interval   = 5000

    image_size = IMAGE_SIZE[args.dataset]
    train_loader, test_loader = build_loaders(args, image_size)

    net_single = build_model(args).to(device)
    net = nn.DataParallel(net_single).to(device) if device.type == "cuda" else net_single

    loss_fn = RealNVPLoss()
    param_groups = util.get_param_groups(net, args.weight_decay)
    optimizer = optim.Adam(param_groups, lr=args.lr)

    dim = 3 * image_size * image_size
    running = {"loss": 0.0, "sldj": 0.0, "n": 0}

    for step in range(1, args.iterations + 1):
        x, _ = next(train_loader)
        x = x.to(device)

        z, sldj = net(x, reverse=False)
        loss = loss_fn(z, sldj)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
        grad_norm = sum(
            p.grad.detach().norm().item() ** 2
            for p in net.parameters() if p.grad is not None
        ) ** 0.5

        optimizer.step()

        n = x.size(0)
        running["loss"] += loss.item() * n
        running["sldj"] += sldj.mean().item() * n
        running["n"]    += n

        if step % log_interval == 0 or step == args.iterations:
            N = running["n"]
            avg_loss = running["loss"] / N
            avg_sldj = running["sldj"] / N
            bpd      = util.bits_per_dim(x, avg_loss)
            print(
                f"step={step}"
                f"  loss={avg_loss:.2f}"
                f"  bpd={float(bpd):.4f}"
                f"  sldj/dim={avg_sldj / dim:.4f}"
                f"  |g|={grad_norm:.2f}"
            )
            running = {"loss": 0.0, "sldj": 0.0, "n": 0}

        if step % sample_interval == 0 or step == args.iterations:
            grid_path = save_samples(net_single, args, image_size, step, output_dir)
            print(f"Saved samples to {grid_path}")

        if step % eval_interval == 0 or step == args.iterations:
            test_bpd = evaluate(net_single, loss_fn, test_loader)
            print(f"[eval] step={step} test_bpd={test_bpd:.4f}")


def main() -> None:
    args = parse_args()
    image_size = IMAGE_SIZE[args.dataset]
    print(f"Training RealNVP on {args.dataset} ({image_size}x{image_size})  device={device}")
    train(args)


if __name__ == "__main__":
    main()
