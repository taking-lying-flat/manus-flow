import argparse
import math
from pathlib import Path

import torch
from torch import nn, optim
from torchvision import utils

from dataloader import IMAGE_SIZE, build_loader
from _impl import RealNVP, RealNVPLoss, bits_per_dim, get_param_groups

PROJECT_DIR = Path(__file__).resolve().parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHANNELS = {
    "stl-10": 1,
    "cifar-10": 3,
    "fashion-mnist": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RealNVP.")
    parser.add_argument("--dataset", default="cifar-10", choices=["stl-10", "cifar-10", "fashion-mnist"])
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--iterations", default=100000, type=int)
    parser.add_argument("--num-scales", default=None, type=int)
    parser.add_argument("--num-blocks", default=2, type=int)
    parser.add_argument("--mid-channels", default=64, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--max-grad-norm", default=100.0, type=float)
    parser.add_argument("--weight-decay", default=5e-5, type=float)
    parser.add_argument("--num-samples", default=64, type=int)
    parser.add_argument("--log-interval", default=100, type=int, help="steps between loss prints")
    parser.add_argument("--ckpt-interval", default=5000, type=int, help="steps between checkpoint saves (overwrites)")
    return parser.parse_args()


def default_num_scales(image_size: int) -> int:
    return int(math.log2(image_size)) - 1


def build_model(args: argparse.Namespace, image_size: int, in_channels: int) -> RealNVP:
    num_scales = args.num_scales or default_num_scales(image_size)
    return RealNVP(
        num_scales=num_scales,
        in_channels=in_channels,
        mid_channels=args.mid_channels,
        num_blocks=args.num_blocks,
    )


@torch.no_grad()
def sample(net: nn.Module, batch_size: int, image_size: int, in_channels: int) -> torch.Tensor:
    z = torch.randn((batch_size, in_channels, image_size, image_size), dtype=torch.float32, device=device)
    x, _ = net(z, reverse=True)
    return x


@torch.no_grad()
def save_samples(net: nn.Module, args: argparse.Namespace, image_size: int, in_channels: int, step: int, output_dir: Path) -> None:
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    net.eval()
    images = sample(net, args.num_samples, image_size, in_channels).cpu()
    net.train()

    grid_path = sample_dir / f"sample_grid_{step:06d}.png"
    nrow = int(args.num_samples ** 0.5)
    grid = utils.make_grid(images, nrow=nrow, padding=2, pad_value=1.0)
    utils.save_image(grid, grid_path)


def train(args: argparse.Namespace) -> None:
    output_dir = PROJECT_DIR / f"runs/{args.dataset}"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoint.pt"

    image_size = IMAGE_SIZE[args.dataset]
    in_channels = CHANNELS[args.dataset]
    train_loader = build_loader(args, image_size)

    net_single = build_model(args, image_size, in_channels).to(device)
    net = nn.DataParallel(net_single).to(device) if device.type == "cuda" else net_single

    loss_fn = RealNVPLoss()
    param_groups = get_param_groups(net, args.weight_decay)
    optimizer = optim.Adam(param_groups, lr=args.lr)

    running = {"loss": 0.0, "n": 0}

    for step in range(1, args.iterations + 1):
        x, _ = next(train_loader)
        x = x.to(device)

        z, sldj = net(x, reverse=False)
        loss = loss_fn(z, sldj)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
        optimizer.step()

        n = x.size(0)
        running["loss"] += loss.item() * n
        running["n"] += n

        if step % args.log_interval == 0 or step == args.iterations:
            N = running["n"]
            avg_loss = running["loss"] / N
            bpd = bits_per_dim(x, avg_loss)
            print(
                f"step={step}"
                f"  loss={avg_loss:.2f}"
                f"  bpd={float(bpd):.4f}"
            )
            running = {"loss": 0.0, "n": 0}

        if step % args.ckpt_interval == 0 or step == args.iterations:
            torch.save(net_single.state_dict(), ckpt_path)
            save_samples(net_single, args, image_size, in_channels, step, output_dir)
            print(f"Step {step}: saved checkpoint + samples")


def main() -> None:
    args = parse_args()
    image_size = IMAGE_SIZE[args.dataset]
    in_channels = CHANNELS[args.dataset]
    print(f"Training RealNVP on {args.dataset} ({in_channels}x{image_size}x{image_size})  device={device}")
    train(args)


if __name__ == "__main__":
    main()
