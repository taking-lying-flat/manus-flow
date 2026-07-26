import argparse
import sys
from pathlib import Path

import torch
import util
from _impl import NICE
from torch import nn, optim
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
    parser = argparse.ArgumentParser(description="Train NICE.")
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--iterations", default=50000, type=int)
    parser.add_argument("--num-coupling-layers", default=4, type=int)
    parser.add_argument("--hidden-dim", default=1000, type=int)
    parser.add_argument("--num-net-layers", default=6, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--max-grad-norm", default=100.0, type=float)
    parser.add_argument("--weight-decay", default=5e-5, type=float)
    parser.add_argument("--num-samples", default=64, type=int)
    parser.add_argument(
        "--log-interval", default=100, type=int, help="steps between loss prints"
    )
    parser.add_argument(
        "--ckpt-interval",
        default=1000,
        type=int,
        help="steps between checkpoint saves (overwrites)",
    )
    return parser.parse_args()


def data_dim() -> int:
    return CHANNELS * IMAGE_SIZE * IMAGE_SIZE


def build_model(args: argparse.Namespace, data_dim: int) -> NICE:
    return NICE(
        data_dim=data_dim,
        num_coupling_layers=args.num_coupling_layers,
        hidden_dim=args.hidden_dim,
        num_net_layers=args.num_net_layers,
    )


def flatten(x: torch.Tensor, channels: int, image_size: int) -> torch.Tensor:
    return x.view(x.size(0), channels * image_size * image_size)


@torch.no_grad()
def save_samples(
    net: nn.Module,
    args: argparse.Namespace,
    channels: int,
    image_size: int,
    step: int,
    output_dir: Path,
) -> None:
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    net.eval()
    raw = net.sample(args.num_samples)
    pixels = util.preprocess_inverse(raw)
    images = pixels.view(args.num_samples, channels, image_size, image_size).cpu()
    net.train()

    grid_path = sample_dir / f"sample_grid_{step:06d}.png"
    nrow = int(args.num_samples**0.5)
    grid = utils.make_grid(images, nrow=nrow, padding=2, pad_value=1.0)
    utils.save_image(grid, grid_path)


def train(args: argparse.Namespace) -> None:
    output_dir = PROJECT_DIR / "runs" / DATASET
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoint.pt"

    image_size = IMAGE_SIZE
    channels = CHANNELS
    input_dim = data_dim()
    train_loader = infinite_batches(build_cifar10_loader(args.batch_size))

    net_single = build_model(args, input_dim).to(device)
    net = (
        nn.DataParallel(net_single).to(device) if device.type == "cuda" else net_single
    )

    param_groups = util.get_param_groups(net, args.weight_decay)
    optimizer = optim.Adam(param_groups, lr=args.lr)

    running = {"nll": 0.0, "n": 0}

    for step in range(1, args.iterations + 1):
        x, _ = next(train_loader)
        x = x.to(device)

        x_proc, log_det_pre = util.preprocess(x, training=True)
        x_in = flatten(x_proc, channels, image_size)

        _, log_likelihood = net(x_in)
        log_likelihood = log_likelihood + log_det_pre
        loss = -torch.mean(log_likelihood)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
        optimizer.step()

        n = x_in.size(0)
        running["nll"] += loss.item() * n
        running["n"] += n

        if step % args.log_interval == 0 or step == args.iterations:
            N = running["n"]
            avg_nll = running["nll"] / N
            bpd = util.bits_per_dim(x_in, avg_nll, num_bits=8)
            print(
                f"step={step}  loss={avg_nll:.4f}  bpd={bpd:.4f}  |g|={grad_norm:.2f}"
            )
            running = {"nll": 0.0, "n": 0}

        if step % args.ckpt_interval == 0 or step == args.iterations:
            torch.save(net_single.state_dict(), ckpt_path)
            save_samples(net_single, args, channels, image_size, step, output_dir)
            print(f"Step {step}: saved checkpoint + samples")


def main() -> None:
    args = parse_args()
    print(f"Training NICE on {DATASET} ({IMAGE_SIZE}x{IMAGE_SIZE})  device={device}")
    train(args)


if __name__ == "__main__":
    main()
