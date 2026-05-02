import argparse
import os
import sys
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, utils
from torchvision.transforms import ToTensor

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dlgm import DLGMNF, beta_schedule


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="./assets/dlgm_image")
    parser.add_argument("--updates", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=40)
    parser.add_argument("--flow_length", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta_start", type=float, default=0.01)
    parser.add_argument("--beta_warmup_steps", type=int, default=10_000)
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1000,
        help="Print train stats every this many optimizer steps (batches).",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=10_000,
        help="Run test eval every this many optimizer steps (batches).",
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=10_000,
        help="Save sample grid every this many optimizer steps (batches).",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def build_loaders(args) -> tuple[DataLoader, DataLoader]:
    train_dataset = datasets.MNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_dataset = datasets.MNIST(
        root=args.data_dir,
        train=False,
        download=True,
        transform=ToTensor(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def preprocess_batch(
    images: torch.Tensor,
    device: torch.device,
    training: bool,
) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    return torch.bernoulli(images) if training else (images > 0.5).float()


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def should_trigger(step: int, interval: int) -> bool:
    return interval > 0 and step % interval == 0


@torch.no_grad()
def evaluate(
    model: DLGMNF,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 100,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "recon_nll": 0.0,
        "kl_like": 0.0,
        "log_px": 0.0,
    }
    count = 0

    for batch_idx, (images, _) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x = preprocess_batch(images, device, training=False)
        out = model(x, beta=1.0)
        batch_size = x.shape[0]
        totals["loss"] += out.loss.item() * batch_size
        totals["recon_nll"] += out.recon_nll.item() * batch_size
        totals["kl_like"] += out.kl_like.item() * batch_size
        totals["log_px"] += out.log_px_given_z.mean().item() * batch_size
        count += batch_size

    model.train()
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def save_samples(
    model: DLGMNF,
    output_dir: str,
    flow_length: int,
    device: torch.device,
    step: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    samples = model.sample(64, device=device).cpu()
    filename = "mnist_K{}_step{}.png".format(flow_length, step)
    utils.save_image(samples, os.path.join(output_dir, filename), nrow=8)


def train(args) -> None:
    device = torch.device(args.device)
    image_shape = (1, 28, 28)
    c, h, w = image_shape
    input_dim = c * h * w
    train_loader, test_loader = build_loaders(args)

    model = DLGMNF(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        flow_length=args.flow_length,
        image_shape=image_shape,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    batches = cycle(train_loader)

    model.train()
    for step in range(1, args.updates + 1):
        images, _ = next(batches)
        x = preprocess_batch(images, device, training=True)
        beta = beta_schedule(step, args.beta_start, args.beta_warmup_steps)
        out = model(x, beta=beta)

        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        optimizer.step()

        if should_trigger(step, args.log_interval):
            print(
                "[MNIST K={}] step={}/{} beta={:.4f} loss={:.4f} "
                "recon_nll={:.4f} kl_like={:.4f} log_det={:.4f}".format(
                    args.flow_length,
                    step,
                    args.updates,
                    beta,
                    out.loss.item(),
                    out.recon_nll.item(),
                    out.kl_like.item(),
                    out.log_det_sum.mean().item(),
                )
            )

        if should_trigger(step, args.eval_interval):
            metrics = evaluate(model, test_loader, device)
            print(
                "[MNIST K={}] test step={} loss={:.4f} recon_nll={:.4f} "
                "kl_like={:.4f} log_px={:.4f}".format(
                    args.flow_length,
                    step,
                    metrics["loss"],
                    metrics["recon_nll"],
                    metrics["kl_like"],
                    metrics["log_px"],
                )
            )

        if should_trigger(step, args.sample_interval):
            save_samples(model, args.output_dir, args.flow_length, device, step)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        args.output_dir,
        "mnist_K{}_final.pt".format(args.flow_length),
    )
    torch.save(
        {
            "args": vars(args),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    print("Saved checkpoint to {}".format(checkpoint_path))


if __name__ == "__main__":
    args = parse_args()
    train(args)