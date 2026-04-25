"""Training entry point for the Sohl-Dickstein 2015 diffusion model.

Reference:
    Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli.
    "Deep Unsupervised Learning using Nonequilibrium Thermodynamics", 2015.
    https://arxiv.org/abs/1503.03585
"""

import argparse
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST

import diffusion
import utils


DATA_ROOT = "/autodl-tmp/data"
DATASETS = {
    "mnist": {
        "image_shape": (1, 28, 28),
        "batch_size": 256,
    },
    "cifar10": {
        "image_shape": (3, 32, 32),
        "batch_size": 128,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=list(DATASETS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def load_data(dataset_name: str, batch_size: int, num_workers: int) -> DataLoader:
    transform = transforms.ToTensor()
    if dataset_name == "mnist":
        train = MNIST(root=DATA_ROOT, train=True, transform=transform, download=True)
    elif dataset_name == "cifar10":
        train = CIFAR10(root=DATA_ROOT, train=True, transform=transform, download=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DataLoader(
        train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )


def train_epoch(diffusion, loader, optimizer, device, scale, shift) -> float:
    """One epoch of variational lower-bound maximization (paper Eq. K)."""
    diffusion.train()
    total_loss, total_n = 0.0, 0

    for inputs, _ in loader:
        x0 = inputs.float().to(device, non_blocking=True) * scale + shift

        optimizer.zero_grad(set_to_none=True)
        loss = diffusion(x0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += float(loss.item()) * x0.size(0)
        total_n += x0.size(0)

    return total_loss / max(total_n, 1)


def fit(diffusion, loader, optimizer, device, epochs, save_dir, logger,
        scale, shift, dataset):
    history = []
    for epoch in range(epochs):
        avg = train_epoch(diffusion, loader, optimizer, device, scale, shift)
        history.append(avg)
        logger.info(f"📈 epoch={epoch + 1:04d}/{epochs:04d} train_cost={avg:.6f}")

    utils.save_checkpoint(
        diffusion, optimizer, history,
        os.path.join(save_dir, "model.pt"),
        scale=scale, shift=shift, dataset=dataset,
    )
    return history


if __name__ == "__main__":
    args = parse_args()
    cfg = DATASETS[args.dataset]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    save_dir = os.path.join("output", args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    logger = utils.setup_logger(save_dir)

    loader = load_data(args.dataset, cfg["batch_size"], args.num_workers)

    # Per-pixel mean/std normalization, plus quantization-noise floor (paper §3.4).
    first_batch, _ = next(iter(loader))
    scale, shift = utils.get_scaling_from_batch(first_batch.float())
    uniform_noise = (1.0 / 255.0) / scale

    n_colors, spatial_width = cfg["image_shape"][0], cfg["image_shape"][1]
    diffusion = diffusion.DiffusionModel(
        spatial_width=spatial_width,
        n_colors=n_colors,
        uniform_noise=uniform_noise,
    ).to(device)

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr)

    logger.info(
        f"🚀 Start dataset={args.dataset} device={device} epochs={args.epochs} "
        f"bs={cfg['batch_size']} lr={args.lr} n_samples={len(loader.dataset)} out={save_dir}"
    )

    history = fit(diffusion, loader, optimizer, device, args.epochs, save_dir, logger,
                  scale, shift, args.dataset)

    logger.info("📷 Generating samples and visualizations...")
    utils.save_samples(diffusion, save_dir, scale=scale, shift=shift, n_samples=64)
    utils.save_loss_curve(history, save_dir)
    logger.info(f"✅ Done final_train_cost={history[-1]:.6f} out={save_dir}")
