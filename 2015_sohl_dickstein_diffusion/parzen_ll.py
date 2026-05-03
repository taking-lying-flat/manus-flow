import argparse
import math
import os

import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

import diffusion as model_module
import utils


DATA_ROOT = "/autodl-tmp/data"
DATASET_SHAPE = {"mnist": (1, 28)}
NUM_SAMPLES = 10_000
NUM_VALID = 1_000
SIGMA_GRID = torch.logspace(-1.0, 0.0, 10).tolist()
EVAL_BATCH = 10
SAMPLE_BATCH = 64


def parzen_log_likelihood_batch(x, mu, sigma):
    """Per-row log p(x) under (1/N) Σ_i N(x; mu_i, σ² I), in nats."""
    D = x.shape[1]
    x_sq = (x * x).sum(-1, keepdim=True)
    mu_sq = (mu * mu).sum(-1)
    sq_dist = x_sq + mu_sq.unsqueeze(0) - 2.0 * x @ mu.t()
    E = torch.logsumexp(-0.5 * sq_dist / (sigma * sigma), dim=1) - math.log(mu.shape[0])
    Z = D * math.log(sigma * math.sqrt(2.0 * math.pi))
    return E - Z


@torch.no_grad()
def parzen_log_likelihood(data, samples, sigma, device):
    out = []
    for i in range(0, data.shape[0], EVAL_BATCH):
        out.append(parzen_log_likelihood_batch(
            data[i:i + EVAL_BATCH].to(device), samples, sigma
        ).cpu())
    return torch.cat(out, dim=0)


def load_split_flat(dataset_name, train, limit=None):
    """Return [N, D] tensor in [0, 1] pixel space."""
    if dataset_name != "mnist":
        raise ValueError(f"Unsupported dataset in parzen_ll.py: {dataset_name}")
    ds = MNIST(DATA_ROOT, train=train, transform=transforms.ToTensor(), download=True)
    n = len(ds) if limit is None else min(limit, len(ds))
    return torch.stack([ds[i][0] for i in range(n)]).flatten(1)


@torch.no_grad()
def generate_samples(diffusion, scale, shift):
    chunks, n = [], 0
    while n < NUM_SAMPLES:
        bs = min(SAMPLE_BATCH, NUM_SAMPLES - n)
        x = diffusion.sample(batch_size=bs)
        chunks.append(((x - shift) / scale).clamp(0.0, 1.0).cpu())
        n += bs
    return torch.cat(chunks, dim=0).flatten(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    dataset_name = ckpt["dataset"]
    scale, shift = float(ckpt["scale"]), float(ckpt["shift"])
    n_colors, spatial_width = DATASET_SHAPE[dataset_name]

    save_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    logger = utils.setup_logger(save_dir, log_file="parzen.log")
    logger.info(f"📦 checkpoint={args.checkpoint} dataset={dataset_name} scale={scale:.4f} shift={shift:.4f}")

    diffusion = model_module.DiffusionModel(spatial_width, n_colors).to(device)
    diffusion.load_state_dict(ckpt["model"])
    diffusion.eval()

    logger.info(f"🎲 Generating {NUM_SAMPLES} samples...")
    samples = generate_samples(diffusion, scale, shift).to(device)

    logger.info("🧪 Cross-validating sigma on validation split...")
    valid = load_split_flat(dataset_name, train=True, limit=NUM_VALID)
    best_sigma, best_ll = None, -float("inf")
    for sigma in SIGMA_GRID:
        ll = parzen_log_likelihood(valid, samples, sigma, device).mean().item()
        logger.info(f"  🔍 sigma={sigma:.4f}  mean_ll={ll:.4f}")
        if ll > best_ll:
            best_sigma, best_ll = sigma, ll
    logger.info(f"🏁 best sigma={best_sigma:.4f}")

    logger.info("🧮 Evaluating test set...")
    test = load_split_flat(dataset_name, train=False)
    ll = parzen_log_likelihood(test, samples, best_sigma, device)
    mean_ll = ll.mean().item()
    se = ll.std().item() / math.sqrt(ll.shape[0])

    logger.info(
        f"✅ test log-likelihood = {mean_ll:.3f} ± {se:.3f} nats  "
        f"({mean_ll / math.log(2):.3f} ± {se / math.log(2):.3f} bits)"
    )
