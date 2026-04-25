import logging
import os
from typing import Optional, Sequence, Tuple
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image


def setup_logger(log_dir: Optional[str], log_file: str = "train.log", name: str = "Diffusion"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_scaling_from_batch(batch: torch.Tensor) -> Tuple[float, float]:
    """Return (scale, shift) so that scale * batch + shift is zero-mean / unit-std."""
    scale = 1.0 / torch.sqrt(torch.mean((batch - batch.mean()) ** 2))
    shift = -torch.mean(batch * scale)
    return scale.item(), shift.item()


def save_checkpoint(model, optimizer, history: list, save_path: str, **meta) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            **meta,
        },
        save_path,
    )


@torch.no_grad()
def save_samples(
    diffusion,
    save_dir: str,
    scale: float,
    shift: float,
    n_samples: int = 64,
    filename: str = "samples.png",
) -> str:
    """Sample x_0 ~ p_theta and save them as a square grid (de-normalized to [0, 1])."""
    samples = diffusion.sample(batch_size=n_samples)
    # Inverse of x = scale * raw + shift, so raw = (x - shift) / scale.
    samples = ((samples - shift) / scale).clamp(0.0, 1.0)

    nrow = max(1, int(round(n_samples**0.5)))
    out_path = os.path.join(save_dir, filename)
    save_image(samples.cpu(), out_path, nrow=nrow, padding=2)
    return out_path


def save_loss_curve(
    history: Sequence[float],
    save_dir: str,
    filename: str = "loss.png",
) -> str:
    """Plot per-epoch training loss (bits/pixel relative to N(0, I) baseline)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(history) + 1), history, marker="o", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train cost (bits/pixel vs. N(0, I))")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(save_dir, filename)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
