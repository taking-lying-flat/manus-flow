import pickle
import sys
from pathlib import Path

import torch
from bernoulli_vae import VAE
from torch import optim
from torchvision.utils import save_image

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR.parent))

from dataloader import CIFAR10_SPEC, DATASET, build_cifar10_loader
from training_utils import setup_logger

IMAGE_SHAPE = (
    CIFAR10_SPEC.in_channels,
    CIFAR10_SPEC.image_size,
    CIFAR10_SPEC.image_size,
)
HIDDEN_DIM = 400
LATENT_DIM = 20
BATCH_SIZE = 100
EPOCHS = 50
LEARNING_RATE = 1e-3
NUM_WORKERS = 8


def load_data(batch_size: int, num_workers: int):
    return build_cifar10_loader(
        batch_size,
        num_workers=num_workers,
        drop_last=False,
    )


def train_epoch(model, train_loader, optimizer, device, epoch, epochs, history, logger):
    model.train()
    train_loss = 0
    train_bce = 0
    train_kld = 0
    samples_cnt = 0

    for inputs, _ in train_loader:
        inputs = inputs.to(device)
        optimizer.zero_grad(set_to_none=True)
        recon_batch, mu, logvar = model(inputs)

        bce, kld, loss = model.loss_function(recon_batch, inputs, mu, logvar)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_bce += bce.item()
        train_kld += kld.item()
        samples_cnt += inputs.size(0)

    avg_loss = train_loss / samples_cnt
    avg_bce = train_bce / samples_cnt
    avg_kld = train_kld / samples_cnt

    history["train_loss"].append(avg_loss)
    history["train_bce"].append(avg_bce)
    history["train_kld"].append(avg_kld)

    logger.info(
        f"📈 Epoch {epoch + 1}/{epochs}  "
        f"loss={avg_loss:.4f}  bce={avg_bce:.4f}  kld={avg_kld:.4f}"
    )

    return avg_loss, avg_bce, avg_kld


def save_history(history, save_path):
    with open(save_path, "wb") as fp:
        pickle.dump(history, fp)


@torch.inference_mode()
def save_samples(model, device, save_path, num_samples=64):
    was_training = model.training
    model.eval()
    samples = model.sample(num_samples, device=device)
    save_image(samples.cpu(), save_path, nrow=8)
    model.train(was_training)


def fit(model, train_loader, optimizer, device, epochs, save_dir, logger):
    history = {
        "train_loss": [],
        "train_bce": [],
        "train_kld": [],
    }

    for epoch in range(epochs):
        train_epoch(
            model, train_loader, optimizer, device, epoch, epochs, history, logger
        )

    logger.info("💾 Saving training history")
    save_history(history, f"{save_dir}/history.pkl")
    logger.info("📷 Saving generated samples")
    save_samples(model, device, save_dir / "samples_final.png")

    return history


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = PROJECT_DIR / "output" / DATASET
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(save_dir, log_file="train.log", name="VAE")

    model = VAE(
        image_shape=IMAGE_SHAPE,
        hidden_dim=HIDDEN_DIM,
        n_latent_features=LATENT_DIM,
    )
    model.to(device)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader = load_data(BATCH_SIZE, NUM_WORKERS)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    logger.info(
        f"🚀 Start  dataset={DATASET}  device={device}  epochs={EPOCHS}  "
        f"bs={BATCH_SIZE}  lr={LEARNING_RATE}  latent={LATENT_DIM}  hidden={HIDDEN_DIM}  "
        f"n_samples={len(train_loader.dataset)}  out={save_dir}"
    )

    history = fit(model, train_loader, optimizer, device, EPOCHS, save_dir, logger)

    logger.info(
        f"✅ Done  loss={history['train_loss'][-1]:.4f}  "
        f"bce={history['train_bce'][-1]:.4f}  kld={history['train_kld'][-1]:.4f}"
    )


if __name__ == "__main__":
    main()
