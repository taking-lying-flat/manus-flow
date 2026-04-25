import os
import pickle
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST
from bernoulli_vae import VAE
from utils import create_visualization_grid, setup_logger


DATASET = "mnist"


def get_config(dataset_name: str) -> dict:
    configs = {
        "mnist": {
            "image_shape": (1, 28, 28),
            "hidden_dim": 400,
            "lr": 1e-3,
            "batch_size": 100,
            "latent_dim": 20,
            "epochs": 50,
            "num_workers": 16,
            "save_dir": None,
        },
        "fashion-mnist": {
            "image_shape": (1, 28, 28),
            "hidden_dim": 400,
            "lr": 1e-3,
            "batch_size": 100,
            "latent_dim": 20,
            "epochs": 50,
            "num_workers": 16,
            "save_dir": None,
        },
    }
    if dataset_name not in configs:
        raise ValueError(f"Unknown or unsupported dataset: {dataset_name}")
    return configs[dataset_name]


def load_data(dataset_name: str, batch_size: int, num_workers: int):
    data_transform = transforms.Compose([transforms.ToTensor()])
    if dataset_name == "mnist":
        train = MNIST(root="./data", train=True, transform=data_transform, download=True)
    elif dataset_name == "fashion-mnist":
        train = FashionMNIST(root="./data", train=True, transform=data_transform, download=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )


def train_epoch(model, train_loader, optimizer, device, epoch, epochs, history, logger):
    model.train()
    train_loss = 0
    train_bce = 0
    train_kld = 0
    samples_cnt = 0

    for inputs, _ in train_loader:
        inputs = inputs.to(device)
        optimizer.zero_grad()
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


def save_checkpoint(model, optimizer, epoch, history, save_path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }
    torch.save(checkpoint, save_path)


def save_history(history, save_path):
    with open(save_path, "wb") as fp:
        pickle.dump(history, fp)


def fit(model, train_loader, optimizer, device, epochs, save_dir, logger):
    history = {
        "train_loss": [],
        "train_bce": [],
        "train_kld": [],
    }

    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, device, epoch, epochs, history, logger)

    logger.info("💾 Saving checkpoint and history")
    save_checkpoint(model, optimizer, epochs, history, f"{save_dir}/checkpoint_final.pth")
    save_history(history, f"{save_dir}/history.pkl")
    logger.info("📷 Saving reconstruction & sample figures")
    create_visualization_grid(model, train_loader, device, save_dir, "final")

    return history


def main():
    cfg = get_config(DATASET)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = cfg["batch_size"]
    lr = cfg["lr"]
    latent_dim = cfg["latent_dim"]
    hidden_dim = cfg["hidden_dim"]
    epochs = cfg["epochs"]
    num_workers = cfg["num_workers"]
    save_dir = cfg["save_dir"] if cfg["save_dir"] else f"output/{DATASET}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = setup_logger(save_dir, log_file="train.log")

    model = VAE(
        image_shape=cfg["image_shape"],
        hidden_dim=hidden_dim,
        n_latent_features=latent_dim,
    )
    model.to(device)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader = load_data(DATASET, batch_size=batch_size, num_workers=num_workers)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logger.info(
        f"🚀 Start  dataset={DATASET}  device={device}  epochs={epochs}  "
        f"bs={batch_size}  lr={lr}  latent={latent_dim}  hidden={hidden_dim}  "
        f"n_samples={len(train_loader.dataset)}  out={save_dir}"
    )

    history = fit(model, train_loader, optimizer, device, epochs, save_dir, logger)

    logger.info(
        f"✅ Done  loss={history['train_loss'][-1]:.4f}  "
        f"bce={history['train_bce'][-1]:.4f}  kld={history['train_kld'][-1]:.4f}"
    )


if __name__ == "__main__":
    main()
