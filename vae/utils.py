import logging
import os
import torch
from torchvision.utils import save_image


def setup_logger(log_dir, log_file="train.log", name="VAE"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
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


def visualize_reconstruction(model, dataloader, device, save_path, num_images=8):
    model.eval()

    with torch.no_grad():
        inputs, _ = next(iter(dataloader))
        inputs = inputs[:num_images].to(device)
        recon, _, _ = model(inputs)

        comparison = torch.cat([inputs, recon])
        save_image(comparison.cpu(), save_path, nrow=num_images, normalize=True, pad_value=1)


def visualize_samples(model, device, save_path, num_samples=64):
    model.eval()

    with torch.no_grad():
        samples = model.sample(num_samples, device=device)
        save_image(samples.cpu(), save_path, nrow=8, normalize=True, pad_value=1)


def create_visualization_grid(model, dataloader, device, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    visualize_reconstruction(
        model, dataloader, device, f"{save_dir}/reconstruction_epoch_{epoch}.png", num_images=8
    )
    visualize_samples(model, device, f"{save_dir}/samples_epoch_{epoch}.png", num_samples=64)
