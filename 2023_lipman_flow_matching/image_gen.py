import argparse
import sys
from pathlib import Path

import torch
from _impl.path import AffineProbPath, CondOTProbPath
from _impl.scheduler import (
    CondOTScheduler,
    CosineScheduler,
    LinearVPScheduler,
    PolynomialConvexScheduler,
    VPScheduler,
)
from _impl.solver import ODESolver
from _impl.utils import ModelWrapper
from torch import nn
from torchvision import utils as tv_utils
from unet import UNetModel

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR.parent))

from dataloader import CIFAR10_SPEC, DATASET, build_cifar10_loader

UNET_CONFIG = {
    "base_channels": 128,
    "num_res_blocks": 2,
    "attention_resolutions": (4, 8, 16),
    "dropout": 0.1,
    "channel_mult": (1, 2, 4),
    "num_heads": 4,
}


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    spec,
    device: torch.device,
    steps: int,
    ode_method: str,
    out_path: Path,
    num_samples: int = 64,
) -> None:
    model.eval()
    wrapper = ModelWrapper(model)
    solver = ODESolver(velocity_model=wrapper)

    x_init = torch.randn(
        num_samples, spec.in_channels, spec.image_size, spec.image_size, device=device
    )
    time_grid = torch.tensor([0.0, 1.0], device=device)
    samples = solver.sample(
        x_init=x_init, step_size=1.0 / steps, method=ode_method, time_grid=time_grid
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tv_utils.save_image(samples, out_path, nrow=8, normalize=True, value_range=(-1, 1))


def build_scheduler(name: str, params: dict):
    if name == "condot":
        return CondOTScheduler()
    if name == "polynomial":
        return PolynomialConvexScheduler(n=params.get("n", 1.0))
    if name == "vp":
        return VPScheduler(
            beta_min=params.get("beta_min", 0.1),
            beta_max=params.get("beta_max", 20.0),
        )
    if name == "linear_vp":
        return LinearVPScheduler()
    if name == "cosine":
        return CosineScheduler()
    raise ValueError(f"Unknown scheduler {name}")


def build_path(args: argparse.Namespace):
    scheduler_params = {
        "n": args.scheduler_n,
        "beta_min": args.scheduler_beta_min,
        "beta_max": args.scheduler_beta_max,
    }
    scheduler = build_scheduler(args.scheduler, scheduler_params)
    name = args.path
    if name == "condot":
        return CondOTProbPath()
    if name == "affine":
        return AffineProbPath(scheduler)
    raise ValueError(f"Unknown path {name}")


def train(args: argparse.Namespace) -> None:
    save_dir = PROJECT_DIR / "runs" / DATASET

    device = torch.device(args.device)

    loader = build_cifar10_loader(
        args.batch_size,
        value_range="minus_one_one",
    )
    spec = CIFAR10_SPEC
    model = UNetModel(
        image_size=spec.image_size,
        in_channels=spec.in_channels,
        model_channels=UNET_CONFIG["base_channels"],
        out_channels=spec.in_channels,
        num_res_blocks=UNET_CONFIG["num_res_blocks"],
        attention_resolutions=UNET_CONFIG["attention_resolutions"],
        dropout=UNET_CONFIG["dropout"],
        channel_mult=UNET_CONFIG["channel_mult"],
        num_heads=UNET_CONFIG["num_heads"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0.0
    )
    loss_fn = nn.MSELoss()
    path = build_path(args)

    save_dir.mkdir(parents=True, exist_ok=True)
    step = 0

    print(f"dataset={DATASET} device={args.device} save_dir={save_dir}")
    print(
        f"path={args.path} scheduler={args.scheduler} ode_method={args.ode_method} lr={args.lr:.6f}"
    )

    for epoch in range(args.epochs):
        model.train()
        for x_1, _ in loader:
            x_1 = x_1.to(device)
            x_0 = torch.randn_like(x_1)
            t = torch.rand(x_1.shape[0], device=device)

            path_sample = path.sample(x_0=x_0, x_1=x_1, t=t)
            pred = model(path_sample.x_t, t)
            loss = loss_fn(pred, path_sample.dx_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.6f}")

            if (
                args.num_generate_steps_interval > 0
                and step % args.num_generate_steps_interval == 0
            ):
                out_path = save_dir / f"samples_step_{step}.png"
                generate_samples(
                    model,
                    spec,
                    device,
                    args.num_sampling_ode_steps,
                    args.ode_method,
                    out_path,
                )

            step += 1

        lr_scheduler.step()
        print(f"epoch={epoch} done")

    ckpt_path = save_dir / "model_final.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": args.epochs - 1,
            "args": vars(args),
        },
        ckpt_path,
    )
    print(f"saved final checkpoint to {ckpt_path}")

    if args.generate_after:
        out_path = save_dir / "samples_final.png"
        generate_samples(
            model,
            spec,
            device,
            args.num_sampling_ode_steps,
            args.ode_method,
            out_path,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flow Matching Training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num_generate_steps_interval", type=int, default=1000)
    parser.add_argument("--num_sampling_ode_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--ode_method",
        type=str,
        default="euler",
        choices=["euler", "dopri5", "midpoint", "heun3"],
    )
    parser.add_argument(
        "--path",
        type=str,
        default="condot",
        choices=["condot", "affine"],
        help="Path type for image generation.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="condot",
        choices=["condot", "polynomial", "vp", "linear_vp", "cosine"],
        help="Scheduler type for affine path.",
    )
    parser.add_argument(
        "--scheduler_n",
        type=float,
        default=1.0,
        help="Polynomial scheduler exponent (for scheduler=polynomial).",
    )
    parser.add_argument(
        "--scheduler_beta_min",
        type=float,
        default=0.1,
        help="VP scheduler beta_min (for scheduler=vp).",
    )
    parser.add_argument(
        "--scheduler_beta_max",
        type=float,
        default=20.0,
        help="VP scheduler beta_max (for scheduler=vp).",
    )
    parser.add_argument(
        "--generate_after",
        action="store_true",
        help="Generate samples after training.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train(args)
