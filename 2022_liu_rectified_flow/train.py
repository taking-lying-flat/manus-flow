import argparse
import sys
from pathlib import Path

import torch
from _impl import RectifiedFlow, Unet
from einops import rearrange
from torch import nn, optim
from torchvision.utils import save_image

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR.parent))

from dataloader import DATASET, IMAGE_SIZE, build_cifar10_loader, infinite_batches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UNET_CONFIG = {"dim": 128, "dim_mults": (1, 2, 4)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Rectified Flow.")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--iterations", default=70000, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--max-grad-norm", default=10, type=float)
    parser.add_argument("--num-sampling-steps", default=16, type=int)
    parser.add_argument("--predict", default="flow", choices=["flow", "noise", "clean"])
    parser.add_argument(
        "--loss-fn",
        default="mse",
        choices=["mse", "pseudo_huber", "pseudo_huber_with_lpips"],
    )
    parser.add_argument(
        "--timestep-sampler",
        default="uniform",
        choices=["uniform", "logit_normal", "cosmap", "mode"],
    )
    parser.add_argument("--ts-logit-m", default=0.0, type=float)
    parser.add_argument("--ts-logit-s", default=1.0, type=float)
    parser.add_argument("--ts-mode-s", default=1.0, type=float)
    parser.add_argument(
        "--ode-method",
        default="midpoint",
        choices=["euler", "midpoint", "rk4", "dopri5"],
    )
    parser.add_argument("--atol", default=1e-5, type=float)
    parser.add_argument("--rtol", default=1e-5, type=float)
    parser.add_argument("--use-consistency", action="store_true")
    parser.add_argument("--consistency-decay", default=0.9999, type=float)
    parser.add_argument("--consistency-delta-time", default=1e-3, type=float)
    parser.add_argument(
        "--log-interval", default=25, type=int, help="steps between loss prints"
    )
    parser.add_argument(
        "--ckpt-interval",
        default=500,
        type=int,
        help="steps between checkpoint saves (overwrites)",
    )
    return parser.parse_args()


@torch.no_grad()
def save_samples(
    rf: RectifiedFlow,
    data_shape: tuple[int, ...],
    sampling_steps: int,
    step: int,
    output_dir: Path,
) -> None:
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    rf.eval()
    sampled = rf.sample(
        batch_size=64,
        steps=sampling_steps,
        data_shape=data_shape,
    ).cpu()
    rf.train()

    grid = rearrange(sampled, "(row col) c h w -> c (row h) (col w)", row=8)
    save_image(grid, sample_dir / f"sample_grid_{step:06d}.png")


def train(args: argparse.Namespace) -> None:
    output_dir = PROJECT_DIR / "runs" / DATASET
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoint.pt"

    train_loader = infinite_batches(build_cifar10_loader(args.batch_size))

    model = Unet(
        dim=UNET_CONFIG["dim"],
        dim_mults=UNET_CONFIG["dim_mults"],
        channels=3,
    ).to(device)

    rf = RectifiedFlow(
        model=model,
        predict=args.predict,
        loss_fn=args.loss_fn,
        timestep_sampler=args.timestep_sampler,
        timestep_sampler_kwargs={
            "m": args.ts_logit_m,
            "s": args.ts_logit_s,
            "mode_s": args.ts_mode_s,
        },
        odeint_kwargs={
            "atol": args.atol,
            "rtol": args.rtol,
            "method": args.ode_method,
        },
        use_consistency=args.use_consistency,
        consistency_decay=args.consistency_decay,
        consistency_delta_time=args.consistency_delta_time,
    ).to(device)

    optimizer = optim.Adam(rf.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iterations, eta_min=0.0
    )

    data_shape = None

    rf.train()
    for step in range(1, args.iterations + 1):
        data = next(train_loader)
        if isinstance(data, (tuple, list)):
            data = data[0]
        data = data.to(device)

        if data_shape is None:
            data_shape = tuple(data.shape[1:])

        loss, _ = rf(data, return_loss_breakdown=True)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(rf.parameters(), args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()

        rf.post_training_step_update()

        if step % args.log_interval == 0 or step == args.iterations:
            print(f"step={step}  loss={loss.item():.6f}")

        if step % args.ckpt_interval == 0 or step == args.iterations:
            if args.use_consistency:
                torch.save(
                    {
                        "model": rf.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "data_shape": data_shape,
                    },
                    ckpt_path,
                )
            save_samples(
                rf,
                data_shape,
                args.num_sampling_steps,
                step,
                output_dir,
            )
            saved = "EMA checkpoint + samples" if args.use_consistency else "samples"
            print(f"Step {step}: saved {saved}")


def main() -> None:
    args = parse_args()
    print(
        f"Training Rectified Flow on {DATASET} ({IMAGE_SIZE}x{IMAGE_SIZE})  device={device}"
    )
    train(args)


if __name__ == "__main__":
    main()
