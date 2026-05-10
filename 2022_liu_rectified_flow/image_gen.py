import argparse
import math
import os
import sys
from copy import deepcopy
from shutil import rmtree

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
sys.path.insert(0, _PARENT)

import torch
from torch import is_tensor, nn
from torch.optim import Adam
from torchvision.utils import save_image

from einops import rearrange

from rectified_flow.rectified_flow import RectifiedFlow
from rectified_flow.dataloader import build_loader
from rectified_flow.unet import Unet
from rectified_flow._impl.utils import default

UNET_CONFIGS = {
    "bedroom": {
        "dim": 128,
        "dim_mults": (1, 2, 4, 8),
        "channels": 3,
    },
    "church": {
        "dim": 128,
        "dim_mults": (1, 2, 4, 8),
        "channels": 3,
    },
    "celeba": {
        "dim": 128,
        "dim_mults": (1, 2, 4, 8),
        "channels": 3,
    },
}

DATASET_ROOTS = {
    "bedroom": "/root/autodl-tmp/lsun_bedroom",
    "church": "/root/autodl-tmp/church_outdoor",
    "celeba": "/root/autodl-tmp/celeba",
}

IMAGE_SIZES = {
    "bedroom": 64,
    "church": 64,
    "celeba": 64,
}


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    data_shape: tuple,
    device: torch.device,
    steps: int,
    out_path: str,
    num_samples: int = 64,
    sample_kwargs: dict | None = None,
) -> None:
    model.eval()
    sample_kwargs = default(sample_kwargs, {})
    num_sample_rows = int(math.sqrt(num_samples))
    assert (num_sample_rows**2) == num_samples

    sampled = model.sample(
        batch_size=num_samples,
        steps=steps,
        data_shape=data_shape,
        **sample_kwargs,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sampled = rearrange(
        sampled,
        "(row col) c h w -> c (row h) (col w)",
        row=num_sample_rows,
    )
    save_image(sampled, out_path)


def train(args: argparse.Namespace) -> None:
    project_root = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(project_root, "runs", args.dataset)
    results_dir = os.path.join(save_dir, "results")

    if args.clear_results and os.path.exists(results_dir):
        rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device(args.device)

    data_root = DATASET_ROOTS.get(args.dataset)
    image_size = IMAGE_SIZES.get(args.dataset, 64)
    if data_root is None:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    dl = build_loader(
        folder=data_root,
        image_size=image_size,
        batch_size=args.batch_size,
        augment_horizontal_flip=True,
    )

    unet_cfg = deepcopy(UNET_CONFIGS.get(args.dataset, UNET_CONFIGS["bedroom"]))
    model = Unet(**unet_cfg).to(device)

    rf = RectifiedFlow(
        model=model,
        predict=args.predict,
        loss_fn=args.loss_fn,
        timestep_sampler=args.timestep_sampler,
        timestep_sampler_kwargs=dict(
            m=args.ts_logit_m,
            s=args.ts_logit_s,
            mode_s=args.ts_mode_s,
        ),
        odeint_kwargs=dict(atol=args.atol, rtol=args.rtol, method=args.ode_method),
        use_consistency=args.use_consistency,
        consistency_decay=args.consistency_decay,
        consistency_delta_time=args.consistency_delta_time,
        clip_during_sampling=args.clip_during_sampling,
        clip_flow_during_sampling=args.clip_flow_during_sampling,
        eps=args.eps,
    ).to(device)

    optimizer = Adam(rf.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_train_steps, eta_min=0.0
    )

    step = 0
    data_shape = None

    print(f"dataset={args.dataset} device={args.device} save_dir={save_dir}")
    print(f"predict={args.predict} loss={args.loss_fn} timestep_sampler={args.timestep_sampler}")
    print(f"lr={args.lr:.6f} batch_size={args.batch_size} steps={args.num_train_steps}")

    dl_iter = cycle(dl)

    while step < args.num_train_steps:
        rf.train()

        data = next(dl_iter)
        if isinstance(data, (tuple, list)):
            data = data[0]
        data = data.to(device)

        if data_shape is None:
            data_shape = tuple(data.shape[1:])

        loss, loss_breakdown = rf(data, return_loss_breakdown=True)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rf.parameters(), args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()

        rf.post_training_step_update()

        if step % 50 == 0:
            if args.use_consistency:
                parts = " | ".join(
                    f"{k}: {v.item() if is_tensor(v) else v:.6f}"
                    for k, v in loss_breakdown._asdict().items()
                )
                print(f"step={step} {parts}")
            else:
                print(f"step={step} loss={loss.item():.6f}")

        if args.generate_every > 0 and step % args.generate_every == 0 and step > 0:
            out_path = os.path.join(results_dir, f"samples_step_{step}.png")
            generate_samples(
                rf,
                data_shape=data_shape,
                device=device,
                steps=args.num_sampling_steps,
                out_path=out_path,
            )

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0 and step > 0:
            ckpt_path = os.path.join(save_dir, f"checkpoint.{step}.pt")
            torch.save(
                {
                    "model": rf.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "step": step,
                    "data_shape": data_shape,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"saved checkpoint to {ckpt_path}")

        step += 1

    ckpt_path = os.path.join(save_dir, "model_final.pt")
    torch.save(
        {
            "model": rf.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "step": step,
            "data_shape": data_shape,
            "args": vars(args),
        },
        ckpt_path,
    )
    print(f"saved final checkpoint to {ckpt_path}")

    if args.generate_after:
        out_path = os.path.join(results_dir, "samples_final.png")
        generate_samples(
            rf,
            data_shape=data_shape,
            device=device,
            steps=args.num_sampling_steps,
            out_path=out_path,
        )

    print("training complete")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rectified Flow Training")
    parser.add_argument("--num_train_steps", type=int, default=70_000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--generate_every", type=int, default=1000)
    parser.add_argument("--num_sampling_steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--clear_results", action="store_true")
    parser.add_argument(
        "--ode_method",
        type=str,
        default="midpoint",
        choices=["euler", "midpoint", "rk4", "dopri5"],
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument(
        "--dataset",
        type=str,
        default="bedroom",
        choices=["bedroom", "church", "celeba"],
    )
    parser.add_argument(
        "--predict",
        type=str,
        default="flow",
        choices=["flow", "noise", "clean"],
    )
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="mse",
        choices=["mse", "pseudo_huber", "pseudo_huber_with_lpips"],
    )
    parser.add_argument(
        "--timestep_sampler",
        type=str,
        default="uniform",
        choices=["uniform", "logit_normal", "cosmap", "mode"],
    )
    parser.add_argument("--ts_logit_m", type=float, default=0.0)
    parser.add_argument("--ts_logit_s", type=float, default=1.0)
    parser.add_argument("--ts_mode_s", type=float, default=1.0)
    parser.add_argument("--use_consistency", action="store_true")
    parser.add_argument("--consistency_decay", type=float, default=0.9999)
    parser.add_argument("--consistency_delta_time", type=float, default=1e-3)
    parser.add_argument("--clip_during_sampling", action="store_true")
    parser.add_argument("--clip_flow_during_sampling", action="store_true")
    parser.add_argument("--eps", type=float, default=5e-3)
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument(
        "--generate_after",
        action="store_true",
        help="Generate samples after training.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train(args)
