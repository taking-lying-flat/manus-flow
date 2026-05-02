import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.distributions import Independent, Normal

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from densities import DENSITIES, LOG_DENSITIES
from flow import NormalizingFlow
from loss import TemperedFreeEnergyBound


FLOW_LENGTH = 32
FLOW_TYPE = "planar"
X_LIMS = (-4, 4)
Y_LIMS = (-4, 4)

def random_normal_samples(n, dim=2):
    return torch.randn(n, dim)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--target",
        choices=["u1", "u2", "u3", "u4", "all"],
        default="all",
        help="Target density in densities.py.",
    )
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--plot_points", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta_start", type=float, default=0.01)
    parser.add_argument("--beta_warmup_steps", type=int, default=10000)
    return parser.parse_args()


def should_trigger(step, interval):
    return interval > 0 and step % interval == 0


def train_one(target_name, args):
    log_density = LOG_DENSITIES[target_name]
    bound = TemperedFreeEnergyBound(
        log_density=log_density,
        beta_start=args.beta_start,
        beta_warmup_steps=args.beta_warmup_steps,
    )
    flow = NormalizingFlow(dim=2, flow_length=FLOW_LENGTH, flow_type=FLOW_TYPE)
    base_dist = Independent(Normal(loc=torch.zeros(2), scale=torch.ones(2)), 1)
    optimizer = optim.RMSprop(
        flow.parameters(),
        lr=args.lr,
        momentum=args.momentum,
    )

    for step in range(1, args.iterations + 1):
        x0 = random_normal_samples(args.batch_size)
        flow_output = flow(x0)

        loss = bound(
            x0,
            flow_output.z,
            flow_output.log_det_sum,
            base_dist=base_dist,
            step=step,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if should_trigger(step, args.log_interval):
            print(
                "[{} {} K={}] step={}/{} loss={:.6f}".format(
                    target_name.upper(),
                    FLOW_TYPE,
                    FLOW_LENGTH,
                    step,
                    args.iterations,
                    loss.item(),
                )
            )

    with torch.no_grad():
        z0 = random_normal_samples(args.plot_points)
        flow_output = flow(z0)
    return flow_output.z.detach().cpu(), loss.item()


def evaluate_density_grid(density_fn):
    x1 = np.linspace(*X_LIMS, 300)
    x2 = np.linspace(*Y_LIMS, 300)
    gx, gy = np.meshgrid(x1, x2)
    z = torch.as_tensor(np.c_[gx.ravel(), gy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        values = density_fn(z).cpu().numpy().reshape(gx.shape)
    return values


def save_grid(results, target_names, output_path):
    fig, axes = plt.subplots(
        len(target_names),
        2,
        figsize=(8, 4 * len(target_names)),
        squeeze=False,
    )

    for row, target_name in enumerate(target_names):
        ax_density = axes[row][0]
        density_values = evaluate_density_grid(DENSITIES[target_name])
        ax_density.imshow(
            density_values,
            extent=(*X_LIMS, *Y_LIMS),
            origin="lower",
            cmap="summer",
        )
        ax_density.set_xlim(*X_LIMS)
        ax_density.set_ylim(*Y_LIMS)
        ax_density.set_aspect("equal", adjustable="box")
        ax_density.set_title("{} | true density".format(target_name.upper()))

        ax_flow = axes[row][1]
        points, loss = results[target_name]
        points = points.numpy()
        ax_flow.scatter(points[:, 0], points[:, 1], alpha=0.6, s=8)
        ax_flow.set_xlim(*X_LIMS)
        ax_flow.set_ylim(*Y_LIMS)
        ax_flow.set_aspect("equal", adjustable="box")
        ax_flow.set_title(
            "{} | {} K={} | loss={:.3f}".format(
                target_name.upper(), FLOW_TYPE, FLOW_LENGTH, loss
            )
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    target_names = list(LOG_DENSITIES.keys()) if args.target == "all" else [args.target]
    results = {}

    for target_name in target_names:
        results[target_name] = train_one(target_name, args)

    assets_dir = os.path.join(os.getcwd(), "assets")
    os.makedirs(assets_dir, exist_ok=True)
    output_path = os.path.join(assets_dir, "toy_density_grid.png")
    save_grid(results, target_names, output_path)
    print("Saved grid to {}".format(output_path))


if __name__ == "__main__":
    main()
