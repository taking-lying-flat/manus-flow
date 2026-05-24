from collections import namedtuple
from typing import Optional

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from torchvision.models import VGG16_Weights

from einops import reduce

from .utils import default

LossBreakdown = namedtuple(
    "LossBreakdown", ["total", "main", "data_match", "velocity_match"]
)


class MSELoss(nn.Module):
    def forward(self, pred: Tensor, target: Tensor, **kwargs) -> Tensor:
        return F.mse_loss(pred, target)


class PseudoHuberLoss(nn.Module):
    def __init__(self, data_dim: int = 3):
        super().__init__()
        self.data_dim = data_dim

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        reduction: str = "mean",
        data_dim: Optional[int] = None,
        **kwargs,
    ) -> Tensor:
        dim = default(data_dim, self.data_dim)
        c = 0.00054 * dim
        loss = (F.mse_loss(pred, target, reduction=reduction) + c * c).sqrt() - c
        if reduction == "none":
            loss = reduce(loss, "b ... -> b", "mean")
        return loss


class LPIPSLoss(nn.Module):
    def __init__(
        self,
        vgg: Optional[nn.Module] = None,
        vgg_weights: VGG16_Weights = VGG16_Weights.DEFAULT,
    ):
        super().__init__()
        if not vgg:
            import torchvision
            vgg = torchvision.models.vgg16(weights=vgg_weights)
            vgg.classifier = nn.Sequential(*vgg.classifier[:-2])
        self.vgg = nn.ModuleList([vgg])

    def forward(
        self, pred_data: Tensor, data: Tensor, reduction: str = "mean"
    ) -> Tensor:
        vgg, = self.vgg
        vgg = vgg.to(data.device)
        pred_embed, embed = map(vgg, (pred_data, data))
        loss = F.mse_loss(embed, pred_embed, reduction=reduction)
        if reduction == "none":
            loss = reduce(loss, "b ... -> b", "mean")
        return loss


class PseudoHuberLossWithLPIPS(nn.Module):
    def __init__(self, data_dim: int = 3, lpips_kwargs: Optional[dict] = None):
        super().__init__()
        self.pseudo_huber = PseudoHuberLoss(data_dim)
        self.lpips = LPIPSLoss(**(lpips_kwargs or {}))

    def forward(
        self,
        pred_flow: Tensor,
        target_flow: Tensor,
        *,
        pred_data: Tensor,
        times: Tensor,
        data: Tensor,
    ) -> Tensor:
        huber_loss = self.pseudo_huber(
            pred_flow, target_flow, reduction="none"
        )
        lpips_loss = self.lpips(data, pred_data, reduction="none")
        # LPIPS dominates at low t (noisy inputs). VGG features on noisy
        # images are less meaningful, but the SD3-style 1/t weighting biases
        # the loss toward intermediate timesteps where flow is hardest.
        time_weighted_loss = huber_loss * times + lpips_loss * (
            1.0 / times.clamp(min=1e-1)
        )
        return time_weighted_loss.mean()


class MeanVarianceNetLoss(nn.Module):
    def forward(self, pred, target: Tensor, **kwargs) -> Tensor:
        dist = Normal(*pred)
        return -dist.log_prob(target).mean()


def build_loss_fn(name: str, **kwargs) -> nn.Module:
    if name == "mse":
        return MSELoss()
    if name == "pseudo_huber":
        return PseudoHuberLoss(**kwargs)
    if name == "pseudo_huber_with_lpips":
        return PseudoHuberLossWithLPIPS(**kwargs)
    if name == "mean_variance":
        return MeanVarianceNetLoss()
    raise ValueError(f"Unknown loss function: {name}")
