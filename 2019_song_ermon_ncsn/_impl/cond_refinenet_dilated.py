from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

NormFactory: TypeAlias = type[nn.Module]


def _dilated_conv3x3(
    in_dim: int,
    out_dim: int,
    dilation: int,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_dim,
        out_dim,
        kernel_size=3,
        padding=dilation,
        dilation=dilation,
    )


class ConditionalInstanceNorm2dPlus(nn.Module):
    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        if num_features <= 0 or num_classes <= 0:
            raise ValueError("num_features and num_classes must be positive")

        self.num_features = num_features
        self.instance_norm = nn.InstanceNorm2d(
            num_features,
            affine=False,
            track_running_stats=False,
        )
        self.embed = nn.Embedding(num_classes, num_features * 3)
        nn.init.normal_(
            self.embed.weight[:, : 2 * num_features],
            mean=1.0,
            std=0.02,
        )
        nn.init.zeros_(self.embed.weight[:, 2 * num_features :])

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if y.dtype != torch.long:
            y = y.long()

        channel_means = x.mean(dim=(2, 3))
        ch_mean = channel_means.mean(dim=-1, keepdim=True)
        ch_var = channel_means.var(dim=-1, keepdim=True, unbiased=False)
        channel_means = (channel_means - ch_mean) * torch.rsqrt(ch_var + 1e-5)

        h = self.instance_norm(x)
        gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)
        h = torch.addcmul(h, channel_means[:, :, None, None], alpha[:, :, None, None])
        return gamma[:, :, None, None] * h + beta[:, :, None, None]


class ConvMeanPool(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        adjust_padding: bool = False,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        _pad = (kernel_size // 2) * dilation
        if adjust_padding:
            self.conv: nn.Module = nn.Sequential(
                nn.ZeroPad2d((1, 0, 1, 0)),
                nn.Conv2d(in_dim, out_dim, kernel_size, padding=_pad, dilation=dilation),
            )
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, padding=_pad, dilation=dilation)

    def forward(self, x: Tensor) -> Tensor:
        return F.avg_pool2d(self.conv(x), 2)


class ConditionalResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_classes: int,
        *,
        resample: str | None = None,
        act: nn.Module | None = None,
        norm_cls: NormFactory = ConditionalInstanceNorm2dPlus,
        adjust_padding: bool = False,
        dilation: int | None = None,
    ) -> None:
        super().__init__()
        if resample not in (None, "down"):
            raise ValueError(f"unsupported resample mode: {resample}")
        if dilation is not None and dilation <= 0:
            raise ValueError("dilation must be positive")

        self.act = nn.ELU() if act is None else act
        self.norm1 = norm_cls(in_dim, num_classes)
        need_shortcut = out_dim != in_dim or resample is not None

        if resample == "down" and dilation is None:
            self.conv1 = nn.Conv2d(in_dim, in_dim, 3, padding=1)
            self.norm2 = norm_cls(in_dim, num_classes)
            self.conv2 = ConvMeanPool(
                in_dim,
                out_dim,
                3,
                adjust_padding=adjust_padding,
            )
            self.shortcut: nn.Module | None = ConvMeanPool(
                in_dim,
                out_dim,
                1,
                adjust_padding=adjust_padding,
            )
        elif resample == "down" and dilation is not None:
            self.conv1 = nn.Conv2d(in_dim, in_dim, 3, padding=1)
            self.norm2 = norm_cls(in_dim, num_classes)
            self.conv2 = ConvMeanPool(
                in_dim,
                out_dim,
                3,
                adjust_padding=adjust_padding,
                dilation=dilation,
            )
            self.shortcut = ConvMeanPool(
                in_dim,
                out_dim,
                1,
                adjust_padding=adjust_padding,
            )
        elif dilation is not None:
            first_out = in_dim if resample == "down" else out_dim
            self.conv1 = _dilated_conv3x3(in_dim, first_out, dilation)
            self.norm2 = norm_cls(first_out, num_classes)
            self.conv2 = _dilated_conv3x3(first_out, out_dim, dilation)
            self.shortcut = (
                _dilated_conv3x3(in_dim, out_dim, dilation)
                if need_shortcut
                else None
            )
        else:
            self.conv1 = nn.Conv2d(in_dim, out_dim, 3, padding=1)
            self.norm2 = norm_cls(out_dim, num_classes)
            self.conv2 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
            self.shortcut = (
                nn.Conv2d(in_dim, out_dim, 1) if need_shortcut else None
            )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        h = self.conv1(self.act(self.norm1(x, y)))
        h = self.conv2(self.act(self.norm2(h, y)))
        residual = self.shortcut(x) if self.shortcut is not None else x
        return residual + h


class CondCRPBlock(nn.Module):
    def __init__(
        self,
        features: int,
        n_stages: int,
        num_classes: int,
        norm_cls: NormFactory,
    ) -> None:
        super().__init__()
        self.act = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.convs = nn.ModuleList(
            nn.Conv2d(features, features, 3, padding=1, bias=False)
            for _ in range(n_stages)
        )
        self.norms = nn.ModuleList(
            norm_cls(features, num_classes) for _ in range(n_stages)
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.act(x)
        path = x
        for norm, conv in zip(self.norms, self.convs):
            path = conv(self.pool(norm(path, y)))
            x = x + path
        return x


class CondRCUBlock(nn.Module):
    def __init__(
        self,
        features: int,
        n_blocks: int,
        n_stages: int,
        num_classes: int,
        norm_cls: NormFactory,
    ) -> None:
        super().__init__()
        self.act = nn.ReLU()
        self.norms = nn.ModuleList(
            nn.ModuleList(
                norm_cls(features, num_classes) for _ in range(n_stages)
            )
            for _ in range(n_blocks)
        )
        self.convs = nn.ModuleList(
            nn.ModuleList(
                nn.Conv2d(features, features, 3, padding=1, bias=False)
                for _ in range(n_stages)
            )
            for _ in range(n_blocks)
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        for norms, convs in zip(self.norms, self.convs):
            residual = x
            for norm, conv in zip(norms, convs):
                x = conv(self.act(norm(x, y)))
            x = x + residual
        return x


class CondMSFBlock(nn.Module):
    def __init__(
        self,
        in_planes: Sequence[int],
        features: int,
        num_classes: int,
        norm_cls: NormFactory,
    ) -> None:
        super().__init__()
        self.features = features
        self.convs = nn.ModuleList(
            nn.Conv2d(in_channels, features, 3, padding=1)
            for in_channels in in_planes
        )
        self.norms = nn.ModuleList(
            norm_cls(in_channels, num_classes) for in_channels in in_planes
        )

    def forward(
        self,
        xs: Sequence[Tensor],
        y: Tensor,
        shape: tuple[int, int],
    ) -> Tensor:
        if len(xs) != len(self.convs):
            raise ValueError(f"expected {len(self.convs)} feature maps, got {len(xs)}")

        out = xs[0].new_zeros((xs[0].shape[0], self.features, *shape))
        for conv, norm, x in zip(self.convs, self.norms, xs):
            out = out + F.interpolate(
                conv(norm(x, y)),
                size=shape,
                mode="bilinear",
                align_corners=True,
            )
        return out


class CondRefineBlock(nn.Module):
    def __init__(
        self,
        in_planes: Sequence[int],
        features: int,
        num_classes: int,
        norm_cls: NormFactory,
        *,
        start: bool = False,
        end: bool = False,
    ) -> None:
        super().__init__()
        if not in_planes:
            raise ValueError("in_planes must not be empty")
        if start and len(in_planes) != 1:
            raise ValueError("a start refine block must have exactly one input")

        self.start = start
        self.adapt_convs = nn.ModuleList(
            CondRCUBlock(channels, 2, 2, num_classes, norm_cls)
            for channels in in_planes
        )
        self.msf = (
            None
            if start
            else CondMSFBlock(in_planes, features, num_classes, norm_cls)
        )
        self.crp = CondCRPBlock(features, 2, num_classes, norm_cls)
        self.output_convs = CondRCUBlock(
            features,
            3 if end else 1,
            2,
            num_classes,
            norm_cls,
        )

    def forward(
        self,
        xs: Sequence[Tensor],
        y: Tensor,
        output_shape: tuple[int, int],
    ) -> Tensor:
        if len(xs) != len(self.adapt_convs):
            raise ValueError(
                f"expected {len(self.adapt_convs)} feature maps, got {len(xs)}"
            )
        hs = [conv(x, y) for conv, x in zip(self.adapt_convs, xs)]
        h = hs[0] if self.start else self.msf(hs, y, output_shape)
        return self.output_convs(self.crp(h, y), y)


class CondRefineNetDilated(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        channels: int,
        ngf: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        if image_size <= 0 or channels <= 0 or ngf <= 0 or num_classes <= 0:
            raise ValueError("all model dimensions must be positive")

        self.channels = channels
        self.num_classes = num_classes
        self.act = nn.ELU()
        norm_cls = ConditionalInstanceNorm2dPlus

        self.begin_conv = nn.Conv2d(channels, ngf, 3, padding=1)
        self.normalizer = norm_cls(ngf, num_classes)
        self.end_conv = nn.Conv2d(ngf, channels, 3, padding=1)

        def residual_group(
            in_dim: int,
            out_dim: int,
            *,
            resample: str | None = None,
            dilation: int | None = None,
            adjust_padding: bool = False,
        ) -> nn.ModuleList:
            return nn.ModuleList(
                [
                    ConditionalResidualBlock(
                        in_dim,
                        out_dim,
                        num_classes,
                        resample=resample,
                        act=nn.ELU(),
                        norm_cls=norm_cls,
                        adjust_padding=adjust_padding,
                        dilation=dilation,
                    ),
                    ConditionalResidualBlock(
                        out_dim,
                        out_dim,
                        num_classes,
                        resample=None,
                        act=nn.ELU(),
                        norm_cls=norm_cls,
                        dilation=dilation,
                    ),
                ]
            )

        self.res1 = residual_group(ngf, ngf)
        self.res2 = residual_group(ngf, 2 * ngf, resample="down")
        self.res3 = residual_group(
            2 * ngf,
            2 * ngf,
            resample="down",
            dilation=2,
        )
        self.res4 = residual_group(
            2 * ngf,
            2 * ngf,
            resample="down",
            dilation=4,
            adjust_padding=image_size == 28,
        )

        self.refine1 = CondRefineBlock(
            [2 * ngf],
            2 * ngf,
            num_classes,
            norm_cls,
            start=True,
        )
        self.refine2 = CondRefineBlock(
            [2 * ngf, 2 * ngf],
            2 * ngf,
            num_classes,
            norm_cls,
        )
        self.refine3 = CondRefineBlock(
            [2 * ngf, 2 * ngf],
            ngf,
            num_classes,
            norm_cls,
        )
        self.refine4 = CondRefineBlock(
            [ngf, ngf],
            ngf,
            num_classes,
            norm_cls,
            end=True,
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected BCHW input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {x.shape[1]}"
            )
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("labels must have shape [batch_size]")

        h = self.begin_conv(2.0 * x - 1.0)
        layer1 = self._run(self.res1, h, y)
        layer2 = self._run(self.res2, layer1, y)
        layer3 = self._run(self.res3, layer2, y)
        layer4 = self._run(self.res4, layer3, y)

        ref1 = self.refine1([layer4], y, layer4.shape[-2:])
        ref2 = self.refine2([layer3, ref1], y, layer3.shape[-2:])
        ref3 = self.refine3([layer2, ref2], y, layer2.shape[-2:])
        h = self.refine4([layer1, ref3], y, layer1.shape[-2:])
        return self.end_conv(self.act(self.normalizer(h, y)))

    @staticmethod
    def _run(blocks: nn.ModuleList, x: Tensor, y: Tensor) -> Tensor:
        for block in blocks:
            x = block(x, y)
        return x
