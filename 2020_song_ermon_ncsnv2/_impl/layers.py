from __future__ import annotations

from collections.abc import Callable, Sequence

import torch.nn.functional as F
from torch import Tensor, nn

from _impl.config import RuntimeConfig

Activation = Callable[[Tensor], Tensor]
Normalization = Callable[[int], nn.Module]


def get_act(config: RuntimeConfig) -> nn.Module:
    match config.model.nonlinearity.lower():
        case "elu":
            return nn.ELU()
        case "relu":
            return nn.ReLU()
        case "lrelu":
            return nn.LeakyReLU(negative_slope=0.2)
        case "swish" | "silu":
            return nn.SiLU()
        case name:
            raise NotImplementedError(f"activation function does not exist: {name}")


def conv1x1(
    in_channels: int,
    out_channels: int,
    *,
    bias: bool = True,
) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)


def conv3x3(
    in_channels: int,
    out_channels: int,
    *,
    bias: bool = True,
    dilation: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        padding=dilation,
        dilation=dilation,
        bias=bias,
    )


class ConvMeanPool(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.avg_pool2d(self.conv(x), kernel_size=2, stride=2)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        resample: str | None = None,
        act: Activation = nn.ELU(),
        normalization: Normalization = nn.BatchNorm2d,
        dilation: int | None = None,
    ) -> None:
        super().__init__()
        self.act = act
        self.normalize1 = normalization(input_dim)

        if resample == "down" and dilation is None:
            self.conv1 = conv3x3(input_dim, input_dim)
            self.normalize2 = normalization(input_dim)
            self.conv2 = ConvMeanPool(
                input_dim,
                output_dim,
                kernel_size=3,
            )
            self.shortcut = ConvMeanPool(
                input_dim,
                output_dim,
                kernel_size=1,
            )

        elif resample == "down" and dilation is not None:
            self.conv1 = conv3x3(input_dim, input_dim, dilation=dilation)
            self.normalize2 = normalization(input_dim)
            self.conv2 = conv3x3(input_dim, output_dim, dilation=dilation)
            self.shortcut = conv3x3(input_dim, output_dim, dilation=dilation)

        elif resample is None:
            conv_dilation = dilation or 1
            self.conv1 = conv3x3(input_dim, output_dim, dilation=conv_dilation)
            self.normalize2 = normalization(output_dim)
            self.conv2 = conv3x3(output_dim, output_dim, dilation=conv_dilation)
            self.shortcut = (
                nn.Identity()
                if input_dim == output_dim
                else conv1x1(input_dim, output_dim)
            )

        else:
            raise ValueError(f"invalid resample value: {resample}")

    def forward(self, x: Tensor) -> Tensor:
        h = self.normalize1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = self.normalize2(h)
        h = self.act(h)
        h = self.conv2(h)
        return self.shortcut(x) + h


class RCUBlock(nn.Module):
    def __init__(
        self,
        features: int,
        n_blocks: int,
        n_stages: int,
        *,
        act: Activation = nn.ReLU(),
    ) -> None:
        super().__init__()
        self.act = act
        self.blocks = nn.ModuleList(
            nn.ModuleList(
                conv3x3(features, features, bias=False)
                for _ in range(n_stages)
            )
            for _ in range(n_blocks)
        )

    def forward(self, x: Tensor) -> Tensor:
        for stages in self.blocks:
            residual = x
            for conv in stages:
                x = conv(self.act(x))
            x = x + residual
        return x


class CRPBlock(nn.Module):
    def __init__(
        self,
        features: int,
        n_stages: int,
        *,
        act: Activation = nn.ReLU(),
    ) -> None:
        super().__init__()
        self.act = act
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.convs = nn.ModuleList(
            conv3x3(features, features, bias=False)
            for _ in range(n_stages)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(x)
        path = x
        for conv in self.convs:
            path = conv(self.pool(path))
            x = x + path
        return x


class MSFBlock(nn.Module):
    def __init__(self, in_channels: Sequence[int], out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.convs = nn.ModuleList(
            conv3x3(channels, out_channels)
            for channels in in_channels
        )

    def forward(self, xs: Sequence[Tensor], shape: Sequence[int]) -> Tensor:
        out = xs[0].new_zeros(xs[0].shape[0], self.out_channels, *shape)
        for x, conv in zip(xs, self.convs, strict=True):
            h = conv(x)
            h = F.interpolate(h, size=shape, mode="bilinear", align_corners=False)
            out = out + h
        return out


class RefineBlock(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        *,
        act: Activation = nn.ReLU(),
        start: bool = False,
        end: bool = False,
    ) -> None:
        super().__init__()
        self.start = start
        self.adapt_convs = nn.ModuleList(
            RCUBlock(channels, 2, 2, act=act)
            for channels in in_channels
        )

        self.msf = None if start else MSFBlock(in_channels, out_channels)
        self.crp = CRPBlock(out_channels, 2, act=act)
        self.output_convs = RCUBlock(out_channels, 3 if end else 1, 2, act=act)

    def forward(self, xs: Sequence[Tensor], output_shape: Sequence[int]) -> Tensor:
        hs = [block(x) for block, x in zip(self.adapt_convs, xs, strict=True)]
        if self.start:
            h = hs[0]
        else:
            h = self.msf(hs, output_shape)

        h = self.crp(h)
        return self.output_convs(h)
