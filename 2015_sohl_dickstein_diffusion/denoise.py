import math
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_scales: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.n_scales = n_scales
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            for _ in range(n_scales)
        ])
        std = math.sqrt(1.0 / max(out_channels, 1)) / kernel_size ** 2
        for conv in self.convs:
            nn.init.normal_(conv.weight, std=std)
            nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0, w0 = x.shape[-2:]
        max_factor = 2 ** max(self.n_scales - 1, 0)
        if h0 % max_factor != 0 or w0 % max_factor != 0:
            raise ValueError(
                f"Input spatial size {(h0, w0)} must be divisible by {max_factor} "
                f"when n_scales={self.n_scales}."
            )

        accum = None
        for scale in range(self.n_scales - 1, -1, -1):
            factor = 2 ** scale
            x_down = F.avg_pool2d(x, factor, factor) if scale > 0 else x
            out = F.leaky_relu(self.convs[scale](x_down), negative_slope=0.05)
            accum = out if accum is None else accum + out
            if scale > 0:
                accum = F.interpolate(accum, scale_factor=2, mode="nearest")

        return accum / max(self.n_scales, 1)


class ConvFeatureStack(nn.Module):
    def __init__(self, n_layers: int, n_hidden: int, n_colors: int, n_scales: int):
        super().__init__()
        channels = n_colors
        layers = []
        for _ in range(n_layers):
            layers.append(MultiScaleConv2d(channels, n_hidden, n_scales))
            channels = n_hidden
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class DenseFeatureStack(nn.Module):
    def __init__(self, layer_sizes: Sequence[int], activate_last: bool = False):
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output size.")
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            is_last = i == len(layer_sizes) - 2
            if not is_last or activate_last:
                layers.append(nn.LeakyReLU(0.05))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvDenseDenoiseNet(nn.Module):
    def __init__(
        self,
        n_layers_conv: int,
        n_layers_dense_lower: int,
        n_layers_dense_upper: int,
        n_hidden_conv: int,
        n_hidden_dense_lower: int,
        n_hidden_dense_lower_output: int,
        n_hidden_dense_upper: int,
        spatial_width: int,
        n_colors: int,
        n_scales: int,
        n_temporal_basis: int,
    ):
        super().__init__()
        self.n_colors = n_colors
        self.spatial_width = spatial_width
        self.n_hidden_conv = n_hidden_conv
        self.n_hidden_dense_lower_output = n_hidden_dense_lower_output

        self.conv_stack = ConvFeatureStack(n_layers_conv, n_hidden_conv, n_colors, n_scales)

        if n_hidden_dense_lower > 0 and n_layers_dense_lower > 0:
            n_flat_in  = n_colors * spatial_width ** 2
            n_flat_out = n_hidden_dense_lower_output * spatial_width ** 2
            sizes = [n_flat_in] + [n_hidden_dense_lower] * max(n_layers_dense_lower - 1, 0) + [n_flat_out]
            self.dense_lower: nn.Module = DenseFeatureStack(sizes)
        else:
            self.dense_lower = None
            self.n_hidden_dense_lower_output = 0

        upper_in = n_hidden_conv + self.n_hidden_dense_lower_output
        n_out = n_colors * 2 * n_temporal_basis
        upper_sizes = [upper_in] + [n_hidden_dense_upper] * max(n_layers_dense_upper - 1, 0) + [n_out]
        self.dense_upper = DenseFeatureStack(upper_sizes)
        last_linear = next(m for m in reversed(list(self.dense_upper.net))
                           if isinstance(m, nn.Linear))
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_feat = self.conv_stack(x).permute(0, 2, 3, 1)
        feat = conv_feat

        if self.dense_lower is not None:
            b = x.shape[0]
            flat = x.reshape(b, self.n_colors * self.spatial_width ** 2)
            dense_feat = self.dense_lower(flat).reshape(
                b, self.spatial_width, self.spatial_width, self.n_hidden_dense_lower_output
            )
            feat = torch.cat([
                conv_feat  / math.sqrt(max(self.n_hidden_conv, 1)),
                dense_feat / math.sqrt(max(self.n_hidden_dense_lower_output, 1)),
            ], dim=-1)

        return self.dense_upper(feat)
