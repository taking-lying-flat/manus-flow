from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from torch import Tensor, nn

from _impl.config import RuntimeConfig
from _impl.layers import RefineBlock, ResidualBlock, get_act
from _impl.normalization import get_normalization
from _impl.scheduler import get_sigmas

StageSpec: TypeAlias = tuple[int, bool, int | None]

_STAGES: dict[str, tuple[StageSpec, ...]] = {
    "base": (
        (1, False, None),
        (2, True, None),
        (2, True, 2),
        (2, True, 4),
    ),
    "deeper": (
        (1, False, None),
        (2, True, None),
        (2, True, None),
        (4, True, 2),
        (4, True, 4),
    ),
    "deepest": (
        (1, False, None),
        (2, True, None),
        (2, True, None),
        (2, True, None),
        (4, True, 2),
        (4, True, 4),
    ),
}


class NCSNv2(nn.Module):
    def __init__(self, config: RuntimeConfig) -> None:
        super().__init__()
        self.config = config
        self.scale_input = not config.data.logit_transform and not config.data.rescaled

        act = get_act(config)
        norm = get_normalization(config)
        ngf = int(config.model.ngf)
        channels = int(config.data.channels)

        stages = _STAGES[config.model.variant]
        stage_channels = [ngf, *(mult * ngf for mult, _, _ in stages)]

        self.register_buffer("sigmas", get_sigmas(config))
        self.begin_conv = nn.Conv2d(channels, ngf, kernel_size=3, padding=1)
        self.encoder_stages = self._build_encoder(
            stage_specs=stages,
            stage_channels=stage_channels,
            act=act,
            norm=norm,
        )
        self.refine_blocks = self._build_decoder(
            stage_channels=stage_channels,
            act=act,
        )
        self.normalizer = norm(ngf)
        self.act = act
        self.end_conv = nn.Conv2d(ngf, channels, kernel_size=3, padding=1)

    @staticmethod
    def _build_encoder(
        *,
        stage_specs: tuple[StageSpec, ...],
        stage_channels: list[int],
        act: nn.Module,
        norm: Callable[[int], nn.Module],
    ) -> nn.ModuleList:
        stages = nn.ModuleList()
        for stage_idx, (_, downsample, dilation) in enumerate(stage_specs):
            in_channels = stage_channels[stage_idx]
            out_channels = stage_channels[stage_idx + 1]
            stages.append(
                nn.Sequential(
                    ResidualBlock(
                        in_channels,
                        out_channels,
                        resample="down" if downsample else None,
                        act=act,
                        normalization=norm,
                        dilation=dilation,
                    ),
                    ResidualBlock(
                        out_channels,
                        out_channels,
                        act=act,
                        normalization=norm,
                        dilation=dilation,
                    ),
                )
            )
        return stages

    @staticmethod
    def _build_decoder(
        *,
        stage_channels: list[int],
        act: nn.Module,
    ) -> nn.ModuleList:
        blocks = nn.ModuleList()
        depth = len(stage_channels) - 1
        for block_idx in range(depth):
            skip_idx = depth - block_idx
            out_idx = skip_idx - 1
            is_start = block_idx == 0
            in_channels = [stage_channels[skip_idx]]
            if not is_start:
                in_channels.append(stage_channels[out_idx + 1])
            blocks.append(
                RefineBlock(
                    in_channels,
                    stage_channels[out_idx],
                    act=act,
                    start=is_start,
                    end=block_idx == depth - 1,
                )
            )
        return blocks

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        h = 2.0 * x - 1.0 if self.scale_input else x
        h = self.begin_conv(h)

        skips: list[Tensor] = []
        for stage in self.encoder_stages:
            h = stage(h)
            skips.append(h)

        skip_iter = reversed(skips)
        block_iter = iter(self.refine_blocks)
        first_skip = next(skip_iter)
        refined = next(block_iter)([first_skip], first_skip.shape[2:])
        for skip, block in zip(skip_iter, block_iter, strict=True):
            refined = block([skip, refined], skip.shape[2:])

        h = self.end_conv(self.act(self.normalizer(refined)))
        sigma_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return h / self.sigmas[y].reshape(sigma_shape)
