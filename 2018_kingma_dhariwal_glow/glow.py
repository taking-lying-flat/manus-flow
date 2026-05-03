from __future__ import annotations

from math import log, pi
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


LOG_2_PI = log(2 * pi)


def log_abs(x: Tensor) -> Tensor:
    return torch.log(torch.abs(x))


def squeeze2d(x: Tensor) -> Tensor:
    batch, channels, height, width = x.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("Glow squeeze expects even spatial dimensions.")

    x = x.view(batch, channels, height // 2, 2, width // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4)
    return x.contiguous().view(batch, channels * 4, height // 2, width // 2)


def unsqueeze2d(x: Tensor) -> Tensor:
    batch, channels, height, width = x.shape
    if channels % 4 != 0:
        raise ValueError(
            "Glow unsqueeze expects the channel count to be divisible by 4."
        )

    x = x.view(batch, channels // 4, 2, 2, height, width)
    x = x.permute(0, 1, 4, 2, 5, 3)
    return x.contiguous().view(batch, channels // 4, height * 2, width * 2)


class ActNorm(nn.Module):
    def __init__(self, in_channel: int, logdet: bool = True) -> None:
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer("initialized", torch.tensor(False))
        self.logdet = logdet

    def initialize(self, x: Tensor) -> None:
        with torch.no_grad():
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            std = x.std(dim=(0, 2, 3), keepdim=True, unbiased=False)

            self.loc.copy_(-mean)
            self.scale.copy_(std.add(1e-6).reciprocal())

    def forward(self, x: Tensor):
        _, _, height, width = x.shape

        if not bool(self.initialized.item()):
            self.initialize(x)
            self.initialized.fill_(1)

        logdet = height * width * log_abs(self.scale).sum()

        if self.logdet:
            return self.scale * (x + self.loc), logdet

        return self.scale * (x + self.loc)

    def reverse(self, output: Tensor) -> Tensor:
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channel: int) -> None:
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.linalg.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        _, _, height, width = x.shape
        out = F.conv2d(x, self.weight)
        weight = self.weight.squeeze(-1).squeeze(-1)
        logdet = (
            height * width * torch.linalg.slogdet(weight.double()).logabsdet.float()
        )
        return out, logdet

    def reverse(self, output: Tensor) -> Tensor:
        weight = self.weight.squeeze(-1).squeeze(-1)
        inverse = torch.linalg.inv(weight).unsqueeze(2).unsqueeze(3)
        return F.conv2d(output, inverse)


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel: int) -> None:
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.linalg.qr(weight)
        w_p, w_l, w_u = torch.linalg.lu(q.float())
        w_s = torch.diagonal(w_u)
        w_u = torch.triu(w_u, diagonal=1)
        u_mask = torch.triu(torch.ones_like(w_u), diagonal=1)
        l_mask = u_mask.T

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", u_mask)
        self.register_buffer("l_mask", l_mask)
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(log_abs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        _, _, height, width = x.shape
        weight = self.calc_weight()
        out = F.conv2d(x, weight)
        logdet = height * width * torch.sum(self.w_s)
        return out, logdet

    def calc_weight(self) -> Tensor:
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ (
                (self.w_u * self.u_mask)
                + torch.diag(self.s_sign * torch.exp(self.w_s))
            )
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output: Tensor) -> Tensor:
        weight = self.calc_weight()
        inverse = (
            torch.linalg.inv(weight.squeeze(-1).squeeze(-1)).unsqueeze(2).unsqueeze(3)
        )
        return F.conv2d(output, inverse)


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, padding: int = 1) -> None:
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        out = F.pad(x, [self.padding] * 4, value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)
        return out


class AffineCoupling(nn.Module):
    def __init__(
        self, in_channel: int, filter_size: int = 512, affine: bool = True
    ) -> None:
        super().__init__()
        self.affine = affine
        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        nn.init.normal_(self.net[0].weight, 0, 0.05)
        nn.init.zeros_(self.net[0].bias)
        nn.init.normal_(self.net[2].weight, 0, 0.05)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        in_a, in_b = x.chunk(2, 1)
        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            s = torch.sigmoid(log_s + 2)
            out_b = (in_b + t) * s
            logdet = torch.log(s).view(x.shape[0], -1).sum(1)
        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output: Tensor) -> Tensor:
        out_a, out_b = output.chunk(2, 1)
        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            s = torch.sigmoid(log_s + 2)
            in_b = out_b / s - t
        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
    def __init__(
        self, in_channel: int, affine: bool = True, conv_lu: bool = True
    ) -> None:
        super().__init__()
        self.actnorm = ActNorm(in_channel)
        self.invconv = InvConv2dLU(in_channel) if conv_lu else InvConv2d(in_channel)
        self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        out, logdet = self.actnorm(x)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out)
        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2
        return out, logdet

    def reverse(self, output: Tensor) -> Tensor:
        x = self.coupling.reverse(output)
        x = self.invconv.reverse(x)
        x = self.actnorm.reverse(x)
        return x


def gaussian_log_p(x: Tensor, mean: Tensor, log_sd: Tensor) -> Tensor:
    return (
        -0.5 * LOG_2_PI
        - log_sd
        - 0.5 * (x - mean).pow(2) / torch.exp(2 * log_sd)
    )


def gaussian_sample(eps: Tensor, mean: Tensor, log_sd: Tensor) -> Tensor:
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    def __init__(
        self,
        in_channel: int,
        n_flow: int,
        split: bool = True,
        affine: bool = True,
        conv_lu: bool = True,
    ) -> None:
        super().__init__()
        squeeze_dim = in_channel * 4
        self.flows = nn.ModuleList(
            Flow(squeeze_dim, affine=affine, conv_lu=conv_lu) for _ in range(n_flow)
        )
        self.split = split
        prior_channels = in_channel * 2 if split else in_channel * 4
        self.prior = ZeroConv2d(prior_channels, prior_channels * 2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = x.shape[0]
        out = squeeze2d(x)
        logdet = x.new_zeros(())
        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(batch, -1).sum(1)

        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(batch, -1).sum(1)
            z_new = out

        return out, logdet, log_p, z_new

    def reverse(
        self,
        output: Tensor,
        eps: Optional[Tensor] = None,
        reconstruct: bool = False,
    ) -> Tensor:
        if eps is None:
            raise ValueError("eps is required when reversing a Glow block.")
        x = output
        if reconstruct:
            if self.split:
                x = torch.cat([output, eps], 1)
            else:
                x = eps
        else:
            if self.split:
                mean, log_sd = self.prior(x).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                x = torch.cat([output, z], 1)
            else:
                zero = torch.zeros_like(x)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                x = z

        for flow in reversed(self.flows):
            x = flow.reverse(x)
        return unsqueeze2d(x)


class Glow(nn.Module):
    def __init__(
        self,
        in_channel: int,
        n_flow: int,
        n_block: int,
        affine: bool = True,
        conv_lu: bool = True,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for _ in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 2
        self.blocks.append(
            Block(n_channel, n_flow, split=False, affine=affine, conv_lu=conv_lu)
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, List[Tensor]]:
        log_p_sum = x.new_zeros(x.shape[0])
        logdet = x.new_zeros(())
        out = x
        z_outs: List[Tensor] = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det
            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    def reverse(self, z_list: Sequence[Tensor], reconstruct: bool = False) -> Tensor:
        if len(z_list) != len(self.blocks):
            raise ValueError("z_list length must match the number of Glow blocks.")
        out: Optional[Tensor] = None
        for i, block in enumerate(reversed(self.blocks)):
            z = z_list[-(i + 1)]
            out = block.reverse(
                z if out is None else out,
                z,
                reconstruct=reconstruct,
            )
        if out is None:
            raise RuntimeError("Glow model has no blocks.")
        return out
