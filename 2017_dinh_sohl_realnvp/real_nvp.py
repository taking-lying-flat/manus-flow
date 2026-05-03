from enum import IntEnum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet import ResNet
from util import checkerboard_mask, squeeze_2x2


class MaskType(IntEnum):
    CHECKERBOARD = 0
    CHANNEL_WISE = 1


class Rescale(nn.Module):
    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x):
        x = self.weight * x
        return x


class BatchNormFlow(nn.Module):
    """Per-channel batch-norm with tractable log-determinant Jacobian.

    Forward (encode):  y = exp(log_gamma) * (x - mu) / sqrt(var + eps) + beta
    Reverse (decode):  uses running mean/var only.
    log|det J| per sample = H * W * sum_c [log_gamma_c - 0.5 * log(var_c + eps)]
    """

    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.log_gamma = nn.Parameter(torch.zeros(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def _broadcast(self, t):
        return t.view(1, -1, 1, 1)

    def forward(self, x, sldj=None, reverse=False):
        if reverse:
            mean = self._broadcast(self.running_mean)
            var = self._broadcast(self.running_var)
            gamma = self._broadcast(torch.exp(self.log_gamma))
            beta = self._broadcast(self.beta)
            x = (x - beta) / gamma * torch.sqrt(var + self.eps) + mean
            return x, sldj

        if self.training:
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(mean.detach(), alpha=self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(var.detach(), alpha=self.momentum)
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - self._broadcast(mean)) / torch.sqrt(self._broadcast(var) + self.eps)
        y = self._broadcast(torch.exp(self.log_gamma)) * x_hat + self._broadcast(self.beta)

        h, w = x.shape[2], x.shape[3]
        ldj = h * w * (self.log_gamma - 0.5 * torch.log(var + self.eps)).sum()
        if sldj is not None:
            sldj = sldj + ldj
        return y, sldj


class CouplingLayer(nn.Module):
    def __init__(self, in_channels, mid_channels, num_blocks, mask_type, reverse_mask):
        super(CouplingLayer, self).__init__()

        self.mask_type = mask_type
        self.reverse_mask = reverse_mask
        self.full_channels = in_channels

        st_in_channels = in_channels // 2 if mask_type == MaskType.CHANNEL_WISE else in_channels
        self.st_net = ResNet(st_in_channels, mid_channels, 2 * st_in_channels,
                             num_blocks=num_blocks, kernel_size=3, padding=1,
                             double_after_norm=(self.mask_type == MaskType.CHECKERBOARD))

        self.rescale = nn.utils.parametrizations.weight_norm(Rescale(st_in_channels))
        self.bn = BatchNormFlow(self.full_channels)

    def _coupling_forward(self, x, sldj):
        if self.mask_type == MaskType.CHECKERBOARD:
            b = checkerboard_mask(x.size(2), x.size(3), self.reverse_mask, device=x.device)
            x_b = x * b
            st = self.st_net(x_b)
            s, t = st.chunk(2, dim=1)
            s = self.rescale(torch.tanh(s))
            s = s * (1 - b)
            t = t * (1 - b)
            exp_s = s.exp()
            if torch.isnan(exp_s).any():
                raise RuntimeError('Scale factor has NaN entries')
            x = x * exp_s + t
            sldj = sldj + s.reshape(s.size(0), -1).sum(-1)
        else:
            if self.reverse_mask:
                x_id, x_change = x.chunk(2, dim=1)
            else:
                x_change, x_id = x.chunk(2, dim=1)
            st = self.st_net(x_id)
            s, t = st.chunk(2, dim=1)
            s = self.rescale(torch.tanh(s))
            exp_s = s.exp()
            if torch.isnan(exp_s).any():
                raise RuntimeError('Scale factor has NaN entries')
            x_change = x_change * exp_s + t
            sldj = sldj + s.reshape(s.size(0), -1).sum(-1)
            if self.reverse_mask:
                x = torch.cat((x_id, x_change), dim=1)
            else:
                x = torch.cat((x_change, x_id), dim=1)
        return x, sldj

    def _coupling_reverse(self, x):
        if self.mask_type == MaskType.CHECKERBOARD:
            b = checkerboard_mask(x.size(2), x.size(3), self.reverse_mask, device=x.device)
            x_b = x * b
            st = self.st_net(x_b)
            s, t = st.chunk(2, dim=1)
            s = self.rescale(torch.tanh(s))
            s = s * (1 - b)
            t = t * (1 - b)
            inv_exp_s = (-s).exp()
            if torch.isnan(inv_exp_s).any():
                raise RuntimeError('Scale factor has NaN entries')
            return (x - t) * inv_exp_s

        if self.reverse_mask:
            x_id, x_change = x.chunk(2, dim=1)
        else:
            x_change, x_id = x.chunk(2, dim=1)
        st = self.st_net(x_id)
        s, t = st.chunk(2, dim=1)
        s = self.rescale(torch.tanh(s))
        inv_exp_s = (-s).exp()
        if torch.isnan(inv_exp_s).any():
            raise RuntimeError('Scale factor has NaN entries')
        x_change = (x_change - t) * inv_exp_s
        if self.reverse_mask:
            return torch.cat((x_id, x_change), dim=1)
        return torch.cat((x_change, x_id), dim=1)

    def forward(self, x, sldj=None, reverse=False):
        if reverse:
            x, sldj = self.bn(x, sldj, reverse=True)
            x = self._coupling_reverse(x)
            return x, sldj

        x, sldj = self._coupling_forward(x, sldj)
        x, sldj = self.bn(x, sldj, reverse=False)
        return x, sldj


class _RealNVP(nn.Module):
    def __init__(self, scale_idx, num_scales, in_channels, mid_channels, num_blocks):
        super(_RealNVP, self).__init__()

        self.is_last_block = scale_idx == num_scales - 1

        self.in_couplings = nn.ModuleList([
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False),
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True),
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False)
        ])

        if self.is_last_block:
            self.in_couplings.append(
                CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True))
        else:
            self.out_couplings = nn.ModuleList([
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False),
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=True),
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False)
            ])
            self.next_block = _RealNVP(scale_idx + 1, num_scales, 2 * in_channels, 2 * mid_channels, num_blocks)

    def forward(self, x, sldj, reverse=False):
        if reverse:
            if not self.is_last_block:
                x = squeeze_2x2(x, reverse=False, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next_block(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, reverse=True, alt_order=True)

                x = squeeze_2x2(x, reverse=False)
                for coupling in reversed(self.out_couplings):
                    x, sldj = coupling(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)

            for coupling in reversed(self.in_couplings):
                x, sldj = coupling(x, sldj, reverse)
        else:
            for coupling in self.in_couplings:
                x, sldj = coupling(x, sldj, reverse)

            if not self.is_last_block:
                x = squeeze_2x2(x, reverse=False)
                for coupling in self.out_couplings:
                    x, sldj = coupling(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)

                x = squeeze_2x2(x, reverse=False, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next_block(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, reverse=True, alt_order=True)

        return x, sldj


class RealNVP(nn.Module):
    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8):
        super(RealNVP, self).__init__()
        self.register_buffer('data_constraint', torch.tensor([0.9], dtype=torch.float32))

        self.flows = _RealNVP(0, num_scales, in_channels, mid_channels, num_blocks)

    def forward(self, x, reverse=False):
        sldj = None
        if not reverse:
            if x.min() < 0 or x.max() > 1:
                raise ValueError('Expected x in [0, 1], got x with min/max {}/{}'
                                 .format(x.min(), x.max()))
            x, sldj = self._pre_process(x)

        x, sldj = self.flows(x, sldj, reverse)

        if reverse:
            x = self._post_process(x)
        return x, sldj

    def _pre_process(self, x):
        y = (x * 255. + torch.rand_like(x)) / 256.
        y = (2 * y - 1) * self.data_constraint
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        ldj = F.softplus(y) + F.softplus(-y) \
            - F.softplus((1. - self.data_constraint).log() - self.data_constraint.log())
        sldj = ldj.reshape(ldj.size(0), -1).sum(-1)

        return y, sldj

    def _post_process(self, y):
        y = torch.sigmoid(y)
        y = (2 * y - 1) / self.data_constraint
        y = (y + 1) / 2
        return y.clamp(0., 1.)


class RealNVPLoss(nn.Module):
    def __init__(self, k=256):
        super(RealNVPLoss, self).__init__()
        self.k = k

    def forward(self, z, sldj):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.reshape(z.size(0), -1).sum(-1) \
            - np.log(self.k) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        nll = -ll.mean()

        return nll
