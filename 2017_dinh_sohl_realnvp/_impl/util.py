import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def squeeze_2x2(x, reverse=False, alt_order=False):
    if alt_order:
        n, c, h, w = x.size()
        if reverse:
            if c % 4 != 0:
                raise ValueError('Number of channels must be divisible by 4, got {}.'.format(c))
            c //= 4
        else:
            if h % 2 != 0:
                raise ValueError('Height must be divisible by 2, got {}.'.format(h))
            if w % 2 != 0:
                raise ValueError('Width must be divisible by 4, got {}.'.format(w))
        squeeze_matrix = torch.tensor([[[[1., 0.], [0., 0.]]],
                                       [[[0., 0.], [0., 1.]]],
                                       [[[0., 1.], [0., 0.]]],
                                       [[[0., 0.], [1., 0.]]]],
                                      dtype=x.dtype,
                                      device=x.device)
        perm_weight = torch.zeros((4 * c, c, 2, 2), dtype=x.dtype, device=x.device)
        for c_idx in range(c):
            slice_0 = slice(c_idx * 4, (c_idx + 1) * 4)
            slice_1 = slice(c_idx, c_idx + 1)
            perm_weight[slice_0, slice_1, :, :] = squeeze_matrix
        shuffle_channels = torch.tensor([c_idx * 4 for c_idx in range(c)]
                                        + [c_idx * 4 + 1 for c_idx in range(c)]
                                        + [c_idx * 4 + 2 for c_idx in range(c)]
                                        + [c_idx * 4 + 3 for c_idx in range(c)])
        perm_weight = perm_weight[shuffle_channels, :, :, :]

        if reverse:
            x = F.conv_transpose2d(x, perm_weight, stride=2)
        else:
            x = F.conv2d(x, perm_weight, stride=2)
    else:
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1)

        if reverse:
            if c % 4 != 0:
                raise ValueError('Number of channels {} is not divisible by 4'.format(c))
            x = x.view(b, h, w, c // 4, 2, 2)
            x = x.permute(0, 1, 4, 2, 5, 3)
            x = x.contiguous().view(b, 2 * h, 2 * w, c // 4)
        else:
            if h % 2 != 0 or w % 2 != 0:
                raise ValueError('Expected even spatial dims HxW, got {}x{}'.format(h, w))
            x = x.view(b, h // 2, 2, w // 2, 2, c)
            x = x.permute(0, 1, 3, 5, 2, 4)
            x = x.contiguous().view(b, h // 2, w // 2, c * 4)

        x = x.permute(0, 3, 1, 2)

    return x


def checkerboard_mask(height, width, reverse=False, dtype=torch.float32,
                      device=None, requires_grad=False):
    checkerboard = [[((i % 2) + j) % 2 for j in range(width)] for i in range(height)]
    mask = torch.tensor(checkerboard, dtype=dtype, device=device, requires_grad=requires_grad)

    if reverse:
        mask = 1 - mask

    mask = mask.view(1, 1, height, width)

    return mask


def get_param_groups(net, weight_decay, norm_suffix='parametrizations.weight.original0', verbose=False):
    norm_params = []
    unnorm_params = []
    for n, p in net.named_parameters():
        if n.endswith(norm_suffix):
            norm_params.append(p)
        else:
            unnorm_params.append(p)

    param_groups = [{'name': 'normalized', 'params': norm_params, 'weight_decay': weight_decay},
                    {'name': 'unnormalized', 'params': unnorm_params}]

    if verbose:
        print('{} normalized parameters'.format(len(norm_params)))
        print('{} unnormalized parameters'.format(len(unnorm_params)))

    return param_groups


class WNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        super(WNConv2d, self).__init__()
        self.conv = nn.utils.parametrizations.weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias))

    def forward(self, x):
        x = self.conv(x)
        return x


def bits_per_dim(x, nll):
    dim = np.prod(x.size()[1:])
    bpd = nll / (np.log(2) * dim)
    return bpd

