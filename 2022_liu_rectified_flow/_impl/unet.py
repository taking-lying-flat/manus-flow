import math
from functools import partial
from typing import Optional, Tuple

import einx
import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange
from hyper_connections.hyper_connections_channel_first import (
    get_init_and_expand_reduce_stream_functions,
    Residual,
)
from torch import cat, nn

from .utils import default, exists


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def divisible_by(num, den):
    return (num % den) == 0


class Upsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        self.seq = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
        )

    def forward(self, x):
        return self.seq(x)


class Downsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        self.seq = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
            nn.Conv2d(dim * 4, default(dim_out, dim), 1),
        )

    def forward(self, x):
        return self.seq(x)


class ChannelRMSNorm(nn.Module):
    def __init__(self, dim, has_scale=True):
        super().__init__()
        self.scale = dim
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1)) if has_scale else 0

    def forward(self, x):
        return F.normalize(x, dim=1) * (self.gamma + 1) * self.scale


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        half_dim = dim // 2
        emb = math.log(theta) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim) * -emb)

        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x):
        emb = einx.multiply("i, j -> i j", x, self.freqs.to(x.device))
        return cat((emb.sin(), emb.cos()), dim=-1)


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(
            torch.randn(half_dim), requires_grad=not is_random
        )

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = cat((freqs.sin(), freqs.cos()), dim=-1)
        return cat((x, fouriered), dim=-1)


class Block(nn.Module):
    def __init__(self, dim, dim_out, dropout=0.0, accept_cond=False):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = ChannelRMSNorm(dim_out, has_scale=not accept_cond)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        x = x.contiguous()
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return self.dropout(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0.0):
        super().__init__()
        has_time_cond = exists(time_emb_dim)
        self.mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out * 2),
            )
            if has_time_cond
            else None
        )
        self.block1 = Block(dim, dim_out, dropout=dropout, accept_cond=has_time_cond)
        self.block2 = Block(dim_out, dim_out, accept_cond=has_time_cond)

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.norm = ChannelRMSNorm(dim)
        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            ChannelRMSNorm(dim),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = tuple(
            rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads) for t in qkv
        )
        mk, mv = tuple(repeat(t, "h c n -> b h c n", b=b) for t in self.mem_kv)
        k, v = map(partial(cat, dim=-1), ((mk, k), (mv, v)))
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = einsum(k, v, "b h d n, b h e n -> b h d e")
        out = einsum(context, q, "b h d e, b h d n -> b h e n")
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.norm = ChannelRMSNorm(dim)
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h (x y) c", h=self.heads), qkv
        )
        mk, mv = map(lambda t: repeat(t, "h n d -> b h n d", b=b), self.mem_kv)
        k, v = map(partial(cat, dim=-2), ((mk, k), (mv, v)))
        q = q * self.scale
        sim = einsum(q, k, "b h i d, b h j d -> b h i j")
        attn = sim.softmax(dim=-1)
        out = einsum(attn, v, "b h i j, b h j d -> b h i d")
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class Unet(nn.Module):
    def __init__(
        self,
        dim: int,
        init_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        channels: int = 3,
        learned_sinusoidal_cond: bool = False,
        random_fourier_features: bool = False,
        learned_sinusoidal_dim: int = 16,
        sinusoidal_pos_emb_theta: int = 10000,
        dropout: float = 0.0,
        attn_dim_head: int = 32,
        attn_heads: int = 4,
        full_attn: Optional[Tuple[bool, ...]] = None,
        num_residual_streams: int = 2,
        accept_time: bool = True,
    ):
        super().__init__()

        init_dim = default(init_dim, dim)
        dim_input = channels
        self.init_conv = nn.Conv2d(dim_input, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 4
        self.accept_time = accept_time
        self.time_mlp = None

        if accept_time:
            self.random_or_learned_sinusoidal_cond = (
                learned_sinusoidal_cond or random_fourier_features
            )
            if self.random_or_learned_sinusoidal_cond:
                sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                    learned_sinusoidal_dim, random_fourier_features
                )
                fourier_dim = learned_sinusoidal_dim + 1
            else:
                sinu_pos_emb = SinusoidalPosEmb(
                    dim, theta=sinusoidal_pos_emb_theta
                )
                fourier_dim = dim

            self.time_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(fourier_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        resnet_block = partial(
            ResnetBlock,
            time_emb_dim=time_dim if accept_time else None,
            dropout=dropout,
        )

        init_hyper_conn, self.expand_streams, self.reduce_streams = (
            get_init_and_expand_reduce_stream_functions(
                num_residual_streams, disable=num_residual_streams == 1
            )
        )
        res_conv = partial(nn.Conv2d, kernel_size=1, bias=False)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (
            (dim_in, dim_out),
            layer_full_attn,
            layer_attn_heads,
            layer_attn_dim_head,
        ) in enumerate(
            zip(in_out, full_attn, attn_heads, attn_dim_head)
        ):
            is_last = ind >= (num_resolutions - 1)
            attn_klass = Attention if layer_full_attn else LinearAttention

            self.downs.append(
                nn.ModuleList(
                    [
                        Residual(branch=resnet_block(dim_in, dim_in)),
                        Residual(branch=resnet_block(dim_in, dim_in)),
                        Residual(
                            branch=attn_klass(
                                dim_in,
                                dim_head=layer_attn_dim_head,
                                heads=layer_attn_heads,
                            )
                        ),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = init_hyper_conn(
            dim=mid_dim, branch=resnet_block(mid_dim, mid_dim)
        )
        self.mid_attn = init_hyper_conn(
            dim=mid_dim,
            branch=Attention(
                mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1]
            ),
        )
        self.mid_block2 = init_hyper_conn(
            dim=mid_dim, branch=resnet_block(mid_dim, mid_dim)
        )

        for ind, (
            (dim_in, dim_out),
            layer_full_attn,
            layer_attn_heads,
            layer_attn_dim_head,
        ) in enumerate(
            zip(
                *map(
                    reversed,
                    (in_out, full_attn, attn_heads, attn_dim_head),
                )
            )
        ):
            is_last = ind == (len(in_out) - 1)
            attn_klass = Attention if layer_full_attn else LinearAttention

            self.ups.append(
                nn.ModuleList(
                    [
                        Residual(
                            branch=resnet_block(dim_out + dim_in, dim_out),
                            residual_transform=res_conv(dim_out + dim_in, dim_out),
                        ),
                        Residual(
                            branch=resnet_block(dim_out + dim_in, dim_out),
                            residual_transform=res_conv(dim_out + dim_in, dim_out),
                        ),
                        Residual(
                            branch=attn_klass(
                                dim_out,
                                dim_head=layer_attn_dim_head,
                                heads=layer_attn_heads,
                            )
                        ),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = Residual(
            branch=resnet_block(init_dim * 2, init_dim),
            residual_transform=res_conv(init_dim * 2, init_dim),
        )

        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, times=None):
        assert all(
            [divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]
        ), (
            f"your input dimensions {x.shape[-2:]} need to be divisible by "
            f"{self.downsample_factor}, given the unet"
        )

        x = self.init_conv(x)
        r = x

        assert not (
            exists(times) and not self.accept_time
        ), "time cannot be passed into Unet when `accept_time` is set to False"

        t = self.time_mlp(times) if exists(self.time_mlp) and exists(times) else None

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.expand_streams(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        x = self.reduce_streams(x)

        for block1, block2, attn, upsample in self.ups:
            x = cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)
