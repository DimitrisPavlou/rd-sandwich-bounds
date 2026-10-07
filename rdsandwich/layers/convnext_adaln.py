"""ConvNeXt blocks with lambda-conditioned AdaLN, and the conv helpers around them.

Ported from Duan et al.'s ``lossy-vae`` (``lvae/models/common.py`` and
``lvae/models/rd/model.py``, https://github.com/duanzhiihao/lossy-vae), used by
:mod:`rdsandwich.models.upper_bound.variable_rate_lossy_vae`. Attribute names follow the
original (including ``ConvNeXtAdaLNPatchDown.downsapmle``) so state dicts are
interchangeable with Duan's checkpoints.

``ConvNeXtBlockAdaLN`` is a ConvNeXt block (depthwise conv -> LayerNorm -> MLP,
layer-scaled residual) whose LayerNorm shift/scale are predicted from an
embedding of the rate-distortion multiplier lambda (adaptive LayerNorm), which
makes a single network conditional on lambda (variable-rate training).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def get_conv(in_ch, out_ch, kernel_size, stride, padding, zero_bias=True, zero_weights=False):
    conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
    if zero_bias:
        conv.bias.data.mul_(0.0)
    if zero_weights:
        conv.weight.data.mul_(0.0)
    return conv


def conv_k1s1(in_ch, out_ch, zero_bias=True, zero_weights=False):
    return get_conv(in_ch, out_ch, 1, 1, 0, zero_bias, zero_weights)


def conv_k3s1(in_ch, out_ch, zero_bias=True, zero_weights=False):
    return get_conv(in_ch, out_ch, 3, 1, 1, zero_bias, zero_weights)


def patch_downsample(in_ch, out_ch, rate=2):
    """Non-overlapping ``rate x rate`` patch embedding (conv with kernel = stride = rate)."""
    return get_conv(in_ch, out_ch, kernel_size=rate, stride=rate, padding=0)


def patch_upsample(in_ch, out_ch, rate=2):
    """1x1 conv to ``out_ch * rate^2`` channels followed by pixel shuffle (sub-pixel conv)."""
    return nn.Sequential(
        get_conv(in_ch, out_ch * (rate ** 2), kernel_size=1, stride=1, padding=0),
        nn.PixelShuffle(rate),
    )


def sinusoidal_embedding(values: torch.Tensor, dim=256, max_period=64):
    """Transformer-style sinusoidal embedding of a 1-D tensor of scalars -> ``[N, dim]``."""
    assert values.dim() == 1 and (dim % 2) == 0
    exponents = torch.linspace(0, 1, steps=(dim // 2))
    freqs = torch.pow(max_period, -1.0 * exponents).to(device=values.device)
    args = values.view(-1, 1) * freqs.view(1, dim // 2)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class Mlp(nn.Module):
    """Two-layer MLP on the last dimension; stands in for ``timm.layers.Mlp`` (same
    ``fc1`` / ``fc2`` parameter names, no dropout, no norm)."""

    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class ConvNeXtBlockAdaLN(nn.Module):
    default_embedding_dim = 256

    def __init__(self, dim, embed_dim=None, out_dim=None, kernel_size=7, mlp_ratio=2,
                 residual=True, ls_init_value=1e-6):
        super().__init__()
        # depthwise conv
        pad = (kernel_size - 1) // 2
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=pad, groups=dim)
        # layer norm (affine params come from the lambda embedding instead)
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        # AdaLN
        embed_dim = embed_dim or self.default_embedding_dim
        self.embedding_layer = nn.Sequential(
            nn.GELU(),
            nn.Linear(embed_dim, 2 * dim),
            nn.Unflatten(1, unflattened_size=(1, 1, 2 * dim)),
        )
        # MLP
        hidden = int(mlp_ratio * dim)
        out_dim = out_dim or dim
        self.mlp = Mlp(dim, hidden_features=hidden, out_features=out_dim, act_layer=nn.GELU)
        # layer scaling
        if ls_init_value >= 0:
            self.gamma = nn.Parameter(torch.full(size=(1, out_dim, 1, 1), fill_value=1e-6))
        else:
            self.gamma = None

        self.residual = residual
        self.requires_embedding = True

    def forward(self, x, emb):
        shortcut = x
        x = self.conv_dw(x)
        x = x.permute(0, 2, 3, 1).contiguous()           # NCHW -> NHWC
        x = self.norm(x)
        shift, scale = torch.chunk(self.embedding_layer(emb), chunks=2, dim=-1)
        x = x * (1 + scale) + shift                       # AdaLN
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2).contiguous()           # NHWC -> NCHW
        if self.gamma is not None:
            x = x.mul(self.gamma)
        if self.residual:
            x = x + shortcut
        return x


class ConvNeXtAdaLNPatchDown(ConvNeXtBlockAdaLN):
    """A ConvNeXt-AdaLN block followed by a ``down_rate`` patch downsampling."""

    def __init__(self, in_ch, out_ch, down_rate=2, **kwargs):
        super().__init__(in_ch, **kwargs)
        self.downsapmle = patch_downsample(in_ch, out_ch, rate=down_rate)  # (sic) original name

    def forward(self, x, emb):
        x = super().forward(x, emb)
        return self.downsapmle(x)
