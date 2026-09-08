"""Simple strided-conv network (ported from ``nn_models.get_convnet``).

Used as the R-D lower bound's ``log u`` model on image-like data.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from .mlp import get_activation


def get_convnet(
    in_channels: int,
    num_units: Sequence[int],
    kernel_dims: Sequence[int] = (5, 3),
    activation: str = "selu",
    input_hw: Optional[int] = None,
    raw_pixel_input: bool = False,
) -> nn.Module:
    """Strided-conv -> flatten -> MLP -> scalar net (mirrors get_convnet).

    ``num_units``: the first ``len(kernel_dims)`` entries are the conv
    layers' channel counts; the remaining entries are dense-layer widths.
    ``input_hw`` (the spatial size of a square input) is required to compute
    the flatten size for the first dense layer.
    """
    n_conv = len(kernel_dims)
    conv_channels = list(num_units[:n_conv])
    dense_units = list(num_units[n_conv:])

    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.raw_pixel_input = raw_pixel_input
            convs = []
            c_in = in_channels
            hw = input_hw
            for c_out, k in zip(conv_channels, kernel_dims):
                convs.append(nn.Conv2d(c_in, c_out, kernel_size=k, stride=2, padding=k // 2))
                convs.append(get_activation(activation, num_channels=c_out))
                c_in = c_out
                if hw is not None:
                    hw = (hw + 1) // 2
            self.convs = nn.Sequential(*[m for m in convs if m is not None])
            flat_dim = c_in * hw * hw if hw is not None else None
            dense: List[nn.Module] = []
            in_dim = flat_dim
            for u in dense_units:
                assert in_dim is not None, "input_hw must be given to size the first dense layer"
                dense.append(nn.Linear(in_dim, u))
                act = get_activation(activation, num_channels=u)
                if act is not None:
                    dense.append(act)
                in_dim = u
            if in_dim is None:
                raise ValueError("input_hw must be provided so the flatten size is known")
            dense.append(nn.Linear(in_dim, 1))
            self.dense = nn.Sequential(*dense)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.raw_pixel_input:
                x = x / 255.0
            h = self.convs(x)
            h = h.flatten(1)
            return self.dense(h)

    return ConvNet()
