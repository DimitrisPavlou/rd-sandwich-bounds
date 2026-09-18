"""Channel-wise autoregressive transform (ported from ``resnet_vae.py``).

Given a latent tensor split along the channel axis into ``num_slices`` slices
``[x_0, x_1, ..., x_{n-1}]``, produces ``[0, f_1(x_0), f_2(x_0, x_1), ...]`` --
each output slice depends only on the *preceding* slices. Used as the
shift/scale functions of a channel-wise IAF-style autoregressive prior in the
ResNet-VAE (a lightweight analogue of Minnen et al. 2020's channel-conditioning).

Each per-slice transform is a 3-conv stack whose channel widths interpolate
between the (larger) conditioning input and the (smaller) slice output, exactly
as in the original TensorFlow implementation.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


def _conv_same(in_ch: int, out_ch: int, k: int, activation: Optional[str]) -> nn.Module:
    conv = nn.Conv2d(in_ch, out_ch, k, stride=1, padding=k // 2)
    if activation == "leaky_relu":
        return nn.Sequential(conv, nn.LeakyReLU(0.2))
    if activation is None:
        return conv
    raise ValueError(f"Unsupported activation {activation!r}")


class ChannelwiseARTransform(nn.Module):
    def __init__(
        self,
        input_num_channels: int,
        num_slices: int,
        max_support_ratio: Optional[float] = 0.5,
        cond_first_slices: bool = True,
    ):
        super().__init__()
        self.cond_first_slices = cond_first_slices

        if num_slices > input_num_channels:
            num_slices = input_num_channels
        slice_depth = input_num_channels // num_slices
        if slice_depth * num_slices != input_num_channels:
            raise ValueError(
                f"Slices do not evenly divide latent depth ({input_num_channels} / {num_slices})"
            )
        self.input_num_channels = input_num_channels
        self.num_slices = num_slices
        self.slice_depth = slice_depth

        if max_support_ratio is None:
            max_support_slices = num_slices - 1
        else:
            max_support_slices = round(num_slices * max_support_ratio)
        self.max_support_slices = max_support_slices

        def interpolate(alpha: float, x: int = max_support_slices, y: int = 1) -> int:
            return round(alpha * x + (1 - alpha) * y)

        # A dummy zeroth transform (slice 0 is always predicted as zeros).
        self.transforms = nn.ModuleList([nn.Identity()])
        for i in range(1, num_slices):
            in_ch = min(max_support_slices, i) * slice_depth
            self.transforms.append(
                nn.Sequential(
                    _conv_same(in_ch, interpolate(0.6) * slice_depth, 3, "leaky_relu"),
                    _conv_same(interpolate(0.6) * slice_depth, interpolate(0.3) * slice_depth, 3, "leaky_relu"),
                    _conv_same(interpolate(0.3) * slice_depth, 1 * slice_depth, 5, None),
                )
            )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        input_slices = torch.split(tensor, self.slice_depth, dim=1)
        output_slices: List[torch.Tensor] = []
        for i in range(self.num_slices):
            if i == 0:
                output_slices.append(torch.zeros_like(input_slices[0]))
                continue
            if self.cond_first_slices:
                support = input_slices[: min(self.max_support_slices, i)]
            else:
                support = input_slices[max(0, i - self.max_support_slices): i]
            output_slices.append(self.transforms[i](torch.cat(support, dim=1)))
        return torch.cat(output_slices, dim=1)
