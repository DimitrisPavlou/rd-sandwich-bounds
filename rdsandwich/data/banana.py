"""2D "banana" source (Ballé et al., 2021) and its random n-d embeddings."""
from __future__ import annotations

import math

import torch

from .base import Source


class BananaSource(Source):
    """2D banana-shaped distribution, ported from ``ntc_sources.get_banana``.

    Reproduces the exact sequence of transforms in the original TFP
    definition (a RealNVP-style quadratic coupling, a rotation, and a
    shift, wrapped in ``Invert(...)`` so that *sampling* applies the
    inverse of each transform to a base ``Normal([0,0],[3,.5])`` draw).
    """

    def __init__(self, device=None):
        self.device = device
        phi = math.radians(240.0)
        self._rot = torch.tensor(
            [[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]],
            dtype=torch.float32,
            device=device,
        )
        self._shift = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
        self._base_scale = torch.tensor([3.0, 0.5], dtype=torch.float32, device=device)

    def sample(self, batchsize: int) -> torch.Tensor:
        base = torch.randn(batchsize, 2, device=self.device) * self._base_scale
        x = base - self._shift  # Shift.inverse
        x = x @ self._rot  # Rotation.inverse (rotation matrix is orthogonal: R^{-1} = R^T,
        #                     and for row-batched x, x @ R == (R^T @ x^T)^T)
        x1, x2 = x[:, 0], x[:, 1]
        x2 = x2 - 0.1 * x1 ** 2  # RealNVP.inverse (coupling only shifts x2 given x1)
        return torch.stack([x1, x2], dim=-1)


class NdBananaEmbedder(Source):
    """Randomly embeds the 2D banana source in R^n via a fixed random linear
    layer + softplus, mirroring ``ntc_sources.get_nd_banana``.
    """

    def __init__(self, n: int, seed: int = 0, device=None):
        self.base = BananaSource(device=device)
        g = torch.Generator(device="cpu").manual_seed(seed)
        weight = torch.empty(n, 2)
        torch.nn.init.xavier_normal_(weight, generator=g)
        bias = torch.zeros(n)
        self.weight = weight.to(device)
        self.bias = bias.to(device)

    def sample(self, batchsize: int) -> torch.Tensor:
        x2 = self.base.sample(batchsize)
        return torch.nn.functional.softplus(x2 @ self.weight.T + self.bias)
