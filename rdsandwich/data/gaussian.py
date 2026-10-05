"""Factorized Gaussian source and its analytical R(D) (reverse water-filling)."""
from __future__ import annotations

import math

import numpy as np
import torch

from .base import Source


class GaussianSource(Source):
    """Factorized (diagonal-covariance) Gaussian source."""

    def __init__(self, loc: np.ndarray, scale: np.ndarray, device=None):
        self.loc = torch.as_tensor(loc, dtype=torch.float32, device=device)
        self.scale = torch.as_tensor(scale, dtype=torch.float32, device=device)

    @classmethod
    def standard(cls, dim: int, device=None):
        return cls(np.zeros(dim, dtype=np.float32), np.ones(dim, dtype=np.float32), device=device)

    @classmethod
    def from_npz(cls, path: str, device=None):
        """Load loc/scale as saved by ``scripts/gen_gaussian_params.py``."""
        z = np.load(path)
        return cls(z["loc"].astype(np.float32), z["scale"].astype(np.float32), device=device)

    def sample(self, batchsize: int) -> torch.Tensor:
        eps = torch.randn(batchsize, *self.loc.shape, device=self.loc.device)
        return self.loc + self.scale * eps


def gen_gaussian_params(dim: int, seed: int = 0) -> dict:
    """Randomly generate loc in [-0.5, 0.5], scale in [0, 2] per dimension.

    Mirrors ``gen_gaussian_params.py``.
    """
    rng = np.random.RandomState(seed)
    loc = rng.uniform(-0.5, 0.5, size=dim).astype(np.float32)
    scale = rng.uniform(0.0, 2.0, size=dim).astype(np.float32)
    return dict(loc=loc, scale=scale)


def gaussian_analytical_rd(scale: np.ndarray, D: float, nats: bool = True) -> float:
    """Reverse water-filling analytical R(D) for a factorized Gaussian (per-sample rate).

    R(D) = sum_i max(0, 0.5 * log(scale_i^2 / theta)), where theta solves
    sum_i min(theta, scale_i^2) = D.
    """
    var = scale.astype(np.float64) ** 2

    def total_distortion(theta):
        return np.minimum(theta, var).sum()

    if D >= var.sum():
        return 0.0
    lo, hi = 0.0, var.max()
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total_distortion(mid) < D:
            lo = mid
        else:
            hi = mid
    theta = 0.5 * (lo + hi)
    rate = 0.5 * np.sum(np.maximum(0.0, np.log(var / theta)))
    if not nats:
        rate = rate / math.log(2.0)
    return float(rate)
