"""Gaussian-latent helpers shared by the image beta-VAEs (ResNet-VAE, MS2020-VAE)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from rdsandwich.utils.torch_utils import SOFTPLUS_INV_1

LN2 = math.log(2.0)
LOG2PI = math.log(2.0 * math.pi)


def softplus_scale(raw: torch.Tensor, scale_min: float = 0.0) -> torch.Tensor:
    """Map a raw feature to a positive scale near 1 at init (softplus(x + softplus^-1(1))).

    ``scale_min`` floors the scale: without it a scale can collapse towards 0, where
    the Gaussian log-density's gradient (~ 1/scale^3) overflows to inf in fp32.
    """
    return F.softplus(raw + SOFTPLUS_INV_1) + scale_min


def normal_log_prob(x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return -0.5 * (((x - loc) / scale) ** 2 + LOG2PI) - torch.log(scale)


def gaussian_kl(q_loc, q_scale, p_loc, p_scale):
    """KL( N(q_loc, q_scale) || N(p_loc, p_scale) ), elementwise."""
    return (
        torch.log(p_scale) - torch.log(q_scale)
        + (q_scale ** 2 + (q_loc - p_loc) ** 2) / (2.0 * p_scale ** 2)
        - 0.5
    )
