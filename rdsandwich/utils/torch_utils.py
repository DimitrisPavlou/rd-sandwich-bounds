"""Seeding, device selection, and small numeric helpers."""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# softplus^{-1}(1), used to init scale params near 1
SOFTPLUS_INV_1 = float(np.log(np.expm1(1.0)))


def upper_bound(x: torch.Tensor, bound: torch.Tensor) -> torch.Tensor:
    """Same semantics as tfc.math_ops.upper_bound: min(x, bound), gradient passes through."""
    return torch.minimum(x, bound)


def lower_bound(x: torch.Tensor, bound: torch.Tensor) -> torch.Tensor:
    """Same semantics as tfc.math_ops.lower_bound: max(x, bound), gradient passes through."""
    return torch.maximum(x, bound)


def call_fp32(fn, *tensors: torch.Tensor):
    """Call ``fn`` on fp32 copies of ``tensors`` with autocast disabled, so its output stays fp32
    inside a bf16/fp16 autocast region. For layers whose output feeds the rate or the distortion
    directly (latent heads, image heads, the DeepFactorized density), where rounding that output
    to 16 bits would coarsen the model. Outside autocast it changes nothing.

    (``fn(x.float())`` alone is not enough: autocast casts a conv/linear/matmul's inputs back down.)
    """
    with torch.autocast(tensors[0].device.type, enabled=False):
        return fn(*(t.float() for t in tensors))


def ema_update(prev: Optional[float], new: float, beta: float) -> float:
    """Exponential moving average: beta * prev + (1 - beta) * new."""
    if prev is None or beta == 0:
        return new
    return beta * prev + (1 - beta) * new
