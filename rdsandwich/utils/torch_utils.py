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


def ema_update(prev: Optional[float], new: float, beta: float) -> float:
    """Exponential moving average: beta * prev + (1 - beta) * new."""
    if prev is None or beta == 0:
        return new
    return beta * prev + (1 - beta) * new
