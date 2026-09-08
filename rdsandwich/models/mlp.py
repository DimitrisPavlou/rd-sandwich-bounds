"""MLP / activation / GDN building blocks (ported from ``nn_models.py``)."""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class GDN(nn.Module):
    """Generalized Divisive Normalization, y_i = x_i / sqrt(beta_i + sum_j gamma_ij x_j^2).

    A lightweight PyTorch port of tfc.GDN (Ballé et al.) sufficient for
    small MLP/CNN heads; not fused/optimized like tensorflow-compression's
    version. Set ``inverse=True`` to get IGDN, y_i = x_i * sqrt(beta_i + ...).
    """

    def __init__(self, num_channels: int, inverse: bool = False, eps: float = 1e-6):
        super().__init__()
        self.inverse = inverse
        self.eps = eps
        self.beta = nn.Parameter(torch.ones(num_channels))
        self.gamma = nn.Parameter(0.1 * torch.eye(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., C] for MLP-style usage, or [B, C, H, W] for conv usage.
        if x.dim() == 2:  # [B, C]
            beta = F.softplus(self.beta) + self.eps
            gamma = F.softplus(self.gamma)
            norm = torch.sqrt(F.linear(x * x, gamma) + beta)
        else:  # [B, C, H, W]
            beta = (F.softplus(self.beta) + self.eps).view(1, -1, 1, 1)
            gamma = F.softplus(self.gamma).view(self.gamma.shape[0], self.gamma.shape[1], 1, 1)
            norm = torch.sqrt(F.conv2d(x * x, gamma) + beta)
        return x * norm if self.inverse else x / norm


def get_activation(name: Optional[str], num_channels: Optional[int] = None) -> Optional[nn.Module]:
    """Look up an activation module by name; mirrors nn_models.get_activation.

    ``num_channels`` is only needed for 'gdn'/'igdn', which are parametric.
    """
    if not name or str(name).lower() == "none":
        return None
    name = name.lower()
    if name == "gdn":
        assert num_channels is not None, "GDN needs num_channels"
        return GDN(num_channels, inverse=False)
    if name == "igdn":
        assert num_channels is not None, "GDN needs num_channels"
        return GDN(num_channels, inverse=True)
    table = {
        "relu": nn.ReLU,
        "leaky_relu": lambda: nn.LeakyReLU(0.2),
        "softplus": nn.Softplus,
        "selu": nn.SELU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
    }
    if name not in table:
        raise ValueError(f"Unknown activation: {name}")
    return table[name]()


def make_mlp(
    input_dim: int,
    units: Sequence[int],
    activation: Optional[str] = "relu",
    no_last_activation: bool = True,
) -> nn.Sequential:
    """Build a fully-connected feedforward net, mirroring nn_models.make_mlp.

    ``units`` gives the size of each layer's *output*, including the final
    (output) layer, e.g. ``units=[128, 128, 2*latent_dim]``.
    """
    layers: List[nn.Module] = []
    in_dim = input_dim
    for i, out_dim in enumerate(units):
        layers.append(nn.Linear(in_dim, out_dim))
        is_last = i == len(units) - 1
        if not (is_last and no_last_activation):
            act = get_activation(activation, num_channels=out_dim)
            if act is not None:
                layers.append(act)
        in_dim = out_dim
    return nn.Sequential(*layers)
