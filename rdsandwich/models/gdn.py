"""Generalized Divisive Normalization (Ballé et al.), a PyTorch port of ``tfc.GDN``.

GDN is the standard nonlinearity in learned image-compression analysis/synthesis
transforms::

    y_i = x_i / sqrt(beta_i + sum_j gamma_ij * x_j^2)        (forward / GDN)
    y_i = x_i * sqrt(beta_i + sum_j gamma_ij * x_j^2)        (inverse / IGDN)

``tfc.GDN`` keeps ``beta`` and ``gamma`` non-negative via a
``NonNegativeParameterizer`` (a shifted square-root reparameterization with a
lower bound / "pedestal"). We reproduce that here rather than the plain
``softplus`` used by the lightweight MLP-head port, so the divisor cannot
collapse to zero and initialization matches tfc's defaults (``gamma_init=0.1``,
``beta_min=1e-6``). The normalization mixes channels at each spatial location,
implemented as a 1x1 convolution of ``x**2``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NonNegativeParameterizer(nn.Module):
    """Reparameterize an unconstrained tensor to a non-negative one, as in tfc.

    Stores ``theta`` and returns ``value = max(theta, theta_min)**2 - pedestal``,
    where ``pedestal = offset**2`` and ``theta_min = sqrt(minimum + pedestal)``.
    This keeps ``value >= minimum`` with a well-behaved gradient near zero.
    """

    def __init__(self, initial_value: torch.Tensor, minimum: float = 0.0, offset: float = 2 ** -18):
        super().__init__()
        self.minimum = float(minimum)
        self.offset = float(offset)
        pedestal = self.offset ** 2
        self.pedestal = pedestal
        self.theta_min = (self.minimum + pedestal) ** 0.5
        # Invert: theta = sqrt(max(value + pedestal, 0))
        theta = torch.sqrt(torch.clamp(initial_value + pedestal, min=0.0))
        self.theta = nn.Parameter(theta)

    def forward(self) -> torch.Tensor:
        theta = torch.clamp(self.theta, min=self.theta_min)
        return theta ** 2 - self.pedestal


class GDN(nn.Module):
    def __init__(
        self,
        num_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
        eps: float = 1e-10,
    ):
        super().__init__()
        self.inverse = inverse
        self.num_channels = num_channels
        self.eps = eps
        beta0 = torch.ones(num_channels)
        gamma0 = gamma_init * torch.eye(num_channels)
        self._beta = NonNegativeParameterizer(beta0, minimum=beta_min)
        self._gamma = NonNegativeParameterizer(gamma0, minimum=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = self._beta()
        gamma = self._gamma()
        if x.dim() == 2:  # [B, C]
            norm_sq = F.linear(x * x, gamma) + beta
        elif x.dim() == 4:  # [B, C, H, W]
            gamma = gamma.view(self.num_channels, self.num_channels, 1, 1)
            norm_sq = F.conv2d(x * x, gamma, bias=beta)
        else:
            raise ValueError(f"GDN expects 2D or 4D input, got {x.dim()}D")
        norm = torch.sqrt(torch.clamp(norm_sq, min=self.eps))
        return x * norm if self.inverse else x / norm
