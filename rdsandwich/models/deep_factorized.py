"""Deep factorized density model (Ballé et al., 2018, Appendix 6.1).

Each channel gets its own univariate density ``p(x)``, defined implicitly through
a learned, monotonically-increasing cumulative ``c: R -> [0, 1]`` built as a
composition of small affine maps with ``tanh``-augmented nonlinearities::

    c = sigmoid(f_K o ... o f_1),   f_k(h) = M_k h + b_k (+ a_k * tanh(M_k h + b_k))

with ``M_k = softplus(H_k) >= 0`` guaranteeing monotonicity, and
``a_k = tanh(F_k) in (-1, 1)`` keeping each layer's derivative positive. The
density is the derivative ``p(x) = dc/dx``, which we evaluate **analytically**
(propagating the scalar derivative through the same layers) so that:

  * it works under ``torch.no_grad()`` (needed at evaluation time), and
  * no second-order autograd is required during training.

This is the ``tfc.DeepFactorized`` distribution used as the hyperprior in
learned image compression. The sandwich-bounds paper (Sec. 6.4) uses exactly
this density but **without** convolving it with a uniform ``U(-1/2, 1/2)`` (i.e.
the continuous ``DeepFactorized``, not the ``NoisyDeepFactorized`` used to
simulate quantization). For completeness we also expose ``log_prob_noisy`` for
the uniform-convolved variant, ``log(c(x + 1/2) - c(x - 1/2))``.

Reference:
  Ballé, Minnen, Singh, Hwang, Johnston. "Variational Image Compression with a
  Scale Hyperprior." ICLR 2018 (Appendix 6.1), and its implementation in
  tensorflow-compression.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepFactorized(nn.Module):
    """A per-channel deep factorized density over ``channels`` independent channels.

    ``log_prob(x)`` accepts ``x`` of shape ``[..., channels]`` (channel last) or,
    via :meth:`log_prob_nchw`, an image tensor ``[B, channels, H, W]``.
    """

    def __init__(
        self,
        channels: int,
        filters: Sequence[int] = (3, 3, 3),
        init_scale: float = 10.0,
    ):
        super().__init__()
        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        # Full per-layer dimensions: 1 -> filters -> 1  (dims[i] -> dims[i+1]).
        dims = (1,) + self.filters + (1,)
        self.dims = dims
        self.num_layers = len(self.filters) + 1  # K + 1 affine layers
        scale = init_scale ** (1.0 / self.num_layers)

        self._H = nn.ParameterList()   # pre-softplus matrices, [C, d_out, d_in]
        self._b = nn.ParameterList()   # biases, [C, d_out, 1]
        self._a = nn.ParameterList()   # pre-tanh factors, [C, d_out, 1] (all but last layer)
        for i in range(self.num_layers):
            d_in, d_out = dims[i], dims[i + 1]
            # softplus(H_init) = 1 / (scale * d_out)  =>  H_init = log(expm1(...))
            h_init = math.log(math.expm1(1.0 / (scale * d_out)))
            self._H.append(nn.Parameter(torch.full((self.channels, d_out, d_in), h_init)))
            self._b.append(nn.Parameter(torch.empty(self.channels, d_out, 1).uniform_(-0.5, 0.5)))
            if i < self.num_layers - 1:
                self._a.append(nn.Parameter(torch.zeros(self.channels, d_out, 1)))

    # ------------------------------------------------------------------ #
    def _cumulative_and_derivative(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logit, dlogit/dx)`` for input ``x`` of shape [C, 1, M].

        ``logit`` is the pre-sigmoid cumulative; the density is
        ``sigmoid'(logit) * dlogit/dx``. Both are computed analytically.
        """
        h = x                                   # [C, 1, M], value along middle dim
        dh = torch.ones_like(x)                 # dh/dx, [C, 1, M]
        for i in range(self.num_layers):
            M = F.softplus(self._H[i])          # [C, d_out, d_in], >= 0
            pre = torch.matmul(M, h) + self._b[i]        # [C, d_out, M]
            dpre = torch.matmul(M, dh)                   # [C, d_out, M]
            if i < self.num_layers - 1:
                a = torch.tanh(self._a[i])               # [C, d_out, 1] in (-1, 1)
                tanh_pre = torch.tanh(pre)
                h = pre + a * tanh_pre
                # d/dx of (pre + a*tanh(pre)) = (1 + a*(1 - tanh(pre)^2)) * dpre
                dh = (1.0 + a * (1.0 - tanh_pre ** 2)) * dpre   # factor in (0, 2) > 0
            else:
                h = pre                                  # [C, 1, M] (d_out == 1)
                dh = dpre
        return h, dh

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log-density with ``x`` shaped ``[..., channels]`` (channel last).

        Returns a tensor of shape ``x.shape`` (per-element log-densities); sum
        over the channel axis for a joint log-density under the factorized model.
        """
        if x.shape[-1] != self.channels:
            raise ValueError(
                f"Last dim of x ({x.shape[-1]}) must equal channels ({self.channels})."
            )
        lead = x.shape[:-1]
        M = int(np.prod(lead)) if lead else 1
        # [..., C] -> [C, 1, M]
        xc = x.reshape(M, self.channels).permute(1, 0).unsqueeze(1).contiguous()
        logit, dlogit = self._cumulative_and_derivative(xc)
        # log p = log sigmoid'(logit) + log|dlogit/dx|
        #       = logsigmoid(logit) + logsigmoid(-logit) + log(dlogit)
        log_p = (
            F.logsigmoid(logit)
            + F.logsigmoid(-logit)
            + torch.log(torch.clamp(dlogit, min=1e-12))
        )                                        # [C, 1, M]
        log_p = log_p.squeeze(1).permute(1, 0).reshape(*lead, self.channels)
        return log_p

    def log_prob_nchw(self, x: torch.Tensor) -> torch.Tensor:
        """Joint log-density for an image tensor ``[B, channels, H, W]``.

        Returns ``[B]`` (summed over channels and spatial positions), matching
        the per-sample "rate" contribution of a fully-convolutional prior.
        """
        if x.dim() != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"Expected [B, {self.channels}, H, W], got {tuple(x.shape)}."
            )
        xl = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        log_p = self.log_prob(xl)                # [B, H, W, C]
        return log_p.flatten(1).sum(-1)          # [B]

    def log_prob_noisy(self, y: torch.Tensor) -> torch.Tensor:
        """Uniform-convolved (quantization) variant: ``log(c(y+.5) - c(y-.5))``.

        Not used by the Sec. 6.4 upper bound (which uses the continuous density
        above); provided for parity with the operational-compression setting.
        """
        if y.shape[-1] != self.channels:
            raise ValueError(
                f"Last dim of y ({y.shape[-1]}) must equal channels ({self.channels})."
            )
        lead = y.shape[:-1]
        M = int(np.prod(lead)) if lead else 1
        yc = y.reshape(M, self.channels).permute(1, 0).unsqueeze(1).contiguous()
        upper, _ = self._cumulative_and_derivative(yc + 0.5)
        lower, _ = self._cumulative_and_derivative(yc - 0.5)
        prob = torch.clamp(torch.abs(torch.sigmoid(upper) - torch.sigmoid(lower)), min=1e-12)
        log_p = torch.log(prob)
        log_p = log_p.squeeze(1).permute(1, 0).reshape(*lead, self.channels)
        return log_p
