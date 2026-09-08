"""
A minimal Masked Autoregressive Flow (MAF), reimplementing the role that
``tfp.bijectors.MaskedAutoregressiveFlow`` + ``tfb.AutoregressiveNetwork``
played in the original ``rdub_mlp.py`` (used as a flexible prior ``Q_Z`` on
the latent space).

Only the density-estimation direction (``log_prob``) is implemented
efficiently (single forward pass); this is all the original code needs,
since the prior is only ever evaluated via ``prior.log_prob(y_tilde)`` on
samples drawn from the encoder. ``sample()`` is provided too, but is
sequential (O(D) network evaluations), as is standard for MAF.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask: torch.Tensor):
        super().__init__(in_features, out_features)
        self.register_buffer("mask", mask)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    """Masked Autoencoder for Distribution Estimation.

    Produces two outputs per input dimension (shift, log_scale), each
    depending only on the *preceding* dimensions in a fixed order — i.e. the
    autoregressive property required for a normalizing flow.
    """

    def __init__(self, dim: int, hidden_units: Sequence[int] = (20, 20)):
        super().__init__()
        self.dim = dim
        degrees = self._make_degrees(dim, hidden_units)
        masks = self._make_masks(degrees)

        layers: List[nn.Module] = []
        sizes = [dim] + list(hidden_units)
        for i in range(len(hidden_units)):
            layers.append(MaskedLinear(sizes[i], sizes[i + 1], masks[i]))
            layers.append(nn.ReLU())
        # output layer: 2*dim outputs (shift, log_scale), masked by the last mask repeated twice
        out_mask = masks[-1]
        self.hidden = nn.Sequential(*layers)
        self.shift_out = MaskedLinear(sizes[-1], dim, out_mask)
        self.log_scale_out = MaskedLinear(sizes[-1], dim, out_mask)

    @staticmethod
    def _make_degrees(dim, hidden_units):
        # input degrees: 0..dim-1 (natural order, no randomization for reproducibility)
        degrees = [np.arange(dim)]
        for h in hidden_units:
            # hidden units get degrees in [min_prev_degree, dim-2], cycling
            min_prev = degrees[-1].min()
            degrees.append(np.arange(h) % max(dim - 1, 1) + min(min_prev, dim - 1))
        return degrees

    @staticmethod
    def _make_masks(degrees):
        masks = []
        for d_in, d_out in zip(degrees[:-1], degrees[1:]):
            # mask[i, j] = 1 if d_out[i] >= d_in[j] (strict '>' would give a fully-autoregressive
            # dependency without self-connections in the first layer; using >= for hidden layers
            # is the standard MADE recipe)
            mask = (d_out[:, None] >= d_in[None, :]).astype(np.float32)
            masks.append(torch.from_numpy(mask))
        # output mask uses strict '>' against the last hidden layer's degrees so that output i
        # depends only on inputs with degree < i (autoregressive, excludes self). Shape must be
        # (dim, hidden_size) to match nn.Linear(hidden_size -> dim)'s weight shape — NOT
        # transposed, unlike the (out, in) mask convention used for the hidden-to-hidden masks
        # above (which happens to already come out as (out, in) from the >= comparison order).
        out_mask = (degrees[0][:, None] > degrees[-1][None, :]).astype(np.float32)
        masks.append(torch.from_numpy(out_mask))
        return masks

    def forward(self, x: torch.Tensor):
        h = self.hidden(x)
        shift = self.shift_out(h)
        log_scale = self.log_scale_out(h)
        log_scale = torch.tanh(log_scale)  # stabilize, as in tfb.AutoregressiveNetwork practice
        return shift, log_scale


class MAFLayer(nn.Module):
    """One MAF transform x -> z with a fixed dimension permutation, plus its log-det-Jacobian."""

    def __init__(self, dim: int, hidden_units: Sequence[int] = (20, 20), permutation: np.ndarray = None):
        super().__init__()
        self.made = MADE(dim, hidden_units)
        if permutation is None:
            permutation = np.arange(dim)
        self.register_buffer("permutation", torch.from_numpy(permutation).long())

    def forward_and_log_det(self, x: torch.Tensor):
        """Density-estimation direction: x (data space) -> z (base space)."""
        x_perm = x[..., self.permutation]
        shift, log_scale = self.made(x_perm)
        z = (x_perm - shift) * torch.exp(-log_scale)
        log_det = -log_scale.sum(-1)
        return z, log_det

    @torch.no_grad()
    def inverse(self, z: torch.Tensor):
        """Sampling direction: z (base space) -> x (data space). Sequential over dim."""
        x_perm = torch.zeros_like(z)
        dim = z.shape[-1]
        for i in range(dim):
            shift, log_scale = self.made(x_perm)
            x_perm[..., i] = z[..., i] * torch.exp(log_scale[..., i]) + shift[..., i]
        inv_perm = torch.argsort(self.permutation)
        return x_perm[..., inv_perm]


class MAF(nn.Module):
    """A stack of MAF layers with circular-shift permutations between them,
    on top of a standard Normal base distribution — the torch analogue of
    ``af_transform`` + ``tfd.MultivariateNormalDiag`` in the original code.
    """

    def __init__(self, dim: int, n_stacks: int = 3, hidden_units: Sequence[int] = (20, 20)):
        super().__init__()
        self.dim = dim
        layers = []
        for i in range(n_stacks):
            perm = np.roll(np.arange(dim), i)
            layers.append(MAFLayer(dim, hidden_units, permutation=perm))
        self.layers = nn.ModuleList(layers)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        total_log_det = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        for layer in self.layers:
            z, log_det = layer.forward_and_log_det(z)
            total_log_det = total_log_det + log_det
        base_log_prob = (-0.5 * (z ** 2 + np.log(2 * np.pi))).sum(-1)
        return base_log_prob + total_log_det

    @torch.no_grad()
    def sample(self, n: int, device=None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        z = torch.randn(n, self.dim, device=device)
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z
