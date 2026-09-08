"""The lower bound's ``log u`` model factory.

Unlike the upper bound, the lower bound has no bespoke model class computing the
objective — the objective lives in ``algorithm.py`` / the trainer. The "model"
is just a network mapping x -> a scalar ``log u(x)``, built from the generic
blocks in ``rdsandwich.models``.
"""
from __future__ import annotations

from typing import Sequence

import torch.nn as nn

from ..models import get_convnet, make_mlp


def build_log_u_model(
    model: str,
    data_dim: int,
    units: Sequence[int],
    activation: str = "selu",
    in_channels: int = 3,
    kernel_dims: Sequence[int] = (),
) -> nn.Module:
    """Build the ``log u`` network. ``model`` is 'mlp' (vector data) or 'cnn'
    (image data); ``units`` are the hidden widths, with a final scalar output."""
    if model == "mlp":
        return make_mlp(data_dim, list(units) + [1], activation=activation)
    if model == "cnn":
        return get_convnet(
            in_channels=in_channels, num_units=units, kernel_dims=kernel_dims,
            activation="selu", input_hw=data_dim,
        )
    raise NotImplementedError(model)
