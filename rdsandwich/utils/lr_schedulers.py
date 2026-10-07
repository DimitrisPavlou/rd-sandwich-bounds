"""Epoch-wise learning-rate schedules, all stepped once per epoch by ``BaseTrainer``.

* ``make_lr_scheduler``: piecewise-constant decay (x0.2 at 50/75/87.5% of the epochs),
  used for the vector-data experiments (``--lr_schedule piecewise``).
* ``make_const_cos_scheduler``: constant for the first half of the epochs, then a cosine
  down to ``final`` (Duan et al.'s recipe; ``--lr_schedule const-cos``).
* ``WarmupReduceLROnPlateau``: ``ReduceLROnPlateau`` that ignores the first ``warmup``
  epochs (``--lr_schedule plateau``); stepped with the monitored metric.

The first two are ``LambdaLR`` schedulers built from the pure per-epoch factor
functions next to them.
"""
from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


def lr_lambda_schedule(epoch: int, epochs: int, decay_factor: float = 0.2) -> float:
    """Piecewise-constant decay matching rdub_mlp.get_lr_scheduler."""
    if epoch < 0.5 * epochs:
        return 1.0
    if epoch < 0.75 * epochs:
        return decay_factor
    if epoch < 0.875 * epochs:
        return decay_factor ** 2
    return decay_factor ** 3


def make_lr_scheduler(optimizer: torch.optim.Optimizer, epochs: int):
    """LambdaLR implementing ``lr_lambda_schedule`` over ``epochs`` epochs."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: lr_lambda_schedule(e, epochs)
    )


def const_cos_factor(epoch: int, epochs: int, final: float = 0.01) -> float:
    """LR factor of Duan et al.'s ``const-0.5-cos`` schedule, per epoch: 1 for the first
    half of the epochs, then a cosine from 1 down to ``final`` over the second half
    (``final`` in the last epoch)."""
    boundary = epochs // 2
    if epoch < boundary:
        return 1.0
    span = max(epochs - boundary - 1, 1)
    t = min(epoch - boundary, span)
    return final + 0.5 * (1.0 - final) * (1.0 + math.cos(math.pi * t / span))


def make_const_cos_scheduler(optimizer: torch.optim.Optimizer, epochs: int, final: float = 0.01):
    """LambdaLR implementing ``const_cos_factor`` over ``epochs`` epochs (stepped once per epoch)."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: const_cos_factor(e, epochs, final)
    )


class WarmupReduceLROnPlateau:
    """``ReduceLROnPlateau`` that ignores the first ``warmup`` epochs.

    Mirrors the original repo's ``MyReduceLROnPlateauCallback``: the learning
    rate is held constant during warmup, then reduced by ``factor`` after
    ``patience`` epochs without ``min_delta`` improvement, down to ``min_lr``.
    """

    def __init__(self, optimizer, *, factor=0.5, patience=10, warmup=100,
                 min_lr=1e-6, min_delta=1e-4, mode="min", verbose=False):
        self.optimizer = optimizer
        self.warmup = warmup
        self._plateau = ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience,
            min_lr=min_lr, threshold=min_delta, threshold_mode="abs",
        )
        self.verbose = verbose
        self._epoch = 0

    def step(self, metric: float) -> None:
        if self._epoch >= self.warmup:
            self._plateau.step(metric)
        self._epoch += 1

    def state_dict(self):
        return {"epoch": self._epoch, "plateau": self._plateau.state_dict()}

    def load_state_dict(self, state):
        self._epoch = state.get("epoch", 0)
        if "plateau" in state:
            self._plateau.load_state_dict(state["plateau"])
