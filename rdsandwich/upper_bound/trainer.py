"""Upper-bound trainer: a thin ``BaseTrainer`` subclass for the beta-VAE.

``train_step`` is just the model's Lagrangian loss; the rest of the loop
(batching, optimizer/scheduler step, logging, checkpointing) comes from
``BaseTrainer``. Also provides the piecewise LR schedule and the (D, R)
evaluator with a 95% CI (Appendix A.6).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch

from ..utils import BaseTrainer
from .model import RDUBModel


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


class UpperBoundTrainer(BaseTrainer):
    """Trains an :class:`RDUBModel` to minimize rate + lambda * distortion."""

    def train_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss, rate, mse = self.model.get_losses(x)
        return loss, {"rate": rate.item(), "mse": mse.item()}

    def checkpoint_extra(self) -> Dict[str, Any]:
        cfg = getattr(self.model, "cfg", None)
        return {"cfg": vars(cfg) if cfg is not None and hasattr(cfg, "__dict__") else None}


@torch.no_grad()
def evaluate_rdub(model: RDUBModel, source, batchsize: int, num_batches: int, device=None):
    """Estimate (D, R) with sample mean +/- 95% CI, as in Appendix A.6 ('Upper Bound Estimator')."""
    import numpy as np

    device = device or next(model.parameters()).device
    model.eval()
    rates, mses = [], []
    for _ in range(num_batches):
        x = source.sample(batchsize).to(device)
        _, rate, mse = model.get_losses(x)
        rates.append(rate.item())
        mses.append(mse.item())

    def mean_ci(vals):
        vals = np.array(vals)
        m, s = vals.mean(), vals.std(ddof=1)
        ci = 1.96 * s / math.sqrt(len(vals))
        return m, s, (m - ci, m + ci)

    r_mean, r_std, r_ci = mean_ci(rates)
    d_mean, d_std, d_ci = mean_ci(mses)
    return dict(rate_mean=r_mean, rate_std=r_std, rate_ci=r_ci,
                mse_mean=d_mean, mse_std=d_std, mse_ci=d_ci)
