"""Lower-bound evaluator: the final R_L(D) estimate (``est_R_`` in the original code)."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn

from rdsandwich.lower_bound.config import RDLBTrainConfig
from rdsandwich.lower_bound.algorithm import run_optimize_y


@torch.no_grad()
def estimate_R_lower_bound(log_u_model: nn.Module, source, lamb: float, cfg: RDLBTrainConfig, device=None):
    """Final evaluation (``est_R_``). Uses 2M samples of C_k — the first M set the
    log expansion point alpha, the second M (with fresh log_u samples) estimate
    the objective xi (Eq. 15). Should be run with ``cfg.y_init == 'exhaustive'``.

    ``source`` is any object exposing ``.sample(batchsize) -> Tensor``.
    """
    device = device or next(log_u_model.parameters()).device
    M = cfg.num_Ck_samples
    assert M >= 1
    total = 2 * M
    log_Ck_samples, E_log_us = [], []
    t_start = time.perf_counter()
    for i in range(total):
        x = source.sample(cfg.batchsize).to(device)
        res = run_optimize_y(cfg, log_u_model, x, lamb)
        log_Ck_samples.append(res["opt_log_supobj"])
        E_log_us.append(log_u_model(x).mean().item())
        # Post-processing (exhaustive-optimizer eval) is the slow part: 2M full
        # hill-climbs. Report progress + ETA so the wait is legible.
        elapsed = time.perf_counter() - t_start
        eta = elapsed / (i + 1) * (total - i - 1)
        print(f"  [eval C_k] sample {i + 1}/{total}  elapsed={elapsed:.1f}s  eta={eta:.1f}s", flush=True)

    E_log_us = np.array(E_log_us)
    log_Ck_samples = np.array(log_Ck_samples)
    log_alpha = torch.logsumexp(torch.tensor(log_Ck_samples[:M]), dim=0).item() - math.log(M)
    xi_samples = -E_log_us[M:] - np.exp(log_Ck_samples[M:] - log_alpha) - log_alpha + 1
    R_ = float(np.mean(xi_samples))
    return dict(R_=R_, xi_samples=xi_samples, log_alpha=log_alpha,
                E_log_us=E_log_us, log_Ck_samples=log_Ck_samples)
