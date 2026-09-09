"""Lower-bound trainer (Algorithm 1) and the final R_L(D) evaluator.

Unlike the upper bound, the LB loop does not fit ``BaseTrainer``'s
``train_step`` shape: it is step-based, logs every step, checkpoints at
intervals, draws M batches per step, runs the inner ``optimize_y`` hill-climb,
maintains an alpha EMA (with a warm-up step), and differentiates through the
inner maximization via an envelope theorem. So ``LowerBoundTrainer`` reuses
``BaseTrainer``'s plumbing (model/optimizer/loader state, ``_next_batch``,
logging/checkpoint helpers) but **overrides ``train()``** with the Algorithm-1
loop; ``train_step`` is intentionally unused.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..utils import BaseTrainer, ema_update, lower_bound as lb_clip, save_checkpoint, upper_bound as ub_clip
from .algorithm import compute_Ck_obj, optimize_y, optimize_y_vectorized
from .config import RDLBTrainConfig


def _run_optimize_y(cfg: RDLBTrainConfig, log_u_fun, x, lamb):
    """Dispatch to the sequential or vectorized inner optimizer per ``cfg``.

    Vectorized is the default (much faster on GPU); ``cfg.y_sequential=True``
    falls back to the per-candidate loop, and ``cfg.cand_chunk`` bounds the
    vectorized path's memory for image data (ignored when sequential).
    """
    common = dict(
        num_steps=cfg.y_steps, lr=cfg.y_lr, tol=cfg.y_tol,
        init=cfg.y_init, quick_topn=cfg.y_quick_topn, chunksize=cfg.chunksize,
        verbose=False,
    )
    if cfg.y_sequential:
        return optimize_y(log_u_fun, x, lamb, **common)
    return optimize_y_vectorized(log_u_fun, x, lamb, cand_chunk=cfg.cand_chunk, **common)


class LowerBoundTrainer(BaseTrainer):
    """Train a ``log u`` model to maximize the R-D lower bound objective (Eq. 8)."""

    def __init__(
        self,
        model: nn.Module,
        loader,
        *,
        optimizer: torch.optim.Optimizer,
        cfg: RDLBTrainConfig,
        device=None,
        logger=None,
        ckpt_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        # epochs/steps_per_epoch are unused (train() is overridden) but kept so
        # BaseTrainer's constructor is satisfied and its plumbing is reused.
        super().__init__(
            model, loader, optimizer=optimizer, epochs=1, steps_per_epoch=cfg.last_step,
            device=device, logger=logger, ckpt_path=None, verbose=verbose,
        )
        self.cfg = cfg
        self.ckpt_dir = ckpt_dir

    def train(self):
        cfg = self.cfg
        M = cfg.num_Ck_samples
        lamb = cfg.lamb
        device = self.device
        model = self.model
        optimizer = self.optimizer
        logger = self.logger
        log_alpha = None
        prev_log_avg_Ck = None

        for step in range(-1, cfg.last_step):
            x_batches, opt_ys, log_Ck_samples = [], [], []
            for _ in range(M):
                x = self._next_batch()  # [k, *dims] on device
                x_batches.append(x)
                res = _run_optimize_y(cfg, model, x, lamb)
                opt_ys.append(res["opt_y"])
                log_Ck_samples.append(res["opt_log_supobj"])

            if step == -1:
                log_avg_Ck = torch.logsumexp(torch.tensor(log_Ck_samples), dim=0).item() - math.log(M)
                log_alpha = prev_log_avg_Ck = log_avg_Ck
                continue

            if abs(prev_log_avg_Ck - log_alpha) <= 10.0:
                log_alpha = ema_update(log_alpha, prev_log_avg_Ck, cfg.beta) if cfg.beta > 0 else prev_log_avg_Ck

            optimizer.zero_grad()
            log_Ck_list, log_u_list = [], []
            for j in range(M):
                log_Ck, log_u = compute_Ck_obj(model, x_batches[j], opt_ys[j], lamb)
                log_Ck_list.append(log_Ck)
                log_u_list.append(log_u)
            log_avg_Ck = torch.logsumexp(torch.stack(log_Ck_list), dim=0) - math.log(M)
            log_avg_Ck = ub_clip(log_avg_Ck, torch.tensor(prev_log_avg_Ck + cfg.log_E_Ck_max_delta, device=device))
            log_avg_Ck = lb_clip(log_avg_Ck, torch.tensor(prev_log_avg_Ck - cfg.log_E_Ck_max_delta, device=device))
            E_log_u = torch.cat(log_u_list).mean()
            # over-estimator of log E[C_k] via linearization of -log(.) around alpha (Eq. 8)
            log_E_Ck_est = torch.exp(log_avg_Ck - log_alpha) + log_alpha - 1
            loss = E_log_u + log_E_Ck_est
            loss.backward()
            optimizer.step()

            prev_log_avg_Ck = log_avg_Ck.item()

            record = dict(
                step=step, loss=loss.item(), log_alpha=log_alpha,
                log_avg_Ck=log_avg_Ck.item(), log_E_Ck_est=log_E_Ck_est.item(),
                E_log_u=E_log_u.item(), lamb=lamb,
            )
            if self.verbose:
                print(f"step {step}: loss={record['loss']:.4g} log_alpha={log_alpha:.4g} "
                      f"log_avg_Ck={record['log_avg_Ck']:.4g} E_log_u={record['E_log_u']:.4g}")
            if logger:
                logger.log(record)

            if self.ckpt_dir and ((step + 1) % cfg.checkpoint_interval == 0 or step + 1 == cfg.last_step):
                save_checkpoint(
                    f"{self.ckpt_dir}/step={step + 1}-lamb={lamb:.5g}-loss={loss.item():.3f}.pt",
                    model, optimizer,
                )

        if logger:
            logger.close()
        return model


@torch.no_grad()
def estimate_R_lower_bound(log_u_model: nn.Module, source, lamb: float, cfg: RDLBTrainConfig, device=None):
    """Final evaluation (``est_R_``). Uses 2M samples of C_k — the first M set the
    log expansion point alpha, the second M (with fresh log_u samples) estimate
    the objective xi (Eq. 15). Should be run with ``cfg.y_init == 'exhaustive'``.

    ``source`` is any object exposing ``.sample(batchsize) -> Tensor``.
    """
    import time

    device = device or next(log_u_model.parameters()).device
    M = cfg.num_Ck_samples
    assert M >= 1
    total = 2 * M
    log_Ck_samples, E_log_us = [], []
    t_start = time.perf_counter()
    for i in range(total):
        x = source.sample(cfg.batchsize).to(device)
        res = _run_optimize_y(cfg, log_u_model, x, lamb)
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
