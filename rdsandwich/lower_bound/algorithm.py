"""Inner subroutine of the R-D lower bound: the C_k sup-partition estimator.

``C_k = sup_y (1/k) sum_i exp(-lambda * mse(x_i, y)) / u(x_i)`` (Eq. 7). For a
squared-error distortion this is (up to scale) a Gaussian-mixture density with
centroids ``x_1..x_k``, so its global mode is found by hill-climbing from the
centroids (``optimize_y``). ``compute_Ck_obj`` evaluates ``log gamma_k(y)`` (with
gradient w.r.t. the log-u parameters) at a fixed ``y*`` for the envelope-theorem
gradient. Restricted to squared error, as in the paper.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


def batch_mse(x: torch.Tensor, y: torch.Tensor, chunksize: Optional[int] = None) -> torch.Tensor:
    """Pairwise MSE between a [B, *dims] batch x and a [P, *dims] batch y -> [B, P].

    ``chunksize`` processes x in chunks to bound peak memory when both B and P
    are large (e.g. images).
    """
    B = x.shape[0]
    if chunksize is None:
        xs = x.unsqueeze(1)  # [B, 1, *dims]
        return ((xs - y) ** 2).mean(dim=tuple(range(2, xs.dim())))
    chunks = []
    for start in range(0, B, chunksize):
        xc = x[start: start + chunksize].unsqueeze(1)  # [C, 1, *dims]
        mse_c = ((xc - y) ** 2).mean(dim=tuple(range(2, xc.dim())))
        chunks.append(mse_c)
    return torch.cat(chunks, dim=0)


def compute_Ck_obj(log_u_fun, x: torch.Tensor, y: torch.Tensor, lamb: float):
    """log of gamma_k(y) = (1/k) sum_i exp(-lamb*mse(x_i,y) - log_u(x_i)), computed
    stably via logsumexp. Returns (log_Ck_at_y, log_u(x))."""
    reduce_dims = tuple(range(1, x.dim()))
    mse = ((x - y) ** 2).mean(dim=reduce_dims)
    log_u = log_u_fun(x).squeeze(-1)
    k = x.shape[0]
    log_gamma = torch.logsumexp(-lamb * mse - log_u, dim=0) - math.log(k)
    return log_gamma, log_u


def optimize_y(
    log_u_fun,
    x: torch.Tensor,
    lamb: float,
    num_steps: int = 500,
    lr: float = 1e-2,
    tol: float = 1e-6,
    init: str = "quick",
    quick_topn: int = 10,
    chunksize: Optional[int] = None,
    verbose: bool = False,
):
    """Approximately solve y* = argmax_y (1/k) sum_i exp(-lamb*mse(x_i,y))/u(x_i)
    by hill-climbing (gradient ascent) from one or more of the k data points.
    Since the objective for squared-error distortion is (up to scale) a
    Gaussian-mixture density with centroids x_1..x_k, each x_i is a natural,
    cheap starting point.
    """
    device = x.device
    if num_steps <= 0:
        return dict(opt_y=x.mean(dim=0))

    with torch.no_grad():
        log_u_all = log_u_fun(x).squeeze(-1)  # [k]
        if init == "exhaustive":
            init_idx = torch.arange(x.shape[0], device=device)
        elif init == "quick":
            mse_pairwise = batch_mse(x, x, chunksize)  # [k, k]
            log_supobj_at_x = torch.logsumexp(-lamb * mse_pairwise - log_u_all.unsqueeze(0), dim=1) - math.log(x.shape[0])
            init_idx = torch.argsort(log_supobj_at_x)[-quick_topn:]
        else:
            raise ValueError(init)

    reduce_dims = tuple(range(1, x.dim()))
    best_log_obj, best_y = -float("inf"), None
    # Force grad on for the hill-climb so this works regardless of the caller's
    # grad mode (estimate_R_lower_bound runs under @torch.no_grad()).
    with torch.enable_grad():
        for idx in init_idx:
            y = x[idx].clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([y], lr=lr)
            prev_loss = float("inf")
            for step in range(num_steps):
                opt.zero_grad()
                mse = ((x - y) ** 2).mean(dim=reduce_dims)
                log_supobj = torch.logsumexp(-lamb * mse - log_u_all, dim=0) - math.log(x.shape[0])
                loss = -log_supobj
                loss_val = loss.item()
                if not math.isfinite(loss_val):
                    if verbose:
                        print(f"\tloss saturated at step {step}, stopping this init")
                    break
                if abs(prev_loss - loss_val) < tol:
                    break
                prev_loss = loss_val
                loss.backward()
                opt.step()
            with torch.no_grad():
                mse = ((x - y) ** 2).mean(dim=reduce_dims)
                final_log_obj = (torch.logsumexp(-lamb * mse - log_u_all, dim=0) - math.log(x.shape[0])).item()
            if final_log_obj > best_log_obj:
                best_log_obj = final_log_obj
                best_y = y.detach().clone()

    return dict(opt_y=best_y, opt_log_supobj=best_log_obj)
