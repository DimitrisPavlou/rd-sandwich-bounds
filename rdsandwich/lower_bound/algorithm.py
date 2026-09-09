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


def optimize_y_vectorized(
    log_u_fun,
    x: torch.Tensor,
    lamb: float,
    num_steps: int = 500,
    lr: float = 1e-2,
    tol: float = 1e-6,
    init: str = "quick",
    quick_topn: int = 10,
    chunksize: Optional[int] = None,
    cand_chunk: Optional[int] = None,
    verbose: bool = False,
):
    """Vectorized equivalent of :func:`optimize_y`.

    Optimizes **all** init candidates simultaneously as a single ``[P, *dims]``
    tensor under one Adam, instead of hill-climbing from each of the P candidates
    in a Python loop. This is numerically equivalent to :func:`optimize_y` --
    Adam's state (m, v) is per-scalar-parameter, so the P candidates still evolve
    independently -- but collapses P sequential optimizations into one batched
    loop, which is dramatically faster on GPU (the sequential version is
    launch-bound on tiny per-candidate kernels).

    The only large intermediate is the pairwise ``[P, k, *dims]`` term. For
    high-dimensional (image) data set ``cand_chunk`` to process the P candidates
    in chunks, bounding peak memory to ``[cand_chunk, k, *dims]``; ``None``
    processes all P at once (fine for low-dim sources like the Gaussians).

    Returns the same ``dict(opt_y=, opt_log_supobj=)`` shape as
    :func:`optimize_y`, taking the best candidate.
    """
    device = x.device
    if num_steps <= 0:
        return dict(opt_y=x.mean(dim=0))

    logk = math.log(x.shape[0])
    with torch.no_grad():
        log_u_all = log_u_fun(x).squeeze(-1)  # [k]
        if init == "exhaustive":
            init_idx = torch.arange(x.shape[0], device=device)
        elif init == "quick":
            mse_pairwise = batch_mse(x, x, chunksize)  # [k, k]
            log_supobj_at_x = torch.logsumexp(-lamb * mse_pairwise - log_u_all.unsqueeze(0), dim=1) - logk
            init_idx = torch.argsort(log_supobj_at_x)[-quick_topn:]
        else:
            raise ValueError(init)

    def neg_log_supobj(y_chunk: torch.Tensor) -> torch.Tensor:
        # y_chunk: [c, *dims] -> [c], the negative log gamma_k(y) for each candidate.
        reduce = tuple(range(2, y_chunk.dim() + 1))
        mse = ((y_chunk.unsqueeze(1) - x.unsqueeze(0)) ** 2).mean(dim=reduce)  # [c, k]
        log_supobj = torch.logsumexp(-lamb * mse - log_u_all.unsqueeze(0), dim=1) - logk  # [c]
        return -log_supobj

    # Force grad on for the hill-climb so this works regardless of the caller's
    # grad mode (estimate_R_lower_bound runs under @torch.no_grad()).
    with torch.enable_grad():
        Y = x[init_idx].clone().detach().requires_grad_(True)  # [P, *dims]
        opt = torch.optim.Adam([Y], lr=lr)
        cc = cand_chunk or Y.shape[0]
        prev_loss = float("inf")
        for step in range(num_steps):
            opt.zero_grad()
            total = 0.0
            for s in range(0, Y.shape[0], cc):
                # Candidates are independent, so the sum's gradient w.r.t. Y[s:s+cc]
                # is exactly each candidate's own gradient; backward per chunk
                # accumulates into Y.grad, bounding the pairwise intermediate.
                chunk_loss = neg_log_supobj(Y[s: s + cc]).sum()
                chunk_loss.backward()
                total += chunk_loss.item()
            if not math.isfinite(total):
                if verbose:
                    print(f"\tloss saturated at step {step}, stopping")
                break
            if abs(prev_loss - total) < tol:
                break
            prev_loss = total
            opt.step()

    with torch.no_grad():
        # Chunk the final selection pass too, so cand_chunk bounds *all* the
        # [P, k, *dims] intermediates (this pass would otherwise OOM on images
        # despite a small cand_chunk).
        final_log_obj = torch.cat([
            -neg_log_supobj(Y[s: s + cc]) for s in range(0, Y.shape[0], cc)
        ])  # [P]
        best = int(torch.argmax(final_log_obj))
        return dict(opt_y=Y[best].detach().clone(), opt_log_supobj=final_log_obj[best].item())
