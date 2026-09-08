"""
Blahut-Arimoto (BA) algorithm for a discretized continuous source, used as a
ground-truth reference on 1D/2D marginals where it's still tractable
(Sec. 6, Figs. 2b/2c/7). The original implementation was pure NumPy/SciPy
(framework-agnostic), so this is a near-verbatim port of ``ba.py`` with the
CLI wrapped into a reusable function.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import logsumexp


def blahut_arimoto(Rho: np.ndarray, p_x: np.ndarray, lamb: float, steps: int = 1000,
                    tol: float = 1e-6, verbose: bool = False):
    """
    :param Rho: [|X|, |Y|] matrix of distortion values, Rho[i, j] = rho(X[i], Y[j]).
    :param p_x: length-|X| array of source probabilities.
    :param lamb: positive Lagrange multiplier on the distortion term.
    :param steps: max number of coordinate-descent steps.
    :param tol: relative-improvement convergence tolerance.
    :return: (records, log_Q, log_q_y)
    """
    Q = np.ones_like(Rho)
    Q /= np.sum(Q, axis=1, keepdims=True)
    log_Q = np.log(Q)

    p_x = p_x / np.sum(p_x)
    log_p_x = np.log(p_x)

    scaled_Rho = -lamb * Rho
    records = []
    ub_prev = np.inf
    rcd = {}
    for step in range(steps):
        log_q_y = logsumexp(log_p_x + log_Q.T, axis=1)

        log_Q = scaled_Rho + log_q_y
        log_Q -= logsumexp(log_Q, axis=1, keepdims=True)

        Q = np.exp(log_Q)
        rate = np.matmul(p_x, Q * (log_Q - log_q_y)).sum()
        distortion = np.matmul(p_x, (Q * Rho)).sum()
        ub = rate + lamb * distortion

        rcd = dict(step=step, ub=ub, D=distortion, R=rate)
        records.append(rcd)
        if verbose and (10 * step) % steps == 0:
            print(rcd)
        if (ub_prev - ub) / ub < tol:
            if verbose:
                print(f"Tolerance reached, terminating after {step} steps")
            break
        ub_prev = ub

    return records, log_Q, log_q_y


def bin_edges_to_grid_pts(edges: np.ndarray) -> np.ndarray:
    return (edges[:-1] + edges[1:]) / 2


def vectorized_mse(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    xs_shape, ys_shape = xs.shape, ys.shape
    assert np.all(np.asarray(xs_shape[1:]) == np.asarray(ys_shape[1:]))
    xs = np.expand_dims(xs, axis=1)
    num_data_dims = len(ys_shape) - 1
    axes = tuple(range(2, 2 + num_data_dims))
    return np.mean((xs - ys) ** 2, axis=axes)


def discretize_and_run_ba(
    samples: np.ndarray,
    lamb: float,
    bins: int = 50,
    steps: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False,
):
    """End-to-end: histogram-discretize ``samples`` (an [N, n] array) and run BA.

    Mirrors the ``__main__`` block of the original ``ba.py``.
    """
    n = samples.shape[1]
    counts, bin_edges = np.histogramdd(samples, bins=bins)[:2]
    grid_axes = [bin_edges_to_grid_pts(e) for e in bin_edges]
    meshgrid = np.meshgrid(*grid_axes, indexing="ij")
    grid_pts_flat = np.reshape(np.dstack(meshgrid), [-1, n])
    counts_flat = counts.ravel()

    good = counts_flat != 0
    src_alphabet = grid_pts_flat[good]
    src_dist = counts_flat[good]
    src_dist = src_dist / src_dist.sum()
    rep_alphabet = grid_pts_flat

    Rho = vectorized_mse(src_alphabet, rep_alphabet)
    if verbose:
        print(f"(source, reproduction) alphabets have size {Rho.shape}")
    return blahut_arimoto(Rho, src_dist, lamb, steps=steps, tol=tol, verbose=verbose)
