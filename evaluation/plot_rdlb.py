#!/usr/bin/env python
"""Aggregate an R-D lower-bound sweep and plot the lower bound vs the true R(D).

Companion to ``plot_rdub.py``. After running e.g.
``scripts/run_sweep.py --config configs/gaussian_lb.yaml`` (the paper's
varying-dimension standard-Gaussian LB study, Sec. 6.1 / Fig. 2a-bottom), each
run folder ``rdlb-dd=<n>-...-lamb=<lambda>-...`` holds an ``rd-*.npz`` written
by ``train_rdlb.py``'s eval. Each such file gives one number ``R_`` = the
estimated intercept xi(lambda) of a tangent line to the true R(D) curve, at
slope ``-lambda`` (Eq. 15).

Units (mirrors ``plot_rdub.collect_lb_lines`` exactly):
  * the distortion in the objective is per-dimension MSE ``d`` (``mean`` over
    dims), and ``lambda`` is its Lagrange multiplier, so the tangent line in
    *per-sample* nats is ``R_total(d) = R_ - lambda * d``;
  * dividing by ``n`` puts it on the paper's per-sample-per-dimension axis:
    ``R_perdim(d) = (R_ - lambda * d) / n``.
Each run contributes one such line; their upper envelope (the max over lambda,
clipped at 0) is the certified lower bound ``R_L(D)``.

This script:
  1. finds every ``rdlb-*`` run folder under ``--checkpoint_dir``,
  2. reads each run's ``R_`` from its ``rd-*.npz`` (and lambda / data_dim from
     the path, as ``plot_rdub`` already does for lambda),
  3. groups runs by ``data_dim`` (one lower-bound curve per dimension),
  4. draws, per dimension, the individual tangent lines (faint) and their upper
     envelope R_L(D) (bold), and
  5. overlays the analytical true R(D). For the ``gaussian_lb`` study the source
     is the *standard* Gaussian N(0, I), whose per-dimension R(D) is the single
     dimension-independent curve 0.5*ln(1/D) (D<=1); pass ``--gparams_path`` for
     a non-standard factorized Gaussian instead.

    python evaluation/plot_rdlb.py --checkpoint_dir checkpoints/gaussian_lb \
        --out results/gaussian_lower_bound.png
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rdsandwich.dataloader import gaussian_analytical_rd


def _parse_token(name: str, key: str):
    """Return the value of a ``key=<value>`` token in a dashed filename, or None."""
    for tok in name.replace(".npz", "").split("-"):
        if tok.startswith(key + "="):
            return tok[len(key) + 1:]
    return None


def collect_lb_runs(checkpoint_dir):
    """Return {data_dim: [(lambda, R_), ...]} from every rd-*.npz under rdlb-* runs."""
    groups = defaultdict(list)
    run_dirs = sorted(glob.glob(os.path.join(checkpoint_dir, "rdlb-*")))
    # Fall back to a recursive scan if the runs don't use the rdlb-* prefix.
    npz_glob = (
        [p for d in run_dirs for p in glob.glob(os.path.join(d, "rd-*.npz"))]
        or glob.glob(os.path.join(checkpoint_dir, "**", "rd-*.npz"), recursive=True)
    )
    for npz_path in sorted(npz_glob):
        run_name = os.path.basename(os.path.dirname(npz_path))
        file_name = os.path.basename(npz_path)
        dd = _parse_token(run_name, "dd")
        lam = _parse_token(file_name, "lamb")
        if dd is None or lam is None:
            print(f"  (skipping {npz_path}: can't parse dd/lamb from path)")
            continue
        try:
            R_ = float(np.load(npz_path)["R_"])
        except Exception as e:  # noqa: BLE001 - best effort
            print(f"  (skipping {npz_path}: {e})")
            continue
        groups[int(dd)].append((float(lam), R_))
    return {n: sorted(v) for n, v in sorted(groups.items())}


def collect_ba_points(ba_dir, n):
    """Read Blahut-Arimoto runs (scripts/run_ba.py output) as per-dim (D, R) points.

    Each ``ba-*`` folder holds a jsonl whose last line is the converged record
    ``{step, ub, D, R}``: ``R`` is per-sample nats and ``D`` is per-dimension MSE
    (``vectorized_mse`` averages over dims), matching the LB's axes. Rate is
    divided by ``n`` for the per-sample-per-dimension axis.
    """
    pts = []
    for jl in sorted(glob.glob(os.path.join(ba_dir, "**", "*.jsonl"), recursive=True)):
        with open(jl, "r", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        if not lines:
            continue
        rec = json.loads(lines[-1])
        pts.append((float(rec["D"]), float(rec["R"]) / n))
    return sorted(pts)


def lb_envelope(runs, n, d_grid):
    """Upper envelope (over lambda) of the per-dim tangent lines, clipped at 0."""
    env = np.full_like(d_grid, -np.inf)
    for lam, R_ in runs:
        env = np.maximum(env, (R_ - lam * d_grid) / n)
    return np.clip(env, 0.0, None)


def true_rd_perdim(scale, d_grid):
    """Analytical per-dimension R(D) for a factorized Gaussian with given scale.

    ``gaussian_analytical_rd`` takes the *total* distortion and returns the
    *total* (per-sample) rate; here d_grid is per-dimension MSE, so scale by n.
    """
    n = scale.shape[0]
    return np.array([gaussian_analytical_rd(scale, D=d * n, nats=True) / n for d in d_grid])


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint_dir", default="checkpoints/gaussian_lb")
    p.add_argument("--gparams_path", default=None,
                   help="Factorized-Gaussian params (loc/scale npz) for the true R(D). "
                        "Omit for the standard Gaussian N(0, I) used by the gaussian_lb study.")
    p.add_argument("--out", default="results/gaussian_lower_bound.png")
    p.add_argument("--num_grid", type=int, default=400)
    p.add_argument("--ba_dir", default=None,
                   help="Directory of Blahut-Arimoto runs (scripts/run_ba.py output) to overlay "
                        "as a discretized ground-truth reference, e.g. checkpoints/gaussian_lb/BA.")
    p.add_argument("--ba_dim", type=int, default=2,
                   help="Source dimension the BA runs used (to convert per-sample rate to per-dim).")
    p.add_argument("--no_tangents", action="store_true",
                   help="Hide the faint per-lambda tangent lines; show only the envelope.")
    p.add_argument("--show", action="store_true", help="Also open an interactive window.")
    args = p.parse_args()

    groups = collect_lb_runs(args.checkpoint_dir)
    if not groups:
        raise SystemExit(f"No rdlb-* runs with rd-*.npz found under {args.checkpoint_dir!r}.")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("viridis")
    dims = sorted(groups)

    # x-axis range: from just above 0 to where the *widest* envelope hits 0
    # (the smallest lambda reaches out to d = R_/lambda), bounded by the source
    # variance (1 for the standard Gaussian).
    d_max_data = max((R_ / lam) for runs in groups.values() for lam, R_ in runs)
    var_cap = 1.0 if not args.gparams_path else float(
        (np.load(args.gparams_path)["scale"].astype(np.float64) ** 2).max())
    d_hi = min(d_max_data, var_cap)
    d_grid = np.linspace(d_hi / args.num_grid, d_hi, args.num_grid)

    # True R(D): standard Gaussian (same per-dim curve for every n) unless a
    # factorized-Gaussian params file is given.
    if args.gparams_path and os.path.exists(args.gparams_path):
        scale = np.load(args.gparams_path)["scale"].astype(np.float64)
        ax.plot(d_grid, true_rd_perdim(scale, d_grid), "k-", lw=2.5, label="true R(D)")
    else:
        ax.plot(d_grid, true_rd_perdim(np.ones(1), d_grid), "k-", lw=2.5,
                label="true R(D) (standard Gaussian)")

    for i, n in enumerate(dims):
        runs = groups[n]
        color = cmap(i / max(len(dims) - 1, 1))
        if not args.no_tangents:
            for lam, R_ in runs:
                line = np.clip((R_ - lam * d_grid) / n, 0.0, None)
                ax.plot(d_grid, line, "-", color=color, lw=0.6, alpha=0.35)
        ax.plot(d_grid, lb_envelope(runs, n, d_grid), "o-", color=color, ms=0,
                lw=2, label=f"R_L(D), n={n} ({len(runs)} lamb)")

    # Blahut-Arimoto ground-truth reference (discretized), if provided.
    if args.ba_dir and os.path.isdir(args.ba_dir):
        ba_pts = collect_ba_points(args.ba_dir, args.ba_dim)
        if ba_pts:
            bd = [d for d, _ in ba_pts]
            br = [r for _, r in ba_pts]
            ax.plot(bd, br, "s--", color="crimson", ms=5, lw=1.2,
                    label=f"Blahut-Arimoto (n={args.ba_dim})")

    ax.set_xlabel("Distortion (mean squared error per dimension)")
    ax.set_ylabel("Rate (nats per sample per dimension)")
    ax.set_title("Gaussian R-D lower bound vs dimension")
    ax.set_xlim(0, d_hi)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved figure to {args.out}")
    print(f"Aggregated {sum(len(v) for v in groups.values())} lower-bound runs "
          f"across dimensions {dims}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
