#!/usr/bin/env python
"""Aggregate an R-D upper-bound sweep and plot the sandwich figure.

After running e.g. ``scripts/run_sweep.py --config configs/gaussian_ub.yaml``,
each run leaves a folder under ``--checkpoint_dir`` containing a per-epoch
``record-*.jsonl`` log and a ``ckpt-*.pt`` checkpoint. This script:

  1. finds every ``rdub-*`` run folder,
  2. reads each run's converged (D, R) point from the last log line,
  3. reads that run's config from the checkpoint so rate can be converted to a
     common unit (nats per sample per dimension, as in the paper),
  4. draws one upper-bound curve per ``latent_dim``, and
  5. overlays the analytical true R(D) (for a Gaussian source, given its
     params) and any lower-bound results (``rd-*.npz`` from train_rdlb eval).

This reproduces the shape of Fig. 2a-top of Yang & Mandt (2022).

    python evaluation/plot_rdub.py --checkpoint_dir checkpoints/gaussian \
        --gparams_path data/gaussian/gaussian_params-dim=1000.npz \
        --out results/gaussian_sandwich.png
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
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rdsandwich.dataloader import gaussian_analytical_rd

LN2 = math.log(2.0)


def _last_jsonl_record(run_dir):
    logs = sorted(glob.glob(os.path.join(run_dir, "record-*.jsonl")))
    if not logs:
        return None
    with open(logs[-1], "r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    return json.loads(lines[-1]) if lines else None


def _load_cfg(run_dir):
    ckpts = sorted(glob.glob(os.path.join(run_dir, "ckpt-*.pt")))
    if not ckpts:
        return None
    state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    return (state.get("extra") or {}).get("cfg")


def _has_decoder(cfg):
    """True unless decoder_units marks a no-decoder (Z==Y / identity) run."""
    dec = cfg.get("decoder_units")
    return not (isinstance(dec, (list, tuple)) and len(dec) == 1 and dec[0] <= 0)


def collect_ub_points(checkpoint_dir, per_sample=False):
    """Return (curves, run_dirs, n_dim), where curves is a list of
    (label, [(D, R, lambda), ...]) ordered for plotting.

    Rate unit: nats per sample per *dimension* by default (the Gaussian panel,
    Fig. 2a); with ``per_sample=True`` it is nats per sample (total over all n
    dims), which is the convention of the physics/speech panels (Fig. 2b,c),
    whose rate axis runs to ~6 nats rather than ~0.4.

    Runs are grouped by (has-decoder, latent_dim) so that a Z==Y run and a
    decoder run at the same latent_dim (both dim(Z)=n) stay as *separate*
    curves rather than being merged.
    """
    groups = defaultdict(list)  # (sortkey, label) -> points
    run_dirs = sorted(glob.glob(os.path.join(checkpoint_dir, "rdub-*")))
    n_dim = None
    for run_dir in run_dirs:
        rec = _last_jsonl_record(run_dir)
        cfg = _load_cfg(run_dir)
        if rec is None or cfg is None:
            print(f"  (skipping {os.path.basename(run_dir)}: missing log or checkpoint)")
            continue
        n = int(cfg["data_dim"])
        n_dim = n
        ld = int(cfg["latent_dim"])
        lam = float(cfg["lmbda"])
        D = float(rec["mse"])                    # mean-squared error per dimension
        R = float(rec["rate"])                   # as logged
        if not cfg.get("nats", False):
            R *= LN2                             # bits -> nats
        # Normalize to per-sample nats first (undo the trainer's --rpd if set)...
        R = R * n if cfg.get("rpd", False) else R
        # ...then to the requested axis unit.
        if not per_sample:
            R = R / n                            # -> nats per sample per dimension
        if _has_decoder(cfg):
            frac = f" ({ld / n:g}n)" if n else ""
            label = f"dim(Z)={ld}{frac}"
            sortkey = (1, ld)                    # decoder runs after the Z=Y run
        else:
            label = "Z=Y"
            sortkey = (0, ld)
        groups[(sortkey, label)].append((D, R, lam))
    curves = [(label, sorted(pts)) for (sortkey, label), pts in sorted(groups.items())]
    return curves, run_dirs, n_dim


def analytical_curve(gparams_path, d_min, d_max, num=200):
    """(D_mean, R_nats_per_dim) points on the true Gaussian R(D)."""
    z = np.load(gparams_path)
    scale = z["scale"].astype(np.float64)
    n = scale.shape[0]
    d_grid = np.linspace(max(d_min, 1e-6), d_max, num)
    r_grid = [gaussian_analytical_rd(scale, D=dm * n, nats=True) / n for dm in d_grid]
    return d_grid, np.array(r_grid)


def collect_lb_lines(checkpoint_dir):
    """Best-effort: read train_rdlb eval outputs (rd-*.npz) as (lambda, intercept_per_sample)."""
    lines = []
    for npz_path in sorted(glob.glob(os.path.join(checkpoint_dir, "**", "rd-*.npz"), recursive=True)):
        try:
            data = np.load(npz_path)
            R_ = float(data["R_"])               # E[-log u], per-sample nats (the R-axis intercept)
            lam = None
            for tok in os.path.basename(npz_path).replace(".npz", "").split("-"):
                if tok.startswith("lamb="):
                    lam = float(tok[len("lamb="):])
            if lam is not None:
                lines.append((lam, R_))
        except Exception as e:  # noqa: BLE001 - best effort, keep plotting UB
            print(f"  (skipping LB file {npz_path}: {e})")
    return lines


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint_dir", default="checkpoints/gaussian")
    p.add_argument("--gparams_path", default="data/gaussian/gaussian_params-dim=1000.npz",
                   help="Gaussian params for the analytical true R(D); omit/point to a missing "
                        "file to skip the true curve (e.g. for non-Gaussian sources).")
    p.add_argument("--out", default="results/gaussian_sandwich.png")
    p.add_argument("--per_sample", action="store_true",
                   help="Plot rate as nats per sample (total over all n dims), the physics/"
                        "speech convention (Fig. 2b,c, axis ~0-6). Default is per-dimension "
                        "(the Gaussian panel, Fig. 2a).")
    p.add_argument("--title", default=None, help="Override the figure title.")
    p.add_argument("--lb_tangents", action="store_true",
                   help="Show each lower-bound lambda as its own tangent line instead of a single "
                        "envelope curve R_L(D) (the default, which matches the paper's two-curve figure).")
    p.add_argument("--show", action="store_true", help="Also open an interactive window.")
    args = p.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves, run_dirs, n_dim = collect_ub_points(args.checkpoint_dir, per_sample=args.per_sample)
    if not curves:
        raise SystemExit(f"No rdub-* runs with logs+checkpoints found under {args.checkpoint_dir!r}.")

    all_D = [D for _, pts in curves for (D, _, _) in pts]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Analytical true R(D), if a Gaussian params file is available.
    if args.gparams_path and os.path.exists(args.gparams_path):
        d_grid, r_grid = analytical_curve(args.gparams_path, min(all_D), max(all_D))
        ax.plot(d_grid, r_grid, "k-", lw=2, label="true R(D)")

    # One upper-bound curve per (decoder?, latent_dim) group.
    for label, pts in curves:
        Ds = [d for d, _, _ in pts]
        Rs = [r for _, r, _ in pts]
        ax.plot(Ds, Rs, "o-", ms=4, label=f"R_U(D), {label}")

    # Lower bound: each train_rdlb eval (one lambda) is a tangent line to R(D); the
    # certified bound is their upper envelope (max over lambda), drawn as ONE curve
    # like the paper. Pass --lb_tangents to instead show the individual tangent lines.
    lb_lines = collect_lb_lines(args.checkpoint_dir)
    if lb_lines and n_dim:
        d_lo, d_hi = min(all_D), max(all_D)
        dd = np.linspace(d_lo, d_hi, 200)
        norm = 1.0 if args.per_sample else float(n_dim)   # per-sample nats -> per dim
        if args.lb_tangents:
            for lam, intercept in sorted(lb_lines):
                r = np.clip((intercept - lam * dd) / norm, 0, None)
                ax.plot(dd, r, "--", lw=1, label=f"R_L(D), lamb={lam:g}")
        else:
            env = np.full_like(dd, -np.inf)
            for lam, intercept in lb_lines:
                env = np.maximum(env, (intercept - lam * dd) / norm)
            ax.plot(dd, np.clip(env, 0, None), "--", color="crimson", lw=2,
                    label="R_L(D)")

    ax.set_xlabel("Distortion (mean squared error)")
    ax.set_ylabel("Rate (nats per sample)" if args.per_sample
                  else "Rate (nats per sample per dimension)")
    ax.set_title(args.title or f"{os.path.basename(args.checkpoint_dir.rstrip('/\\')) or 'R-D'} R-D bound (n={n_dim})")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved figure to {args.out}")
    print(f"Aggregated {sum(len(pts) for _, pts in curves)} upper-bound runs "
          f"across curves {[label for label, _ in curves]}"
          + (f"; {len(lb_lines)} lower-bound line(s)" if lb_lines else ""))
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
