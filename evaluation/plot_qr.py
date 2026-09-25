#!/usr/bin/env python
"""Plot the natural-image quality-rate (Q-R) figure (paper Sec. 6.4, Fig. 3-Right
/ Fig. 11): PSNR vs. bits-per-pixel for the two proposed R-D **upper-bound**
models -- the sandwich-bounds ResNet-VAE and the Minnen & Singh 2020 beta-VAE.

Only these two upper-bound curves are drawn (no operational baselines). Each
curve is one point per lambda, read *directly* from the eval outputs written by
``train/train_image_ub.py`` (no retraining), so the script composes cleanly with
the train/eval CLI:

    <results_dir>/rdub-model=<model>-lambda=<lambda>-dataset=<dataset>.npz
        -> per-image ``bpp`` / ``mse`` / ``psnr`` arrays, averaged to one point.

    # after running the *_eval.yaml sweeps:
    python evaluation/plot_qr.py --dataset kodak  --out results/qr_kodak.png
    python evaluation/plot_qr.py --dataset tecnick --out results/qr_tecnick.png

Pass ``--rd`` to instead draw the R-D upper bound itself (bpp vs. MSE on the
[0, 255] scale) from the same eval files:

    python evaluation/plot_qr.py --dataset kodak --rd --out results/rd_kodak.png
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

MODELS = {
    # key -> (legend label, color, marker)
    "resnet_vae": ("proposed $R_U(D)$ (ResNet-VAE)", "tab:blue", "o"),
    "ms2020_vae": ("proposed $R_U(D)$ (Minnen 2020 $\\beta$-VAE)", "tab:orange", "s"),
}


def points_from_npz(results_dir, model, dataset):
    """One (bpp, psnr, mse) point per lambda, averaged over images, from the eval npz."""
    pts = []
    pattern = os.path.join(results_dir, f"rdub-model={model}-lambda=*-dataset={dataset}.npz")
    for path in sorted(glob.glob(pattern)):
        d = np.load(path)
        pts.append((float(np.mean(d["bpp"])), float(np.mean(d["psnr"])), float(np.mean(d["mse"]))))
    return sorted(pts)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="kodak", help="Test set short name (kodak / tecnick).")
    p.add_argument("--models", type=lambda s: [m for m in s.split(",") if m],
                   default=["resnet_vae", "ms2020_vae"])
    p.add_argument("--results_dir", default="results/img_compression")
    p.add_argument("--out", default="results/qr_kodak.png")
    p.add_argument("--title", default=None)
    p.add_argument("--rd", action="store_true",
                   help="Plot the R-D curve (bpp vs. MSE) instead of quality-rate (PSNR vs. bpp).")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    curves = []
    for model in args.models:
        pts = points_from_npz(args.results_dir, model, args.dataset)
        if not pts:
            print(f"  (no eval npz for model={model!r}, dataset={args.dataset!r} in "
                  f"{args.results_dir!r}; skipping)")
            continue
        curves.append((model, pts))
    if not curves:
        raise SystemExit(
            f"No Q-R points found for {args.models} on {args.dataset!r} under "
            f"{args.results_dir!r}. Run the *_eval.yaml sweeps first.")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for model, pts in curves:
        label, color, marker = MODELS.get(model, (model, None, "o"))
        bpp = [b for b, _, _ in pts]
        if args.rd:
            ax.plot([m for _, _, m in pts], bpp, marker=marker, color=color, lw=2, ms=5, label=label)
        else:
            ax.plot(bpp, [q for _, q, _ in pts], marker=marker, color=color, lw=2, ms=5, label=label)

    if args.rd:
        ax.set_xlabel("Distortion (MSE, [0, 255] scale)")
        ax.set_ylabel("Rate (bits per pixel)")
        ax.set_title(args.title or f"R-D upper bound on {args.dataset.capitalize()}")
    else:
        ax.set_xlabel("Rate (bits per pixel)")
        ax.set_ylabel("Quality (PSNR, dB)")
        ax.set_title(args.title or f"R-D upper bound (quality-rate) on {args.dataset.capitalize()}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved figure to {args.out}  ({', '.join(m for m, _ in curves)})")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
