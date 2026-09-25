#!/usr/bin/env python
"""Plot the image R-D upper bound (R vs. D) and quality-rate (PSNR vs. bpp)
curves directly from the per-epoch training logs of ``train/train_image_ub.py``.

Use this when only the ``record-*.jsonl`` logs are available (no eval npz on
Kodak/Tecnick). Each ``rdub-model=<model>-lambda=<lambda>-...`` run folder
gives one point: rate (bpp) and MSE (on the [0, 255] scale) averaged over the
last ``--last_n`` epochs. PSNR is computed from that mean MSE.

Note: these are *training-set* numbers (random crops, continuous relaxation),
not the held-out eval that ``evaluation/plot_qr.py`` plots.

    python evaluation/plot_image_ub_training.py --checkpoint_dir checkpoints \
        --out_prefix results/resnet_vae_train
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

_RUN_RE = re.compile(r"rdub-model=(?P<model>[^-]+)-lambda=(?P<lmbda>[0-9.eE+-]+?)(?:-|$)")


def _load_records(run_dir):
    recs = []
    for path in sorted(glob.glob(os.path.join(run_dir, "record-*.jsonl"))):
        with open(path, "r", encoding="utf-8") as f:
            recs.extend(json.loads(ln) for ln in f if ln.strip())
    return sorted(recs, key=lambda r: r["epoch"])


def collect_points(checkpoint_dir, last_n):
    """Return {model: [(lambda, bpp, mse, psnr), ...]} sorted by lambda."""
    curves = defaultdict(list)
    for run_dir in sorted(glob.glob(os.path.join(checkpoint_dir, "rdub-*"))):
        m = _RUN_RE.match(os.path.basename(run_dir))
        recs = _load_records(run_dir) if m else []
        if not recs:
            print(f"  (skipping {os.path.basename(run_dir)}: no jsonl log)")
            continue
        tail = recs[-last_n:]
        bpp = float(np.mean([r["rate"] for r in tail]))
        mse = float(np.mean([r["mse"] for r in tail]))
        psnr = 20 * math.log10(255.0) - 10 * math.log10(mse)
        curves[m["model"]].append((float(m["lmbda"]), bpp, mse, psnr))
        print(f"  {m['model']:>12s}  lambda={float(m['lmbda']):<6g} epochs={recs[-1]['epoch'] + 1:<5d} "
              f"bpp={bpp:.4f}  mse={mse:.3f}  psnr={psnr:.3f} dB")
    return {k: sorted(v) for k, v in curves.items()}


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--last_n", type=int, default=50,
                   help="Average rate/MSE over this many final epochs.")
    p.add_argument("--out_prefix", default="results/resnet_vae_train")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    curves = collect_points(args.checkpoint_dir, args.last_n)
    if not curves:
        raise SystemExit(f"No rdub-* runs with jsonl logs under {args.checkpoint_dir!r}.")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2a78d6", "#e8743b", "#2f9e6a", "#8a5cd1"]
    figs = {
        "rd": ("Distortion (MSE, [0, 255] scale)", "Rate (bits per pixel)",
               "R-D upper bound $R_U(D)$ (training set)", lambda pt: (pt[2], pt[1])),
        "qr": ("Rate (bits per pixel)", "Quality (PSNR, dB)",
               "Quality-rate, R-D upper bound (training set)", lambda pt: (pt[1], pt[3])),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    for kind, (xlabel, ylabel, title, xy) in figs.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, (model, pts) in enumerate(sorted(curves.items())):
            xs, ys = zip(*(xy(pt) for pt in pts))
            ax.plot(xs, ys, "o-", color=colors[i % len(colors)], lw=2, ms=6, label=model)
            for pt, x, y in zip(pts, xs, ys):
                ax.annotate(f"λ={pt[0]:g}", (x, y), textcoords="offset points",
                            xytext=(6, 4) if kind == "rd" else (-6, 6),
                            ha="left" if kind == "rd" else "right", fontsize=8, color="#555555")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if len(curves) > 1:
            ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = f"{args.out_prefix}_{kind}.png"
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
