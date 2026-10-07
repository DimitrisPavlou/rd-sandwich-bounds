#!/usr/bin/env python
"""Plot the natural-image quality-rate (Q-R) figure (paper Sec. 6.4, Fig. 3-Right
/ Fig. 11): PSNR vs. bits-per-pixel for the two proposed R-D **upper-bound**
models -- the sandwich-bounds ResNet-VAE and the Minnen & Singh 2020 beta-VAE.

Only upper-bound curves are drawn (no operational baselines). Each curve is one
point per lambda, read *directly* from the eval outputs written by
``evaluation/eval_ub.py`` (no retraining), so the script composes cleanly with
the train/eval CLI:

    <results_dir>/rdub-model=<key>-lambda=<lambda>-dataset=<dataset>.npz
        -> per-image ``bpp`` / ``mse`` / ``psnr`` (and ``mse_uint8`` / ``psnr_uint8``)
           arrays, averaged to one point.

``<key>`` is the model name plus any curve-defining tags (``resnet_vae``,
``resnet_vae-range=pm1``, ``variable_rate_lossy_vae-C=128-L=1_2_3_4_5-lmb=4_2048``, ...).
By default every key found in ``--results_dir`` for the dataset is plotted;
``--models`` restricts it to a comma-separated list of keys. Distortion is the
unrounded reconstruction's by default, the 8-bit-rounded one with ``--uint8``.

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
import re

import numpy as np

MODELS = {
    # base model -> (legend label, color, marker); tags in the key are appended to the label
    "resnet_vae": ("proposed $R_U(D)$ (ResNet-VAE)", "tab:blue", "o"),
    "ms2020_vae": ("proposed $R_U(D)$ (Minnen 2020 $\\beta$-VAE)", "tab:orange", "s"),
    "variable_rate_lossy_vae": ("$R_U(D)$ (variable-rate VAE, Duan et al. 2023)", "tab:green", "^"),
}
EXTRA_COLORS = ["tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]

_FILE_RE = re.compile(r"^rdub-model=(?P<key>.+?)-lambda=(?P<lmbda>.+?)-dataset=(?P<dataset>.+)\.npz$")


def find_keys(results_dir, dataset):
    """All model keys with eval results for ``dataset`` in ``results_dir``."""
    keys = set()
    for path in glob.glob(os.path.join(results_dir, f"rdub-model=*-dataset={dataset}.npz")):
        m = _FILE_RE.match(os.path.basename(path))
        if m and m["dataset"] == dataset:
            keys.add(m["key"])
    return sorted(keys)


def points_from_npz(results_dir, key, dataset, uint8=False):
    """One (bpp, psnr, mse) point per lambda, averaged over images, from the eval npz.

    Files written before the uint8 keys existed hold a single ``psnr`` / ``mse``,
    which was 8-bit rounded unless ``--no_cast_xhat`` was passed; they are used
    as-is, with a note.
    """
    pts = []
    for path in sorted(glob.glob(os.path.join(results_dir, f"rdub-model={key}-lambda=*-dataset={dataset}.npz"))):
        m = _FILE_RE.match(os.path.basename(path))
        if not m or m["key"] != key:  # the glob's * can span tags of another key
            continue
        d = np.load(path)
        if "psnr_uint8" in d:
            psnr, mse = (d["psnr_uint8"], d["mse_uint8"]) if uint8 else (d["psnr"], d["mse"])
        else:
            print(f"  note: {os.path.basename(path)} is an old-format result "
                  f"(one distortion, 8-bit rounded by default); plotted as-is")
            psnr, mse = d["psnr"], d["mse"]
        pts.append((float(np.mean(d["bpp"])), float(np.mean(psnr)), float(np.mean(mse))))
    return sorted(pts)


def styles(keys):
    """{key: (label, color, marker)}: each key gets its base model's style with its tags
    appended to the label; further keys of an already-used base model get extra colors."""
    out, used, n_extra = {}, set(), 0
    for key in keys:
        base, _, tags = key.partition("-")
        label, color, marker = MODELS.get(base, (base, None, "o"))
        if tags:
            label = f"{label} [{tags}]"
        if base in used:
            color = EXTRA_COLORS[n_extra % len(EXTRA_COLORS)]
            n_extra += 1
        used.add(base)
        out[key] = (label, color, marker)
    return out


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="kodak", help="Test set short name (kodak / tecnick).")
    p.add_argument("--models", type=lambda s: [m for m in s.split(",") if m], default=None,
                   help="Comma-separated model keys (default: every key found for the dataset).")
    p.add_argument("--results_dir", default="results/img_compression")
    p.add_argument("--out", default="results/qr_kodak.png")
    p.add_argument("--title", default=None)
    p.add_argument("--rd", action="store_true",
                   help="Plot the R-D curve (bpp vs. MSE) instead of quality-rate (PSNR vs. bpp).")
    p.add_argument("--uint8", action="store_true",
                   help="Use the distortion of the 8-bit-rounded reconstruction.")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    keys = args.models if args.models is not None else find_keys(args.results_dir, args.dataset)
    curves = []
    for model in keys:
        pts = points_from_npz(args.results_dir, model, args.dataset, uint8=args.uint8)
        if not pts:
            print(f"  (no eval npz for model={model!r}, dataset={args.dataset!r} in "
                  f"{args.results_dir!r}; skipping)")
            continue
        curves.append((model, pts))
    if not curves:
        raise SystemExit(
            f"No Q-R points found for {keys} on {args.dataset!r} under "
            f"{args.results_dir!r}. Run the *_eval.yaml sweeps first.")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    key_styles = styles([model for model, _ in curves])
    for model, pts in curves:
        label, color, marker = key_styles[model]
        bpp = [b for b, _, _ in pts]
        if args.rd:
            ax.plot([m for _, _, m in pts], bpp, marker=marker, color=color, lw=2, ms=5, label=label)
        else:
            ax.plot(bpp, [q for _, q, _ in pts], marker=marker, color=color, lw=2, ms=5, label=label)

    rounded = " of 8-bit-rounded reconstructions" if args.uint8 else ""
    if args.rd:
        ax.set_xlabel(f"Distortion (MSE{rounded}, [0, 255] scale)")
        ax.set_ylabel("Rate (bits per pixel)")
        ax.set_title(args.title or f"R-D upper bound on {args.dataset.capitalize()}")
    else:
        ax.set_xlabel("Rate (bits per pixel)")
        ax.set_ylabel(f"Quality (PSNR{rounded}, dB)")
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
