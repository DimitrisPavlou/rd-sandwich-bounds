#!/usr/bin/env python
"""Plot the per-epoch training loss, rate and MSE of ``train/train_ub.py`` runs from their
``record-*.jsonl`` logs, one figure per quantity:

    <results_dir>/train_loss/<name>.png
    <results_dir>/train_rate/<name>.png
    <results_dir>/train_mse/<name>.png

Each ``--runs`` entry is a run directory (all its ``record-*.jsonl`` files are merged by
epoch, so a run resumed across several jobs gives one curve) or a single ``.jsonl`` file.
``<name>`` is the run directory's name, so figures of different models do not collide.

    python evaluation/plot_training_curves.py --runs checkpoints/lvae_variable_rate_full_coco

These are training-set numbers (random crops), averaged over each epoch. For the
variable-rate model they are also averaged over the per-image lambdas drawn during
training, so they describe no single point of its R-D curve.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

# key in the log -> (output folder, y-axis label, title)
QUANTITIES = {
    "loss": ("train_loss", "Training loss", "Training loss"),
    "rate": ("train_rate", "Rate (bits per pixel)", "Training rate"),
    "mse": ("train_mse", "MSE ([0, 255] scale)", "Training distortion"),
}


def load_log(path: str) -> list:
    """Per-epoch records of a run directory or a single jsonl file, sorted by epoch. When a
    resumed run logged an epoch twice, the later file's record wins."""
    files = sorted(glob.glob(os.path.join(path, "record-*.jsonl"))) if os.path.isdir(path) else [path]
    by_epoch = {}
    for file in files:
        with open(file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    by_epoch[record["epoch"]] = record
    return [by_epoch[e] for e in sorted(by_epoch)]


def run_name(path: str) -> str:
    path = path.rstrip("/")
    return os.path.basename(path) if os.path.isdir(path) else os.path.basename(os.path.dirname(path))


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="Run directories or record-*.jsonl files.")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--linear", action="store_true", help="Linear y-axis (default: log).")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for path in args.runs:
        records = load_log(path)
        if not records:
            raise SystemExit(f"No training records under {path!r}.")
        name = run_name(path)
        epochs = [r["epoch"] + 1 for r in records]  # epochs completed
        for key, (folder, ylabel, title) in QUANTITIES.items():
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(epochs, [r[key] for r in records], color="tab:blue", lw=1.8)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title}: {name}", fontsize=10)
            if not args.linear:
                ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both")
            fig.tight_layout()
            out = os.path.join(args.results_dir, folder, f"{name}.png")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"Saved {out}  (final {key} = {records[-1][key]:.5g})")


if __name__ == "__main__":
    main()
