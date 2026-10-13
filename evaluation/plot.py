#!/usr/bin/env python
"""Plot and compare curves evaluated by ``evaluation/evaluate.py``, driven by a YAML config.

For every dataset of the config, one figure per kind, with one line per curve:

* ``qr``: quality-rate, PSNR (dB) vs. bits per pixel;
* ``rd``: rate-distortion, bits per pixel vs. MSE on the [0, 255] scale.

With ``combined: true`` it also draws ``combined.png``: one row per kind, one column per
dataset, one legend. Legend labels end with the model's parameter count. A curve keeps its
color and marker in every figure. Only the result
files are read, so no model or GPU is needed. Template:
``configs/templates/plots/comparison.template.yaml``.

    python evaluation/plot.py --config configs/plots/images.yaml
"""
from __future__ import annotations

import argparse
import dataclasses
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rdsandwich.upper_bound.results import load_results, load_yaml_config, result_path  # noqa: E402

# Categorical colors in a fixed order (validated for color-vision deficiencies as adjacent
# pairs), each with its own marker, so curves differ by more than color.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# kind -> (x key, y key, x label, y label); "{u}" marks the 8-bit-rounded distortion
KINDS = {
    "qr": ("bpp", "psnr", "Rate (bits per pixel)", "Quality (PSNR{u}, dB)"),
    "rd": ("mse", "bpp", "Distortion (MSE{u}, [0, 255] scale)", "Rate (bits per pixel)"),
}
DISTORTIONS = {"float": "", "uint8": "_uint8"}  # -> suffix of the mse / psnr keys


@dataclasses.dataclass
class PlotConfig:
    out_dir: str                            # figures go here
    curves: List[str]                       # names of evaluation configs; the legend order
    datasets: List[str]                     # dataset short names, as in the evaluation configs
    results_dir: str = "results"
    kinds: List[str] = dataclasses.field(default_factory=lambda: ["qr", "rd"])
    distortion: str = "float"               # float | uint8 (8-bit-rounded reconstructions)
    combined: bool = True                   # also combined.png: kinds x datasets
    labels: Dict[str, str] = dataclasses.field(default_factory=dict)  # name -> legend text override

    def __post_init__(self):
        if set(self.kinds) - set(KINDS) or not self.kinds:
            raise SystemExit(f"kinds must be a non-empty subset of {sorted(KINDS)}, got {self.kinds}")
        if self.distortion not in DISTORTIONS:
            raise SystemExit(f"distortion must be one of {sorted(DISTORTIONS)}, got {self.distortion!r}")
        if len(self.curves) > len(COLORS):
            raise SystemExit(f"at most {len(COLORS)} curves per figure, got {len(self.curves)}")


def param_count(points: List[dict]) -> str:
    """``"18.6M params"`` (a range if the curve's checkpoints differ), or ``""`` for result
    files written before the count was recorded."""
    counts = [pt["num_params"] / 1e6 for pt in points if "num_params" in pt]
    if not counts:
        return ""
    lo, hi = min(counts), max(counts)
    return f"{lo:.1f}M params" if lo == hi else f"{lo:.1f}-{hi:.1f}M params"


def load_curves(cfg: PlotConfig, dataset: str) -> List[dict]:
    """The result files of ``dataset``, in the config's curve order; missing ones are skipped."""
    curves = []
    for i, name in enumerate(cfg.curves):
        path = result_path(cfg.results_dir, name, dataset)
        if not os.path.exists(path):
            print(f"  (no {path}; run evaluation/evaluate.py for {name!r} first; skipping)")
            continue
        res = load_results(path)
        label, params = cfg.labels.get(name, res["label"]), param_count(res["points"])
        curves.append(dict(label=f"{label}, {params}" if params else label, points=res["points"],
                           color=COLORS[i], marker=MARKERS[i]))
    return curves


def draw(ax, curves: List[dict], kind: str, distortion: str, title: str) -> None:
    xkey, ykey, xlabel, ylabel = KINDS[kind]
    suffix = DISTORTIONS[distortion]
    xkey, ykey = (k + suffix if k in ("mse", "psnr") else k for k in (xkey, ykey))
    for c in curves:
        pts = sorted(c["points"], key=lambda pt: pt["bpp"])
        ax.plot([pt[xkey] for pt in pts], [pt[ykey] for pt in pts], color=c["color"],
                marker=c["marker"], lw=1.0, ms=3.5, label=c["label"])
    rounded = " of 8-bit-rounded reconstructions" if suffix else ""
    ax.set_xlabel(xlabel.format(u=rounded))
    ax.set_ylabel(ylabel.format(u=rounded))
    ax.set_title(title)
    ax.grid(color="#d9d9d6", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, path: str) -> None:
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="YAML file (see configs/templates/plots/).")
    args = p.parse_args()

    cfg = load_yaml_config(PlotConfig, args.config)
    os.makedirs(cfg.out_dir, exist_ok=True)
    curves = {ds: load_curves(cfg, ds) for ds in cfg.datasets}
    datasets = [ds for ds in cfg.datasets if curves[ds]]
    if not datasets:
        raise SystemExit(f"No result files for {cfg.curves} in {cfg.results_dir!r}.")

    for ds in datasets:
        for kind in cfg.kinds:
            fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
            draw(ax, curves[ds], kind, cfg.distortion, ds)
            ax.legend(fontsize=9)
            save(fig, os.path.join(cfg.out_dir, f"{kind}_{ds}.png"))

    if cfg.combined:
        rows, cols = len(cfg.kinds), len(datasets)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False, layout="constrained")
        for r, kind in enumerate(cfg.kinds):
            for c, ds in enumerate(datasets):
                draw(axes[r][c], curves[ds], kind, cfg.distortion, ds)
        # one legend for all panels, with every curve that appears in any of them
        handles = {}
        for ax in axes.flat:
            for h, label in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(label, h)
        # as many columns as fit the figure width: ~14 characters of 9-pt text per inch,
        # plus the line sample of each entry
        fit = int(fig.get_figwidth() * 14) // (max(map(len, handles)) + 6)
        fig.legend(handles.values(), handles.keys(), loc="outside lower center",
                   ncol=max(1, min(len(handles), 4, fit)), fontsize=9, frameon=False)
        save(fig, os.path.join(cfg.out_dir, "combined.png"))


if __name__ == "__main__":
    main()
