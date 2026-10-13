#!/usr/bin/env python
"""Evaluate trained image upper-bound models on whole test images, driven by a YAML config.

One config is one curve (one legend entry): a list of checkpoints, evaluated on every
dataset of the config. The model is rebuilt from each checkpoint, so no model flags.

* Fixed-rate models (``resnet_vae``, ``ms2020_vae``): one checkpoint per lambda; each
  is evaluated at the lambda it was trained with. Leave ``lambdas`` out.
* Variable-rate models (``variable_rate_lossy_vae``): one checkpoint, evaluated at every
  value of ``lambdas`` (which must lie in its training range).

Writes ``<results_dir>/<name>_<dataset>.json`` per dataset (format in
``rdsandwich/upper_bound/results.py``), read by ``evaluation/plot.py``. Datasets whose
file already exists are skipped unless ``--force``. Template:
``configs/templates/eval/curve.template.yaml``.

    python evaluation/evaluate.py --config configs/eval/vr_c64_l15.yaml
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import os
from typing import Dict, List, Optional

import torch

from rdsandwich.data.image import ImageFolderDataset
from rdsandwich.upper_bound.evaluate import evaluate_full_images, load_model
from rdsandwich.upper_bound.results import load_yaml_config, result_path, save_results
from rdsandwich.utils.io import final_checkpoint
from rdsandwich.utils.torch_utils import get_device, seed_everything


@dataclasses.dataclass
class EvalConfig:
    name: str                               # results/<name>_<dataset>.json; plot configs refer to it
    checkpoints: List[str]                  # files, run directories (-> their final checkpoint) or globs
    datasets: Dict[str, str]                # short name -> image directory or glob
    label: Optional[str] = None             # legend text (default: name)
    lambdas: Optional[List[float]] = None   # variable-rate models only
    results_dir: str = "results"
    use_ema: bool = True                    # EMA weights when the checkpoint has them
    device: Optional[str] = None            # default: cuda if available
    seed: int = 0                           # the latents are sampled
    max_images: Optional[int] = None        # first N images of each dataset (quick checks)

    def __post_init__(self):
        if isinstance(self.checkpoints, str):
            self.checkpoints = [self.checkpoints]


def resolve_checkpoints(patterns: List[str]) -> List[str]:
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise SystemExit(f"checkpoints: {pattern!r} matches nothing")
        paths += [final_checkpoint(m) if os.path.isdir(m) else m for m in matches]
    # A glob like ckpt-lambda=*.pt also matches the periodic ckpt-lambda=8-epoch=0010.pt files.
    runs = [os.path.dirname(os.path.abspath(path)) for path in paths]
    repeated = sorted({run for run in runs if runs.count(run) > 1})
    if repeated:
        raise SystemExit(f"checkpoints: several files of the same run {repeated}; a curve takes one "
                         "checkpoint per run (list the run directory to use its final checkpoint)")
    return paths


def lambdas_for(model, checkpoint: str, lambdas: Optional[List[float]]) -> List[float]:
    """The config's lambdas for a variable-rate model; a fixed-rate model's own lambda."""
    if getattr(model, "variable_rate", False):
        if not lambdas:
            raise SystemExit(f"{checkpoint} is a variable-rate model: set `lambdas` in the config")
        lo, hi = model.cfg.lmb_range
        outside = [lam for lam in lambdas if not lo <= lam <= hi]
        if outside:
            raise SystemExit(f"lambdas {outside} lie outside the model's training range [{lo:g}, {hi:g}]")
        return lambdas
    if lambdas:
        raise SystemExit(f"{checkpoint} is a fixed-rate model (trained at lambda={model.cfg.lmbda:g}): "
                         "remove `lambdas` from the config")
    return [model.cfg.lmbda]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="YAML file (see configs/templates/eval/).")
    p.add_argument("--force", action="store_true", help="Re-evaluate datasets that already have results.")
    args = p.parse_args()

    cfg = load_yaml_config(EvalConfig, args.config)
    lambdas = [float(lam) for lam in cfg.lambdas] if cfg.lambdas else None
    out_paths = {ds: result_path(cfg.results_dir, cfg.name, ds) for ds in cfg.datasets}
    todo = [ds for ds, path in out_paths.items() if args.force or not os.path.exists(path)]
    for ds in sorted(set(cfg.datasets) - set(todo)):
        print(f"[skip] {out_paths[ds]} exists (--force to redo)")
    if not todo:
        return

    checkpoints = resolve_checkpoints(cfg.checkpoints)
    device = get_device(cfg.device)
    datasets = {ds: ImageFolderDataset(cfg.datasets[ds], patchsize=None, max_images=cfg.max_images,
                                       preload=True) for ds in todo}
    points = {ds: [] for ds in todo}

    for ckpt in checkpoints:
        model, meta = load_model(ckpt, device, use_ema=cfg.use_ema)
        print(f"Loaded {ckpt} ({meta['model']}, {meta['num_params'] / 1e6:.1f}M parameters, "
              f"{meta['epochs']} epochs, {meta['weights']} weights)")
        variable = getattr(model, "variable_rate", False)
        for lam in lambdas_for(model, ckpt, lambdas):
            for ds, data in datasets.items():
                seed_everything(cfg.seed)  # per point, so a point does not depend on the ones before it
                res = evaluate_full_images(model, data.all_images(), model.max_stride, device,
                                           lmbda=lam if variable else None)
                means = {k: float(v.mean()) for k, v in res.items()}
                points[ds].append(dict(lmbda=float(lam), checkpoint=ckpt, **meta, **means,
                                       per_image={k: v.tolist() for k, v in res.items()}))
                print(f"  [{ds}] lambda={lam:<8g} bpp={means['bpp']:.4f}  psnr={means['psnr']:.3f} dB  "
                      f"mse={means['mse']:.3f}  (8-bit: psnr={means['psnr_uint8']:.3f} dB)")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for ds, pts in points.items():
        save_results(out_paths[ds], dict(
            name=cfg.name, label=cfg.label or cfg.name, dataset=ds, dataset_path=cfg.datasets[ds],
            images=[os.path.basename(path) for path in datasets[ds].paths], seed=cfg.seed,
            max_images=cfg.max_images, points=sorted(pts, key=lambda pt: pt["lmbda"]),
        ))
        print(f"Saved {out_paths[ds]} ({len(pts)} points)")


if __name__ == "__main__":
    main()
