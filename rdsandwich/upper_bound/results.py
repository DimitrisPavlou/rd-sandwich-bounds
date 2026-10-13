"""Result files of ``evaluation/evaluate.py`` (read by ``evaluation/plot.py``), and the
YAML configs of both scripts. Kept free of torch, so plotting does not import it.

One result file per (curve, dataset), ``<results_dir>/<name>_<dataset>.json``::

    {"name": ..., "label": ..., "dataset": ..., "dataset_path": ..., "images": [file names],
     "seed": ..., "max_images": ...,
     "points": [{"lmbda", "checkpoint", "model", "epochs", "weights", "num_params",
                 "bpp", "mse", "psnr", "mse_uint8", "psnr_uint8",   # means over the images
                 "per_image": {"bpp": [...], "mse": [...], ...}}, ...]}   # sorted by lambda
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Dict, Type, TypeVar

import yaml

T = TypeVar("T")


def result_path(results_dir: str, name: str, dataset: str) -> str:
    return os.path.join(results_dir, f"{name}_{dataset}.json")


def save_results(path: str, results: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)


def load_results(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_yaml_config(cls: Type[T], path: str) -> T:
    """Build the dataclass ``cls`` from a YAML file; unknown or missing keys exit with a message."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    try:
        return cls(**raw)
    except TypeError as e:
        fields = ", ".join(f.name for f in dataclasses.fields(cls))
        raise SystemExit(f"{path}: {e} (keys: {fields})") from None
