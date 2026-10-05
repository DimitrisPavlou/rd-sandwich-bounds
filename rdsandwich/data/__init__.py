"""Data sources and the DataLoader factory used by the trainers.

``get_dataset`` turns a ``--dataset`` spec into a dataset; ``build_loader``
turns any dataset into a DataLoader of ``x`` batches. Synthetic sources
(Gaussian, banana) sample on demand; finite ones (arrays, image folders) are
map-style datasets that also expose ``.sample(batchsize)``.
"""
from __future__ import annotations

import glob
import os
from typing import Optional

from .base import InfiniteBatchDataset, Source, build_loader
from .gaussian import GaussianSource, gaussian_analytical_rd, gen_gaussian_params
from .banana import BananaSource, NdBananaEmbedder
from .array import ArraySource, find_val_test_counterpart, load_array_source
from .image import ImageFolderDataset

__all__ = [
    "Source", "InfiniteBatchDataset", "build_loader",
    "GaussianSource", "gaussian_analytical_rd", "gen_gaussian_params",
    "BananaSource", "NdBananaEmbedder",
    "ArraySource", "load_array_source", "find_val_test_counterpart",
    "ImageFolderDataset",
    "SYNTHETIC_DATASETS", "get_dataset", "dataset_name",
]

SYNTHETIC_DATASETS = ("gaussian", "banana")


def get_dataset(
    spec: str,
    *,
    data_dim: Optional[int] = None,
    gparams_path: Optional[str] = None,
    seed: int = 0,
    device=None,
    patchsize: Optional[int] = None,
    max_images: Optional[int] = None,
    preload: bool = False,
):
    """Build a dataset from a ``--dataset`` spec.

    ``spec`` is either
      - a synthetic source name:
          'gaussian' : factorized Gaussian, params from ``gparams_path`` or N(0, I) of ``data_dim``
          'banana'   : the 2D banana source, or its random n-d embedding if ``data_dim > 2``
      - or a path, used exactly as given (relative to the working directory, or
        absolute -- e.g. a large dataset on another disk):
          a ``.npy`` / ``.npz`` file    -> ``ArraySource`` (rows are samples), placed on ``device``
          a directory or glob of images -> ``ImageFolderDataset`` (``patchsize`` crops;
                                           ``None`` = whole images), kept on the CPU
    """
    if spec == "gaussian":
        if gparams_path:
            return GaussianSource.from_npz(gparams_path, device=device)
        if data_dim is None:
            raise ValueError("dataset 'gaussian' needs --data_dim (or --gparams_path)")
        return GaussianSource.standard(data_dim, device=device)
    if spec == "banana":
        if data_dim is None:
            raise ValueError("dataset 'banana' needs --data_dim")
        if data_dim == 2:
            return BananaSource(device=device)
        return NdBananaEmbedder(data_dim, seed=seed, device=device)
    if spec.endswith(".npy") or spec.endswith(".npz"):
        if not os.path.isfile(spec):
            raise FileNotFoundError(f"No such array file: {spec!r}")
        return load_array_source(spec, device=device)
    if os.path.isdir(spec) or glob.glob(spec):
        return ImageFolderDataset(spec, patchsize=patchsize, max_images=max_images, preload=preload)
    raise ValueError(
        f"Unknown dataset {spec!r}: expected one of {SYNTHETIC_DATASETS}, a .npy/.npz file, "
        "or an existing image directory/glob."
    )


def dataset_name(spec: str) -> str:
    """Short name for a dataset spec, used in result filenames ('kodak', 'tecnick', ...)."""
    p = spec.rstrip("/").lower()
    if "kodak" in p:
        return "kodak"
    if "tecnick" in p:
        return "tecnick"
    base = os.path.basename(spec.rstrip("/"))
    return os.path.splitext(base)[0] or "eval"
