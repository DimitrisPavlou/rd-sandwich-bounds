"""Data sources and the DataLoader factory used by the trainers.

Analytic sources (Gaussian, banana) sample on demand; finite ones (arrays,
image folders) are map-style datasets. ``get_source`` builds a source from a
spec string; ``build_loader`` turns any source into a DataLoader of ``x`` batches.
"""
from __future__ import annotations

from typing import Optional

from .base import InfiniteBatchDataset, Source, build_loader
from .gaussian import GaussianSource, gaussian_analytical_rd, gen_gaussian_params
from .banana import BananaSource, NdBananaEmbedder
from .array import ArraySource, find_val_test_counterpart, load_array_source

__all__ = [
    "Source", "InfiniteBatchDataset", "build_loader",
    "GaussianSource", "gaussian_analytical_rd", "gen_gaussian_params",
    "BananaSource", "NdBananaEmbedder",
    "ArraySource", "load_array_source", "find_val_test_counterpart",
    "get_source",
]

# Image loading pulls in Pillow/torchvision; keep it optional.
try:  # pragma: no cover - exercised only when the image deps are installed
    from .image import ImageFolderSource
    __all__.append("ImageFolderSource")
except ImportError:  # pragma: no cover
    ImageFolderSource = None


def get_source(
    dataset_spec: str,
    data_dim: Optional[int] = None,
    gparams_path: Optional[str] = None,
    seed: int = 0,
    device=None,
) -> Source:
    """Build a ``Source`` from a dataset spec string.

    ``dataset_spec`` is one of:
      - 'gaussian'  : factorized Gaussian, params from --gparams_path or standard
      - 'banana'    : 2D banana source, or its random n-d embedding if data_dim > 2
      - a path ending in .npy/.npz : loaded as an ArraySource (physics/speech/etc.)
    """
    if dataset_spec == "gaussian":
        if gparams_path:
            return GaussianSource.from_npz(gparams_path, device=device)
        assert data_dim is not None
        return GaussianSource.standard(data_dim, device=device)
    if dataset_spec == "banana":
        assert data_dim is not None
        if data_dim == 2:
            return BananaSource(device=device)
        return NdBananaEmbedder(data_dim, seed=seed, device=device)
    if dataset_spec.endswith(".npy") or dataset_spec.endswith(".npz"):
        return load_array_source(dataset_spec, device=device)
    raise NotImplementedError(f"Unknown dataset_spec: {dataset_spec!r}")
