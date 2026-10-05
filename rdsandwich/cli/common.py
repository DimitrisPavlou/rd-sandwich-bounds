"""Arguments and helpers shared by every training / evaluation CLI."""
from __future__ import annotations

import argparse
from typing import List

from ..data import ImageFolderDataset, get_dataset


def int_list(s: str) -> List[int]:
    """Parse ``"4,8,16"`` -> ``[4, 8, 16]`` (an empty string gives ``[]``)."""
    return [int(i) for i in s.split(",") if i]


def new_parser(description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=description,
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def add_run_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("run")
    g.add_argument("--checkpoint_dir", default="./checkpoints",
                   help="Each run writes to <checkpoint_dir>/<run name>/.")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default=None, help="e.g. cuda, cuda:1, cpu (default: cuda if available).")
    g.add_argument("--verbose", "-V", action="store_true")


def add_data_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("data")
    g.add_argument("--dataset", required=True,
                   help="'gaussian' | 'banana' | path to a .npy/.npz file | path to an image "
                        "directory or glob. Paths are used as given, so they can point inside "
                        "data/ or anywhere else on disk.")
    g.add_argument("--data_dim", type=int, default=None,
                   help="Data dimension. Required for 'gaussian' (without --gparams_path) and "
                        "'banana'; inferred from the data otherwise.")
    g.add_argument("--gparams_path", default=None, help="'gaussian' only: .npz with loc/scale.")
    g.add_argument("--patchsize", type=int, default=None,
                   help="Images only: random square crop size (default: whole images).")
    g.add_argument("--max_images", type=int, default=None,
                   help="Images only: cap the number of images (e.g. for smoke tests).")
    g.add_argument("--preload", action="store_true",
                   help="Images only: decode every image into RAM (uint8) once at startup; "
                        "forces num_workers=0.")
    g.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader worker processes (finite datasets only).")


def load_dataset(args, device):
    """Build the dataset for ``args.dataset``. Synthetic and array data live on
    ``device``; image folders stay on the CPU and are moved per batch."""
    return get_dataset(
        args.dataset, data_dim=args.data_dim, gparams_path=args.gparams_path, seed=args.seed,
        device=device, patchsize=args.patchsize, max_images=args.max_images, preload=args.preload,
    )


def is_image_dataset(dataset) -> bool:
    return isinstance(dataset, ImageFolderDataset)


def example_shape(dataset):
    """Shape of one example (without the batch dimension)."""
    return tuple(dataset.sample(1).shape[1:])
