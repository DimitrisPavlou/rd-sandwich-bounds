"""Finite datasets backed by a fixed array (particle physics / speech / any .npy)."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .base import Source


class ArraySource(Source, Dataset):
    """A fixed array of samples exposed both as a map-style ``Dataset`` (for a
    shuffling ``DataLoader``) and via ``.sample(batchsize)`` (legacy sampling
    with replacement, the analogue of tf.data's ``.shuffle().repeat().batch()``).
    """

    def __init__(self, array: np.ndarray, device=None):
        self.data = torch.as_tensor(array, dtype=torch.float32, device=device)
        self.device = device

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

    def sample(self, batchsize: int) -> torch.Tensor:
        idx = torch.randint(0, len(self), (batchsize,), device=self.data.device)
        return self.data[idx]


def load_array_source(path: str, device=None) -> ArraySource:
    if path.endswith(".npz"):
        arr = np.load(path)
        key = "data" if "data" in arr else list(arr.keys())[0]
        array = arr[key]
    else:
        array = np.load(path)
    return ArraySource(array, device=device)


def find_val_test_counterpart(train_path: str) -> Optional[str]:
    """Given a '...split=train....' path, look for the matching test/val file."""
    for split in ("test", "val"):
        candidate = train_path.replace("split=train", f"split={split}")
        if os.path.exists(candidate) and candidate != train_path:
            return candidate
    return None
