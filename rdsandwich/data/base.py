"""Common data-source interface and the ``DataLoader`` factory.

Every training source ultimately provides batches of ``x`` (shape
``[B, *data_shape]``) to a trainer. Two flavours exist:

  * **analytic / infinite** sources (Gaussian, banana) generate fresh samples
    on demand — they implement ``.sample(batch_size) -> Tensor`` and are turned
    into an ``IterableDataset`` (``InfiniteBatchDataset``) by ``build_loader``.
  * **finite** sources (arrays from physics/speech, image folders) are ordinary
    map-style ``torch.utils.data.Dataset``s and get a standard shuffling
    ``DataLoader``.

``build_loader`` hides that distinction so a trainer just iterates an object
that yields ``x`` batches.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset


class Source:
    """Minimal interface for an analytic i.i.d. data source."""

    def sample(self, batchsize: int) -> torch.Tensor:
        raise NotImplementedError


class InfiniteBatchDataset(IterableDataset):
    """Wrap a ``sample_fn(batch_size) -> Tensor`` as an infinite stream of batches.

    Used with ``DataLoader(..., batch_size=None)`` so each iteration returns one
    ready-made batch (vectorized sampling), rather than one example at a time.
    """

    def __init__(self, sample_fn, batch_size: int):
        self.sample_fn = sample_fn
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield self.sample_fn(self.batch_size)


def build_loader(source, batch_size: int, *, shuffle: bool = True,
                 drop_last: bool = True, num_workers: int = 0) -> DataLoader:
    """Build a ``DataLoader`` of ``x`` batches from a source.

    Finite map-style ``Dataset``s get a standard shuffling loader; analytic
    sources exposing ``.sample`` are wrapped in an ``InfiniteBatchDataset``.
    """
    if isinstance(source, Dataset) and not isinstance(source, IterableDataset) and hasattr(source, "__len__"):
        return DataLoader(source, batch_size=batch_size, shuffle=shuffle,
                          drop_last=drop_last, num_workers=num_workers)
    if hasattr(source, "sample"):
        ds = InfiniteBatchDataset(source.sample, batch_size)
        return DataLoader(ds, batch_size=None, num_workers=num_workers)
    raise TypeError(f"Don't know how to build a DataLoader from {type(source).__name__}")
