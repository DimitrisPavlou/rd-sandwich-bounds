"""Tests for the image-folder loaders, including the in-RAM ``preload`` path.

These exercise shape/range/dtype contracts and check that ``preload=True``
returns the same kind of batch as the lazy path while decoding each file only
once at construction.
"""
import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("PIL")

from rdsandwich.dataloader.image import ImageFolderSource, ImagePatchDataset


def _make_images(dirpath, n=5, size=(64, 80)):
    """Write ``n`` random RGB PNGs (H, W = size) and return their dir."""
    w, h = size
    for i in range(n):
        arr = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(dirpath / f"img_{i:02d}.png")
    return str(dirpath)


@pytest.mark.parametrize("preload", [False, True])
def test_patch_dataset_shape_and_range(tmp_path, preload):
    root = _make_images(tmp_path, n=6)
    ds = ImagePatchDataset(root, patchsize=32, preload=preload)
    assert len(ds) == 6
    x = ds[0]
    assert x.shape == (3, 32, 32)
    assert x.dtype == torch.float32
    assert 0.0 <= float(x.min()) and float(x.max()) <= 255.0


@pytest.mark.parametrize("preload", [False, True])
def test_folder_source_sample_batch(tmp_path, preload):
    root = _make_images(tmp_path, n=4)
    src = ImageFolderSource(root, patchsize=16, preload=preload)
    batch = src.sample(3)
    assert batch.shape == (3, 3, 16, 16)
    assert batch.dtype == torch.float32
    assert float(batch.max()) <= 255.0


def test_preload_decodes_once(tmp_path, monkeypatch):
    """With preload, files are opened at construction and never re-opened on access."""
    root = _make_images(tmp_path, n=4)
    ds = ImagePatchDataset(root, patchsize=16, preload=True)

    import rdsandwich.dataloader.image as image_mod
    calls = {"n": 0}
    real_open = Image.open

    def counting_open(*a, **k):
        calls["n"] += 1
        return real_open(*a, **k)

    monkeypatch.setattr(image_mod.Image, "open", counting_open)
    for i in range(len(ds)):
        _ = ds[i]
        _ = ds[i]
    assert calls["n"] == 0  # everything served from RAM


def test_preload_matches_lazy_dtype_and_values(tmp_path):
    """Preload must return the same content as lazy for a fixed crop (patch == image)."""
    root = _make_images(tmp_path, n=3, size=(24, 24))
    lazy = ImagePatchDataset(root, patchsize=24, preload=False)
    eager = ImagePatchDataset(root, patchsize=24, preload=True)
    # patchsize == image size -> the (only) crop is deterministic
    for i in range(len(lazy)):
        assert torch.equal(lazy[i], eager[i])


def test_smaller_than_patch_is_resized(tmp_path):
    root = _make_images(tmp_path, n=2, size=(20, 20))
    ds = ImagePatchDataset(root, patchsize=32, preload=True)
    x = ds[0]
    assert x.shape == (3, 32, 32)


def test_all_images_full_resolution(tmp_path):
    root = _make_images(tmp_path, n=3, size=(48, 40))  # W=48, H=40
    src = ImageFolderSource(root, patchsize=None, preload=True)
    imgs = list(src.all_images())
    assert len(imgs) == 3
    assert imgs[0].shape == (1, 3, 40, 48)
    assert imgs[0].dtype == torch.float32
