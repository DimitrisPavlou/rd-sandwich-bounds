import numpy as np
import pytest
import torch
from PIL import Image

from rdsandwich.data import (
    ArraySource, BananaSource, GaussianSource, ImageFolderDataset, NdBananaEmbedder,
    build_loader, dataset_name, gaussian_analytical_rd, gen_gaussian_params, get_dataset,
)


def test_gaussian_source_shape():
    src = GaussianSource.standard(16)
    x = src.sample(32)
    assert x.shape == (32, 16)


def test_banana_source_shape():
    src = BananaSource()
    x = src.sample(64)
    assert x.shape == (64, 2)


def test_nd_banana_embedder_shape():
    src = NdBananaEmbedder(n=10, seed=0)
    x = src.sample(8)
    assert x.shape == (8, 10)
    assert torch.all(x >= 0)  # softplus output


def test_gen_gaussian_params_range():
    params = gen_gaussian_params(dim=100, seed=0)
    assert params["loc"].shape == (100,)
    assert np.all(params["loc"] >= -0.5) and np.all(params["loc"] <= 0.5)
    assert np.all(params["scale"] >= 0.0) and np.all(params["scale"] <= 2.0)


def test_gaussian_analytical_rd_monotonic():
    scale = np.ones(8) * 2.0
    d_lo = gaussian_analytical_rd(scale, D=0.1)
    d_hi = gaussian_analytical_rd(scale, D=1.0)
    assert d_lo >= d_hi >= 0


# --------------------------------------------------------------------------- #
# get_dataset: synthetic names or a path (relative or absolute)
# --------------------------------------------------------------------------- #
def test_get_dataset_synthetic():
    assert get_dataset("gaussian", data_dim=3).sample(5).shape == (5, 3)
    assert isinstance(get_dataset("banana", data_dim=2), BananaSource)
    assert get_dataset("banana", data_dim=6).sample(4).shape == (4, 6)
    with pytest.raises(ValueError):
        get_dataset("gaussian")


def test_get_dataset_array_path(tmp_path):
    path = tmp_path / "x.npy"
    np.save(path, np.random.randn(20, 7).astype(np.float32))
    ds = get_dataset(str(path))
    assert isinstance(ds, ArraySource) and len(ds) == 20 and ds[0].shape == (7,)
    with pytest.raises(FileNotFoundError):
        get_dataset(str(tmp_path / "missing.npy"))


def test_get_dataset_image_folder_any_path(tmp_path):
    folder = tmp_path / "somewhere" / "my_images"
    folder.mkdir(parents=True)
    for i in range(3):
        Image.fromarray((np.random.rand(40, 50, 3) * 255).astype(np.uint8)).save(folder / f"{i}.png")
    ds = get_dataset(str(folder), patchsize=16)
    assert isinstance(ds, ImageFolderDataset) and len(ds) == 3
    assert next(iter(build_loader(ds, batch_size=2))).shape == (2, 3, 16, 16)
    assert ds.sample(4).shape == (4, 3, 16, 16)
    assert next(get_dataset(str(folder)).all_images()).shape == (1, 3, 40, 50)


def test_get_dataset_unknown_spec(tmp_path):
    with pytest.raises(ValueError):
        get_dataset(str(tmp_path / "does_not_exist"))


def test_dataset_name():
    assert dataset_name("data/kodak/") == "kodak"
    assert dataset_name("/mnt/hdd/Tecnick_TESTIMAGES/RGB/RGB_OR_1200x1200") == "tecnick"
    assert dataset_name("data/physics/ppzee-split=test.npy") == "ppzee-split=test"
