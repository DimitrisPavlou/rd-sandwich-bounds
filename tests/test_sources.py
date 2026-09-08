import numpy as np
import torch

from rdsandwich.dataloader import (
    BananaSource, GaussianSource, NdBananaEmbedder, gaussian_analytical_rd, gen_gaussian_params,
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
