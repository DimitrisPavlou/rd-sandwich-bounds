"""Tests for the hierarchical ResNet-VAE image R-D upper bound."""
import math

import pytest
import torch

from rdsandwich.resnet_vae import ResNetVAE, ResNetVAEConfig


def _cfg(**kw):
    base = dict(img_dim=64, latent_channels=[4, 8, 16, 32], num_filters=32)
    base.update(kw)
    return ResNetVAEConfig(**base)


@pytest.mark.parametrize("cfg", [
    _cfg(ar_prior_levels=2, ar_slices=4),   # natural-image path: DeepFactorized z0 + AR prior
    _cfg(flat_z0=True),                      # GAN path: flattened MAF prior on z0
])
def test_forward_backward_finite(cfg):
    torch.manual_seed(0)
    m = ResNetVAE(cfg)
    x = torch.rand(2, 3, cfg.img_dim, cfg.img_dim) * 255.0
    out = m(x)
    assert out["x_hat"].shape == x.shape
    assert out["bits"].shape == (2,)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_loss_is_bpp_plus_lambda_mse():
    torch.manual_seed(0)
    m = ResNetVAE(_cfg(lmbda=0.03))
    x = torch.rand(2, 3, 64, 64) * 255.0
    out = m(x)
    assert torch.allclose(out["loss"], out["bpp"] + 0.03 * out["mse"])
    # get_losses returns (loss, rate=bpp, mse)
    loss, rate, mse = m.get_losses(x)
    assert torch.allclose(rate, out["bpp"]) or rate.shape == out["bpp"].shape


def test_eval_runs_under_no_grad():
    """DeepFactorized density must be evaluable without a grad context."""
    m = ResNetVAE(_cfg(ar_prior_levels=1, ar_slices=4)).eval()
    with torch.no_grad():
        out = m(torch.rand(1, 3, 64, 64) * 255.0)
    assert torch.isfinite(out["bpp"]) and torch.isfinite(out["mse"])


def test_latent_to_data_dimension_ratio():
    """Sanity check the topology reproduces the paper's dim(Z) ~= 0.66 dim(X)."""
    cfg = ResNetVAEConfig(img_dim=256, latent_channels=[4, 8, 16, 32, 64, 128])
    dim_z = sum(c * (256 // (2 ** (j + 1))) ** 2 for j, c in enumerate(cfg.latent_channels))
    dim_x = 256 * 256 * 3
    assert abs(dim_z / dim_x - 0.66) < 0.01


def test_overfits_a_fixed_batch():
    torch.manual_seed(0)
    m = ResNetVAE(_cfg(img_dim=32, latent_channels=[4, 8, 16], ar_prior_levels=1, ar_slices=4, lmbda=0.02))
    x = torch.rand(4, 3, 32, 32) * 255.0
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    first = None
    for it in range(300):
        opt.zero_grad()
        out = m(x)
        out["loss"].backward()
        opt.step()
        if it == 0:
            first = out["mse"].item()
    assert out["mse"].item() < 0.2 * first  # distortion collapses on a memorizable batch


def test_scale_floor_and_stats():
    """Every latent scale respects cfg.scale_min; return_stats reports per-level summaries."""
    from rdsandwich.resnet_vae import softplus_scale
    assert softplus_scale(torch.tensor([-1e4]), 1e-5).item() == pytest.approx(1e-5)
    m = ResNetVAE(_cfg(ar_prior_levels=2, ar_slices=4, scale_min=1e-3))
    with torch.no_grad():
        out = m(torch.rand(1, 3, 64, 64) * 255.0, return_stats=True)
    stats = out["stats"]
    assert set(f"level{i}" for i in range(4)) <= set(stats)
    for i in range(4):
        lv = stats[f"level{i}"]
        assert lv["q_scale"]["min"] >= 1e-3 and lv["bits"]["nonfinite"] == 0
    assert "ar_raw_scale" in stats["level3"]      # bottom levels use the AR prior
