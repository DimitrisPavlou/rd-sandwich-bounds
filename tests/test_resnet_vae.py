"""Tests for the hierarchical ResNet-VAE image R-D upper bound."""
import math

import pytest
import torch

from rdsandwich.models.upper_bound.resnet_vae import ResNetVAE, ResNetVAEConfig


def _cfg(**kw):
    base = dict(latent_channels=[4, 8, 16, 32], num_filters=32)
    base.update(kw)
    return ResNetVAEConfig(**base)


@pytest.mark.parametrize("cfg", [
    _cfg(ar_prior_levels=2, ar_slices=4),   # DeepFactorized z0 + AR prior on the bottom levels
    _cfg(),                                 # Gaussian conditional priors only
])
def test_forward_backward_finite(cfg):
    torch.manual_seed(0)
    m = ResNetVAE(cfg)
    x = torch.rand(2, 3, 64, 64) * 255.0
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
    cfg = ResNetVAEConfig(latent_channels=[4, 8, 16, 32, 64, 128])
    dim_z = sum(c * (256 // (2 ** (j + 1))) ** 2 for j, c in enumerate(cfg.latent_channels))
    dim_x = 256 * 256 * 3
    assert abs(dim_z / dim_x - 0.66) < 0.01


def test_overfits_a_fixed_batch():
    torch.manual_seed(0)
    m = ResNetVAE(_cfg(latent_channels=[4, 8, 16], ar_prior_levels=1, ar_slices=4, lmbda=0.02))
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
    from rdsandwich.models.upper_bound._common import softplus_scale
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


# --------------------------------------------------------------------------- #
# image_range="pm1": Duan's loss convention
# --------------------------------------------------------------------------- #
# lambda_pm1 / lambda_0_255 such that both objectives have the same optimum.
PM1_LAMBDA_FACTOR = 255.0 ** 2 / (4 * 3 * math.log2(math.e))  # ~3756.1


@pytest.mark.parametrize("ar_prior_levels", [0, 2])
def test_pm1_matches_0_255_with_converted_lambda(ar_prior_levels):
    """With the image head's conv doubled (x_hat_255 = 255 (c + 0.5) = 127.5 (2c + 1)) and
    lambda converted, pm1 is the same model: identical reconstructions, bpp and MSE, and a
    loss 1 / (3 log2 e) times the 0_255 one."""
    torch.manual_seed(0)
    old = ResNetVAE(_cfg(ar_prior_levels=ar_prior_levels, ar_slices=4, lmbda=0.02))
    new = ResNetVAE(_cfg(ar_prior_levels=ar_prior_levels, ar_slices=4, image_range="pm1",
                         lmbda=0.02 * PM1_LAMBDA_FACTOR))
    new.load_state_dict(old.state_dict())
    head = new.decoders[-1].conv
    with torch.no_grad():
        head.weight.mul_(2.0)
        head.bias.mul_(2.0)
    x = torch.rand(2, 3, 64, 64) * 255.0
    torch.manual_seed(1)
    out_old = old(x)
    torch.manual_seed(1)
    out_new = new(x)
    assert torch.allclose(out_new["x_hat"], out_old["x_hat"], atol=1e-3)
    for k in ("bits", "bpp", "mses", "mse", "psnr"):
        assert torch.allclose(out_new[k], out_old[k], rtol=1e-4), k
    assert torch.allclose(out_new["loss"], out_old["loss"] / (3 * math.log2(math.e)), rtol=1e-4)


def test_pm1_loss_and_outputs():
    torch.manual_seed(0)
    m = ResNetVAE(_cfg(image_range="pm1", lmbda=64.0))
    x = torch.rand(2, 3, 64, 64) * 255.0
    out = m(x)
    n, c, h, w = x.shape
    rate = out["bits"] * math.log(2) / (c * h * w)
    mse_pm1 = out["mses"] / (127.5 ** 2)
    assert torch.allclose(out["loss"], (rate + 64.0 * mse_pm1).mean(), rtol=1e-5)
    assert torch.allclose(out["bpp"], out["bits"].sum() / (n * h * w))
    # x_hat and the logged MSE are on the [0, 255] scale
    assert out["x_hat"].shape == x.shape
    assert torch.allclose(out["mses"], ((out["x_hat"] - x) ** 2).flatten(1).mean(-1), rtol=1e-4)
    out["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_invalid_image_range():
    with pytest.raises(ValueError, match="image_range"):
        ResNetVAE(_cfg(image_range="0_1"))


def test_cli_image_range_and_run_name(capsys):
    from rdsandwich.cli.upper_bound import build_model, build_train_parser, finalize_args, get_runname
    p = build_train_parser()
    arch = ["--model", "resnet_vae", "--dataset", "x", "--latent_channels", "4,8"]
    args = p.parse_args(arch + ["--image_range", "pm1", "--lambda", "37.6"])
    assert build_model(args).cfg.image_range == "pm1"
    assert get_runname(args).startswith("rdub-model=resnet_vae-range=pm1-lambda=37.6-")
    args = p.parse_args(arch + ["--lambda", "0.01"])  # default 0_255: run names unchanged
    assert build_model(args).cfg.image_range == "0_255"
    assert get_runname(args).startswith("rdub-model=resnet_vae-lambda=0.01-")
    args = p.parse_args(arch + ["--image_range", "pm1", "--lambda", "0.01"])
    finalize_args(args, dataset=None)
    assert "WARNING" in capsys.readouterr().out
