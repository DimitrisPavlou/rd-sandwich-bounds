"""Every image upper-bound model follows the same output contract, whatever its
training range and loss units, so evaluation and plots report bpp and PSNR on the
[0, 255] scale for all of them:

  * ``forward(x)``: ``x`` in [0, 255] -> ``x_hat`` on the [0, 255] scale (same shape)
    and per-image ``bits`` of shape [B]; ``mses`` is the per-image MSE of ``x_hat``
    against ``x`` and ``bpp`` is bits per pixel.
  * ``get_losses(x)`` -> ``(loss, bpp, mse)`` scalars, ``mse`` on the [0, 255] scale.
"""
import pytest
import torch

from rdsandwich.models.upper_bound.ms2020_vae import MS2020VAE, MS2020VAEConfig
from rdsandwich.models.upper_bound.resnet_vae import ResNetVAE, ResNetVAEConfig
from rdsandwich.models.upper_bound.variable_rate_lossy_vae import (
    VariableRateLossyVAE,
    VariableRateLossyVAEConfig,
)

IMAGE_MODELS = {
    "resnet_vae": lambda: ResNetVAE(ResNetVAEConfig(latent_channels=[4, 8, 16], num_filters=16)),
    "resnet_vae-range=pm1": lambda: ResNetVAE(ResNetVAEConfig(
        latent_channels=[4, 8, 16], num_filters=16, image_range="pm1", lmbda=64.0)),
    "ms2020_vae": lambda: MS2020VAE(MS2020VAEConfig(
        latent_depth=40, hyperprior_depth=24, num_filters=32, num_slices=5,
        max_support_slices=3, hyper_synth_out=48)),
    "variable_rate_lossy_vae": lambda: VariableRateLossyVAE(
        VariableRateLossyVAEConfig.from_preset("tiny")),
}


@pytest.mark.parametrize("name", sorted(IMAGE_MODELS))
def test_forward_follows_the_output_contract(name):
    torch.manual_seed(0)
    model = IMAGE_MODELS[name]().train()
    x = torch.randint(0, 256, (2, 3, 64, 64)).float()
    out = model(x)
    assert out["x_hat"].shape == x.shape
    assert out["bits"].shape == (2,) and (out["bits"] > 0).all()
    assert torch.allclose(out["bpp"], out["bits"].sum() / (2 * 64 * 64))
    mses = ((out["x_hat"] - x) ** 2).flatten(1).mean(-1)   # x_hat is on the [0, 255] scale
    assert torch.allclose(out["mses"], mses, rtol=1e-4)
    assert torch.allclose(out["mse"], mses.mean(), rtol=1e-4)


@pytest.mark.parametrize("name", sorted(IMAGE_MODELS))
def test_get_losses_returns_loss_bpp_and_255_scale_mse(name):
    model = IMAGE_MODELS[name]().train()
    x = torch.randint(0, 256, (2, 3, 64, 64)).float()
    torch.manual_seed(1)
    loss, bpp, mse = model.get_losses(x)
    torch.manual_seed(1)
    out = model(x)
    assert loss.dim() == bpp.dim() == mse.dim() == 0
    assert torch.allclose(bpp, out["bpp"]) and torch.allclose(mse, out["mse"])
    assert torch.allclose(mse, ((out["x_hat"] - x) ** 2).mean(), rtol=1e-4)
