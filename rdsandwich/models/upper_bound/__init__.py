"""R-D upper-bound models (beta-VAEs).

Every model here implements ``get_losses(x) -> (loss, rate, distortion)``,
which is all ``rdsandwich.upper_bound.UpperBoundTrainer`` needs. The image
models additionally return ``x_hat`` and ``bits`` from ``forward(x)``, used by
the full-image evaluator.
"""
from .mlp_vae import RDUBConfig, RDUBModel, check_no_decoder
from .resnet_vae import ResNetVAE, ResNetVAEConfig
from .ms2020_vae import MS2020VAE, MS2020VAEConfig

__all__ = [
    "RDUBConfig", "RDUBModel", "check_no_decoder",
    "ResNetVAE", "ResNetVAEConfig",
    "MS2020VAE", "MS2020VAEConfig",
]
