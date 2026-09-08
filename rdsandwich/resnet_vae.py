"""
R-D upper bound specialized to images: a hierarchical convolutional VAE,
torch-ported from ``resnet_vae.py``.

Scope note: the original ``resnet_vae.py`` implements a 6-level *bidirectional*
inference ResNet-VAE (Kingma et al., 2016) with a channel-wise-autoregressive
top-level prior and 3x3 conv blocks from Cheng et al. (2020) — the same
machinery used in Minnen & Singh (2020). This port keeps the same top-level
structure (a ladder of strided-conv latent levels, each level's decoder
using the level below as context, an optional flow/prior on the top-most
latent, factorized-Gaussian variational posteriors) but uses a simpler
"top-down as separate encoder pass" (Sønderby-style, not fully bidirectional)
VAE at each level for tractability. It is drop-in with the same
``get_losses`` API as ``rdsandwich.rdub.RDUBModel`` so the same training
loop can drive both. For a research-grade, bit-exact reproduction of the
paper's Kodak/Tecnick numbers, port ``resnet_vae.py``'s bidirectional
inference pass level-by-level from this scaffold, or (for the *operational*
baselines, as opposed to the R-D upper bound itself) use
``rdsandwich.compression_baselines``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import MAF


def conv_act(in_ch, out_ch, stride=2, activation="leaky_relu"):
    act = {"leaky_relu": nn.LeakyReLU(0.2), "relu": nn.ReLU(), "selu": nn.SELU()}[activation]
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1), act)


def deconv_act(in_ch, out_ch, stride=2, activation="leaky_relu"):
    act = {"leaky_relu": nn.LeakyReLU(0.2), "relu": nn.ReLU(), "selu": nn.SELU()}[activation]
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 3, stride=stride, padding=1, output_padding=stride - 1), act
    )


@dataclass
class ResNetVAEConfig:
    in_channels: int = 3
    img_dim: int = 128
    latent_channels: List[int] = field(default_factory=lambda: [4, 8, 16, 32, 64, 128])
    num_filters: int = 256
    lmbda: float = 1e-4
    flat_z0: bool = False  # flatten the top-most latent and give it a MAF prior (small images / GAN experiments)
    maf_stacks: int = 3
    nats: bool = False


class LatentLevel(nn.Module):
    """One rung of the ladder: downsample -> produce a factorized-Gaussian
    stochastic tensor -> (during generation) upsample back."""

    def __init__(self, in_ch, hidden_ch, latent_ch):
        super().__init__()
        self.down = conv_act(in_ch, hidden_ch)
        self.to_qparams = nn.Conv2d(hidden_ch, 2 * latent_ch, 3, padding=1)
        self.up = deconv_act(latent_ch, in_ch)

    def encode(self, h):
        h = self.down(h)
        qparams = self.to_qparams(h)
        loc, raw_scale = qparams.chunk(2, dim=1)
        scale = F.softplus(raw_scale + 0.5413)  # softplus^{-1}(1)
        return h, loc, scale

    def sample(self, loc, scale):
        eps = torch.randn_like(loc)
        z = loc + scale * eps
        log_q = (-0.5 * (eps ** 2 + math.log(2 * math.pi)) - torch.log(scale)).flatten(1).sum(-1)
        return z, log_q

    def decode(self, z):
        return self.up(z)


class ResNetVAE(nn.Module):
    """Ladder VAE for images with a top-level standard-normal or MAF prior
    and standard-normal priors on every other level (a simplification of
    the original's channel-wise-autoregressive hyperprior)."""

    def __init__(self, cfg: ResNetVAEConfig):
        super().__init__()
        self.cfg = cfg
        n_levels = len(cfg.latent_channels)
        self.levels = nn.ModuleList()
        in_ch = cfg.in_channels
        hw = cfg.img_dim
        self._hw_per_level = []
        for lc in cfg.latent_channels:
            self.levels.append(LatentLevel(in_ch, cfg.num_filters, lc))
            in_ch = lc
            hw = (hw + 1) // 2
            self._hw_per_level.append(hw)
        self.output_head = nn.Conv2d(cfg.in_channels, cfg.in_channels, 3, padding=1)

        top_ch = cfg.latent_channels[-1]
        top_hw = self._hw_per_level[-1]
        self.top_dim = top_ch * top_hw * top_hw
        if cfg.flat_z0:
            self.top_prior = MAF(self.top_dim, n_stacks=cfg.maf_stacks, hidden_units=(32, 16))
        else:
            self.top_prior = None  # standard normal, closed form

    def encode_decode(self, x: torch.Tensor):
        h = x
        locs, scales, zs = [], [], []
        for level in self.levels:
            h, loc, scale = level.encode(h)
            z, _ = level.sample(loc, scale)
            locs.append(loc)
            scales.append(scale)
            zs.append(z)
            h = z

        # top-level rate: either standard normal (closed form) or a MAF prior on the flattened code
        z_top = zs[-1]
        eps_top = (z_top - locs[-1]) / scales[-1]
        log_q_top = (-0.5 * (eps_top ** 2 + math.log(2 * math.pi)) - torch.log(scales[-1])).flatten(1).sum(-1)
        if self.top_prior is not None:
            log_p_top = self.top_prior.log_prob(z_top.flatten(1))
        else:
            log_p_top = (-0.5 * (z_top ** 2 + math.log(2 * math.pi))).flatten(1).sum(-1)
        kl_top = log_q_top - log_p_top

        # remaining levels: standard-normal prior in closed form (KL has a closed-form expression)
        kl_rest = torch.zeros_like(kl_top)
        for loc, scale in zip(locs[:-1], scales[:-1]):
            kl = 0.5 * (scale ** 2 + loc ** 2 - 1 - 2 * torch.log(scale))
            kl_rest = kl_rest + kl.flatten(1).sum(-1)

        kl = kl_top + kl_rest

        h = zs[-1]
        for level in reversed(self.levels):
            h = level.decode(h)
            # (a simple ladder: no extra skip-connections beyond what's implicit in each level's
            # own decode; a full bidirectional-inference model would additionally fuse the
            # corresponding encoder-side z at this resolution here.)
        x_hat = self.output_head(h)
        return x_hat, kl

    def get_losses(self, x: torch.Tensor):
        x_hat, kl = self.encode_decode(x)
        mse = F.mse_loss(x_hat, x)
        rate = kl.mean() if self.cfg.nats else kl.mean() / math.log(2.0)
        loss = rate + self.cfg.lmbda * mse
        return loss, rate, mse

    def forward(self, x):
        return self.get_losses(x)
