"""R-D upper bound specialized to images: a beta-VAE built on the Minnen &
Singh (2020) channel-wise autoregressive autoencoder.

This is the second "proposed" upper-bound model of Sec. 6.4 (the *orange*
Q-R curve in Fig. 3 / Fig. 11, ``rdub-model=ms2020_vae-*`` in the authors'
results). The authors' public repo ships only the *operational* NTC model
(``ms2020.py``, with rounding / entropy coding), used as a comparison baseline;
the beta-VAE upper-bound variant has no released implementation (only its output
``.npz`` files survive). We therefore port the autoencoder **architecture** from
``ms2020.py`` (Minnen & Singh 2020, and its layer table in Ballé et al. 2018's
appendix) to PyTorch and apply the sandwich-bounds recipe (paper Sec. 3 /
App. A.5.7):

  * ``Q_{Z|X}`` and ``Q_{Y|X}`` are **factorized Gaussians with learned means
    and variances** (the analysis and hyper-analysis transforms output twice as
    many channels, for a mean and a raw scale), sampled with the
    reparameterization trick -- *not* the factorized uniform posteriors that
    simulate rounding in the operational model.
  * The hyperprior ``Q_Z`` is the **deep factorized density, no longer convolved
    with a uniform** (:class:`DeepFactorized`, continuous).
  * The conditional prior ``p(y_i | z, y_{<i})`` is a Gaussian ``N(mu_i, s_i)``
    whose mean/scale come from the channel-conditional transforms, evaluated on
    the sampled ``y_i`` (no scale-index quantization table).
  * Latent residual prediction (LRP) is retained as part of the decoder ``omega``.

The objective is the R-D Lagrangian ``bpp + lambda * MSE`` on images in the
``[0, 255]`` range; ``(D, R)`` operating points upper-bound the source R(D).

Note on ``latent_depth``: the repo's operational ``ms2020.py`` uses
``latent_depth=320`` (our default, architecturally faithful). The paper reports
``dim(Z) ~= 0.28 dim(X)`` for its beta-VAE; at 256x256 that corresponds to
``latent_depth ~= 200`` (see :meth:`ResNetVAE`-style ratio note below). Expose
``latent_depth`` as a knob and pick per your reproduction target; the paper notes
increasing it did not improve their bound.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import DeepFactorized, GDN
from .resnet_vae import LN2, normal_log_prob, softplus_scale


# --------------------------------------------------------------------------- #
# Conv helpers (same-padding down/up convolutions), matching tfc SignalConv2D.
# --------------------------------------------------------------------------- #
def down(in_ch, out_ch, k, stride, act=None):
    conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=k // 2)
    return conv if act is None else nn.Sequential(conv, act)


def up(in_ch, out_ch, k, stride, act=None):
    if stride > 1:
        conv = nn.ConvTranspose2d(in_ch, out_ch, k, stride=stride, padding=k // 2,
                                  output_padding=stride - 1)
    else:
        conv = nn.Conv2d(in_ch, out_ch, k, stride=1, padding=k // 2)
    return conv if act is None else nn.Sequential(conv, act)


# --------------------------------------------------------------------------- #
# Transforms (Minnen & Singh 2020 / Ballé 2018 Table 1)
# --------------------------------------------------------------------------- #
class AnalysisTransform(nn.Module):
    """x/255 -> [GDN conv]x3 -> conv ; emits ``2*latent_depth`` (mean, raw scale)."""

    def __init__(self, latent_depth, num_filters=192):
        super().__init__()
        self.net = nn.Sequential(
            down(3, num_filters, 5, 2, GDN(num_filters)),
            down(num_filters, num_filters, 5, 2, GDN(num_filters)),
            down(num_filters, num_filters, 5, 2, GDN(num_filters)),
            down(num_filters, 2 * latent_depth, 5, 2, None),
        )

    def forward(self, x):
        return self.net(x / 255.0)


class SynthesisTransform(nn.Module):
    """y_hat -> [IGDN deconv]x3 -> deconv -> *255 (image in [0,255])."""

    def __init__(self, latent_depth, num_filters=192):
        super().__init__()
        self.net = nn.Sequential(
            up(latent_depth, num_filters, 5, 2, GDN(num_filters, inverse=True)),
            up(num_filters, num_filters, 5, 2, GDN(num_filters, inverse=True)),
            up(num_filters, num_filters, 5, 2, GDN(num_filters, inverse=True)),
            up(num_filters, 3, 5, 2, None),
        )

    def forward(self, y_hat):
        return self.net(y_hat) * 255.0


class HyperAnalysisTransform(nn.Module):
    """y -> conv/relu stack ; emits ``2*hyperprior_depth`` (mean, raw scale)."""

    def __init__(self, latent_depth, hyperprior_depth):
        super().__init__()
        self.net = nn.Sequential(
            down(latent_depth, 320, 3, 1, nn.ReLU()),
            down(320, 256, 5, 2, nn.ReLU()),
            down(256, 2 * hyperprior_depth, 5, 2, None),
        )

    def forward(self, y):
        return self.net(y)


class HyperSynthesisTransform(nn.Module):
    """z -> deconv/relu stack ; emits ``out_channels`` conditioning features."""

    def __init__(self, hyperprior_depth, out_channels=320):
        super().__init__()
        self.net = nn.Sequential(
            up(hyperprior_depth, 192, 5, 2, nn.ReLU()),
            up(192, 256, 5, 2, nn.ReLU()),
            up(256, out_channels, 3, 1, nn.ReLU()),
        )

    def forward(self, z):
        return self.net(z)


class SliceTransform(nn.Module):
    """Channel-conditional transform: conditioning features -> one slice's params."""

    def __init__(self, in_channels, slice_depth):
        super().__init__()
        self.net = nn.Sequential(
            down(in_channels, 224, 5, 1, nn.ReLU()),
            down(224, 128, 5, 1, nn.ReLU()),
            down(128, slice_depth, 3, 1, None),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
@dataclass
class MS2020VAEConfig:
    latent_depth: int = 320
    hyperprior_depth: int = 192
    num_filters: int = 192
    num_slices: int = 10
    max_support_slices: int = 5
    hyper_synth_out: int = 320
    df_filters: List[int] = field(default_factory=lambda: [3, 3, 3])
    lmbda: float = 0.01


class MS2020VAE(nn.Module):
    def __init__(self, cfg: MS2020VAEConfig):
        super().__init__()
        self.cfg = cfg
        ld, hd, ns = cfg.latent_depth, cfg.hyperprior_depth, cfg.num_slices
        if ld % ns != 0:
            raise ValueError(f"latent_depth ({ld}) must be divisible by num_slices ({ns}).")
        self.slice_depth = sd = ld // ns

        self.analysis = AnalysisTransform(ld, cfg.num_filters)
        self.synthesis = SynthesisTransform(ld, cfg.num_filters)
        self.hyper_analysis = HyperAnalysisTransform(ld, hd)
        self.hyper_synth_mean = HyperSynthesisTransform(hd, cfg.hyper_synth_out)
        self.hyper_synth_scale = HyperSynthesisTransform(hd, cfg.hyper_synth_out)

        cond = cfg.hyper_synth_out
        self.cc_mean = nn.ModuleList()
        self.cc_scale = nn.ModuleList()
        self.lrp = nn.ModuleList()
        for i in range(ns):
            n_support = min(i, cfg.max_support_slices)  # support = y_hat_slices[:max_support]
            base = cond + n_support * sd
            self.cc_mean.append(SliceTransform(base, sd))
            self.cc_scale.append(SliceTransform(base, sd))
            self.lrp.append(SliceTransform(base + sd, sd))

        self.z_prior = DeepFactorized(hd, filters=tuple(cfg.df_filters))

    # ------------------------------------------------------------------ #
    def forward(self, x, training: Optional[bool] = None):
        cfg = self.cfg
        # q(y | x): factorized Gaussian with learned mean/scale.
        y_loc, y_raw = torch.chunk(self.analysis(x), 2, dim=1)
        y_scale = softplus_scale(y_raw)
        y = y_loc + y_scale * torch.randn_like(y_loc)
        log_q_y = normal_log_prob(y, y_loc, y_scale).flatten(1).sum(-1)  # [B]

        # q(z | y): factorized Gaussian; p(z): deep factorized (not noise-convolved).
        z_loc, z_raw = torch.chunk(self.hyper_analysis(y), 2, dim=1)
        z_scale = softplus_scale(z_raw)
        z = z_loc + z_scale * torch.randn_like(z_loc)
        log_q_z = normal_log_prob(z, z_loc, z_scale).flatten(1).sum(-1)
        log_p_z = self.z_prior.log_prob_nchw(z)
        z_bits = (log_q_z - log_p_z) / LN2

        latent_means = self.hyper_synth_mean(z)
        latent_scales = self.hyper_synth_scale(z)

        # Channel-autoregressive prior over y slices + LRP decoder refinement.
        y_slices = torch.split(y, self.slice_depth, dim=1)
        y_hat_slices: List[torch.Tensor] = []
        log_p_y = torch.zeros_like(log_q_y)
        for i, y_slice in enumerate(y_slices):
            support = y_hat_slices[: cfg.max_support_slices]
            mean_support = torch.cat([latent_means, *support], dim=1)
            scale_support = torch.cat([latent_scales, *support], dim=1)
            mu = self.cc_mean[i](mean_support)
            p_scale = softplus_scale(self.cc_scale[i](scale_support))
            log_p_y = log_p_y + normal_log_prob(y_slice, mu, p_scale).flatten(1).sum(-1)

            # decoder-side latent residual prediction (no rounding in the beta-VAE).
            lrp = 0.5 * torch.tanh(self.lrp[i](torch.cat([mean_support, y_slice], dim=1)))
            y_hat_slices.append(y_slice + lrp)

        y_bits = (log_q_y - log_p_y) / LN2
        bits = y_bits + z_bits  # [B]

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.synthesis(y_hat)

        mses = ((x - x_hat) ** 2).flatten(1).mean(-1)
        mse = mses.mean()
        num_pixels = x.shape[0] * x.shape[-2] * x.shape[-1]
        bpp = bits.sum() / num_pixels
        loss = bpp + cfg.lmbda * mse
        psnr = (20 * math.log10(255.0) - 10.0 * torch.log10(mses.clamp_min(1e-12))).mean()
        return dict(loss=loss, bpp=bpp, mse=mse, mses=mses, bits=bits, x_hat=x_hat, psnr=psnr)

    def get_losses(self, x):
        out = self(x)
        return out["loss"], out["bpp"], out["mse"]
