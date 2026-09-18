"""R-D upper bound specialized to images: a hierarchical ResNet-VAE.

Faithful PyTorch port of the original ``resnet_vae.py`` (Yang & Mandt, 2022,
Sec. 6.3/6.4). The model is a beta-VAE whose Gaussian likelihood is induced by
the squared-error distortion, so its ``(D, R)`` operating points upper-bound the
source rate-distortion function (Theorem A.3 / Eq. 3 of the paper).

Architecture (see the paper's Appendix A.5.6 / A.5.7 and Cheng et al. 2020 /
Kingma et al. 2016 for the components):

  * A ladder of ``num_levels`` stages. Each stage has an ``EncoderBlock``
    (strided residual conv, GDN) going bottom-up and a ``DecoderBlock``
    (transposed residual conv) going top-down.
  * Each stage below the top has a ``LatentBlock`` implementing **bidirectional
    inference** (IAF paper, Kingma et al. 2016): the variational posterior at a
    level fuses bottom-up features (from the encoder) with top-down features
    (from the decoder above), and its parameters can be expressed *relative to*
    the conditional prior (``res_q_param``).
  * The bottom ``ar_prior_levels`` generative levels use a **channel-wise
    autoregressive (IAF) prior** (:class:`ChannelwiseARTransform`) for added
    expressiveness; the remaining levels use a Gaussian conditional prior.
  * The top-most latent ``z0`` is coded under either a per-channel
    :class:`DeepFactorized` prior (fully convolutional, the natural-image
    setting of Sec. 6.4) or, with ``flat_z0=True``, a flattened MAF prior (the
    fixed-size GAN-image setting of Sec. 6.3).

The objective is the rate-distortion Lagrangian ``bpp + lambda * MSE`` computed
on images in the ``[0, 255]`` range (matching the paper and the neural-
compression baselines it compares against). ``get_losses`` returns
``(loss, rate=bpp, mse)``; ``forward`` returns a dict with per-image metrics for
evaluation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import DeepFactorized, MAF, ChannelwiseARTransform, GDN
from .utils import SOFTPLUS_INV_1

LN2 = math.log(2.0)
LOG2PI = math.log(2.0 * math.pi)
LOGIT_OFFSET = 1.0


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def softplus_scale(raw: torch.Tensor) -> torch.Tensor:
    """Map a raw feature to a positive scale near 1 at init (softplus(x + softplus^-1(1)))."""
    return F.softplus(raw + SOFTPLUS_INV_1)


def normal_log_prob(x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return -0.5 * (((x - loc) / scale) ** 2 + LOG2PI) - torch.log(scale)


def gaussian_kl(q_loc, q_scale, p_loc, p_scale):
    """KL( N(q_loc, q_scale) || N(p_loc, p_scale) ), elementwise."""
    return (
        torch.log(p_scale) - torch.log(q_scale)
        + (q_scale ** 2 + (q_loc - p_loc) ** 2) / (2.0 * p_scale ** 2)
        - 0.5
    )


def enc_conv(in_ch, out_ch, k=3, stride=1, activation=None):
    """Strided (down) convolution; ``activation`` is an nn.Module or None."""
    conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=k // 2)
    return conv if activation is None else nn.Sequential(conv, activation)


def dec_conv(in_ch, out_ch, k=3, stride=1, activation=None):
    """Transposed (up) convolution when stride>1, else a same-size convolution."""
    if stride > 1:
        conv = nn.ConvTranspose2d(in_ch, out_ch, k, stride=stride, padding=k // 2,
                                  output_padding=stride - 1)
    else:
        conv = nn.Conv2d(in_ch, out_ch, k, stride=1, padding=k // 2)
    return conv if activation is None else nn.Sequential(conv, activation)


# --------------------------------------------------------------------------- #
# Residual blocks (Cheng et al. 2020 style)
# --------------------------------------------------------------------------- #
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, filters, k=3, stride=2, out_activation="gdn", scale_img=False):
        super().__init__()
        self.scale_img = scale_img
        act2 = GDN(filters) if out_activation == "gdn" else None
        self.conv1 = enc_conv(in_ch, filters, k, stride, activation=nn.LeakyReLU(0.2))
        self.conv2 = enc_conv(filters, filters, k, 1, activation=act2)
        if in_ch == filters and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = enc_conv(in_ch, filters, 1, stride, activation=None)

    def forward(self, x):
        if self.scale_img:
            x = x / 255.0 - 0.5
        return self.shortcut(x) + self.conv2(self.conv1(x))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, filters, k=3, stride=2, img_output=False):
        super().__init__()
        self.img_output = img_output
        if not img_output:
            self.conv1 = dec_conv(in_ch, filters, k, stride, activation=nn.LeakyReLU(0.2))
            self.conv2 = dec_conv(filters, filters, k, 1, activation=nn.LeakyReLU(0.2))
            if in_ch == filters and stride == 1:
                self.shortcut = nn.Identity()
            else:
                self.shortcut = dec_conv(in_ch, filters, 1, stride, activation=None)
        else:
            assert filters == 3, "image-output decoder block must emit 3 channels"
            self.conv = dec_conv(in_ch, filters, k, stride, activation=None)

    def forward(self, x):
        if self.img_output:
            return (self.conv(x) + 0.5) * 255.0
        return self.shortcut(x) + self.conv2(self.conv1(x))


# --------------------------------------------------------------------------- #
# Bidirectional-inference latent block
# --------------------------------------------------------------------------- #
class LatentBlock(nn.Module):
    """One rung of the ladder (below the top). Fuses bottom-up feature ``b`` and
    top-down feature ``t`` into a posterior, samples ``z``, returns the level's
    bits and the residually-updated ``t``.
    """

    def __init__(self, feature_channels, latent_channels, *, ar_prior=False,
                 res_q_param=True, ar_num_slices=0, ar_max_support_ratio=0.5, k=3):
        super().__init__()
        self.latent_channels = latent_channels
        self.ar_prior = ar_prior
        self.res_q_param = res_q_param

        self.inf_net = enc_conv(feature_channels, 2 * latent_channels, k, 1)      # bottom-up q stats
        self.td_inf_net = dec_conv(feature_channels, 2 * latent_channels, k, 1)   # top-down q stats
        self.gen_net = dec_conv(feature_channels, 3 * latent_channels, k, 1)      # det feats + prior stats
        self.t_adaptor = dec_conv(2 * latent_channels, feature_channels, 3, 1)    # [z, det_feats] -> t update

        if ar_prior:
            self.ar_shift = ChannelwiseARTransform(latent_channels, ar_num_slices, ar_max_support_ratio)
            self.ar_scale = ChannelwiseARTransform(latent_channels, ar_num_slices, ar_max_support_ratio)

    def forward(self, b, t):
        det_feats, p_loc, p_scale = torch.split(self.gen_net(t), self.latent_channels, dim=1)
        p_scale = softplus_scale(p_scale)

        bu_loc, bu_scale = torch.chunk(self.inf_net(b), 2, dim=1)
        td_loc, td_scale = torch.chunk(self.td_inf_net(t), 2, dim=1)
        q_loc = bu_loc + td_loc
        q_scale = softplus_scale(bu_scale) + softplus_scale(td_scale)
        if self.res_q_param:  # parameterize q relative to the prior
            q_loc = q_loc + p_loc
            q_scale = q_scale + p_scale

        z = q_loc + q_scale * torch.randn_like(q_loc)

        if self.ar_prior:
            log_q = normal_log_prob(z, q_loc, q_scale).flatten(1).sum(-1)
            shift = self.ar_shift(z)
            scale = torch.sigmoid(LOGIT_OFFSET + self.ar_scale(z))
            epsilon = scale * z + (1.0 - scale) * shift          # inverse AR transform z -> noise
            log_det_J = torch.log(scale).flatten(1).sum(-1)
            log_p = normal_log_prob(epsilon, p_loc, p_scale).flatten(1).sum(-1) + log_det_J
            bits = (log_q - log_p) / LN2
        else:
            kl = gaussian_kl(q_loc, q_scale, p_loc, p_scale)
            bits = kl.flatten(1).sum(-1) / LN2

        new_t_feats = self.t_adaptor(torch.cat([z, det_feats], dim=1))
        t = t + new_t_feats
        return z, bits, t


# --------------------------------------------------------------------------- #
# Config + model
# --------------------------------------------------------------------------- #
@dataclass
class ResNetVAEConfig:
    in_channels: int = 3
    img_dim: int = 256
    latent_channels: List[int] = field(default_factory=lambda: [4, 8, 16, 32, 64, 128])
    num_filters: int = 256
    downsample_factor: int = 2
    lmbda: float = 0.01
    # Channel-wise AR prior on the bottom `ar_prior_levels` generative levels.
    ar_prior_levels: int = 0
    ar_slices: int = 8
    ar_max_support_ratio: float = 0.5
    # Top-level prior.
    flat_z0: bool = False            # True -> flattened MAF prior (fixed-size GAN images)
    maf_units: List[int] = field(default_factory=lambda: [32, 16])
    maf_stacks: int = 3
    df_filters: List[int] = field(default_factory=lambda: [3, 3, 3])  # DeepFactorized layer widths


class ResNetVAE(nn.Module):
    def __init__(self, cfg: ResNetVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.num_levels = num_levels = len(cfg.latent_channels)
        self.z0_channels = z0_channels = cfg.latent_channels[-1]
        stride = cfg.downsample_factor
        nf = cfg.num_filters

        # Encoders in bottom-up (inference) order: enc[0] sees the image.
        self.encoders = nn.ModuleList()
        in_ch = cfg.in_channels
        for j in range(num_levels):
            topmost = j == num_levels - 1
            out_ch = 2 * z0_channels if topmost else nf
            self.encoders.append(EncoderBlock(
                in_ch, out_ch, stride=stride,
                out_activation=None if topmost else "gdn",
                scale_img=(j == 0),
            ))
            in_ch = out_ch

        # Decoders in top-down (generative) order: dec[num_levels-1] emits the image.
        self.decoders = nn.ModuleList()
        for i in range(num_levels):
            dec_in = z0_channels if i == 0 else nf
            if i == num_levels - 1:
                self.decoders.append(DecoderBlock(dec_in, cfg.in_channels, stride=stride, img_output=True))
            else:
                self.decoders.append(DecoderBlock(dec_in, nf, stride=stride, img_output=False))

        # Latent blocks in generative order; index 0 (top) has none (handled as z0).
        self.latent_blocks = nn.ModuleList()
        for i in range(num_levels):
            if i == 0:
                self.latent_blocks.append(nn.Identity())  # placeholder, unused
                continue
            bu_idx = num_levels - 1 - i
            use_ar = bu_idx < cfg.ar_prior_levels
            self.latent_blocks.append(LatentBlock(
                nf, cfg.latent_channels[bu_idx],
                ar_prior=use_ar, res_q_param=(not use_ar),
                ar_num_slices=cfg.ar_slices, ar_max_support_ratio=cfg.ar_max_support_ratio,
            ))

        # Top-level z0 prior.
        if cfg.flat_z0:
            z0_spatial = cfg.img_dim // (stride ** num_levels)
            assert z0_spatial * (stride ** num_levels) == cfg.img_dim, "img_dim must divide evenly"
            self.z0_spatial = z0_spatial
            self.z0_flat_dim = z0_spatial * z0_spatial * z0_channels
            self.top_prior = MAF(self.z0_flat_dim, n_stacks=cfg.maf_stacks,
                                 hidden_units=tuple(cfg.maf_units))
            self.df_prior = None
        else:
            self.z0_spatial = None
            self.df_prior = DeepFactorized(z0_channels, filters=tuple(cfg.df_filters))
            self.top_prior = None

    # ------------------------------------------------------------------ #
    def forward(self, x, training: Optional[bool] = None):
        if training is None:
            training = self.training

        # Bottom-up: collect features in generative (top-to-bottom) order.
        bu_features: List[torch.Tensor] = []
        fx = x
        for enc in self.encoders:
            fx = enc(fx)
            bu_features.insert(0, fx)  # last (topmost) ends at index 0

        bits_per_level: List[torch.Tensor] = []
        t = None
        for i in range(self.num_levels):
            b = bu_features[i]
            if i == 0:  # top latent z0
                q_loc, q_raw_scale = torch.chunk(b, 2, dim=1)
                q_scale = softplus_scale(q_raw_scale)
                z0 = q_loc + q_scale * torch.randn_like(q_loc)
                log_q = normal_log_prob(z0, q_loc, q_scale).flatten(1).sum(-1)
                if self.df_prior is not None:
                    log_p = self.df_prior.log_prob_nchw(z0)
                else:
                    z0_flat = z0.flatten(1)
                    log_p = self.top_prior.log_prob(z0_flat)
                z_bits = (log_q - log_p) / LN2
                t = z0
            else:
                _, z_bits, t = self.latent_blocks[i](b, t)
            bits_per_level.append(z_bits)
            t = self.decoders[i](t)

        x_hat = t
        bits = torch.stack(bits_per_level, dim=0).sum(dim=0)  # [B]

        mses = ((x - x_hat) ** 2).flatten(1).mean(-1)         # per image, in [0,255]^2 units
        mse = mses.mean()
        num_pixels = x.shape[0] * x.shape[-2] * x.shape[-1]
        bpp = bits.sum() / num_pixels
        loss = bpp + self.cfg.lmbda * mse
        # PSNR (dB) per image, at 8-bit peak, for logging/eval.
        psnr = (20 * math.log10(255.0) - 10.0 * torch.log10(mses.clamp_min(1e-12))).mean()
        return dict(loss=loss, bpp=bpp, mse=mse, mses=mses, bits=bits, x_hat=x_hat, psnr=psnr)

    def get_losses(self, x):
        out = self(x)
        return out["loss"], out["bpp"], out["mse"]
