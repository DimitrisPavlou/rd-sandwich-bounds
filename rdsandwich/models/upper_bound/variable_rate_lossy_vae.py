"""R-D upper bound for images: Duan et al.'s variable-rate ResNet-VAE.

Port of the model in

> Zhihao Duan, Jack Ma, Jiangpeng He, Fengqing Zhu. **An Improved Upper Bound on
> the Rate-Distortion Function of Images.** ICIP 2023. https://arxiv.org/abs/2309.02574

from ``lvae/models/rd`` in https://github.com/duanzhiihao/lossy-vae. The class
keeps the original's name, ``VariableRateLossyVAE``, and parameter names match
the original, so its state dicts load unchanged.

Model (paper Sec. 3, Fig. 2, Table 1):

  * A bottom-up ConvNeXt encoder producing features at 1/4 ... 1/64 of the input
    resolution, and a top-down path of ``N`` latent-variable blocks at 1/64 ... 1/4,
    starting from a learned constant ``bias``. Each block fuses the encoder feature
    with the top-down feature into a Gaussian posterior ``q(z_i | x, z_<i)``,
    computes a Gaussian prior ``p(z_i | z_<i)`` from the top-down feature alone,
    samples ``z_i`` from the posterior and adds it back into the top-down path.
  * Posterior and prior means go through the smoothing function of eq. 8,
    ``sign(a) * |a|^(1 - 0.5 tanh|a|)``, which bounds the gradient of the
    ``(mu - mu_hat)^2`` KL term (``smooth=False`` gives the paper's ablation).
  * Variable rate: every residual block is conditioned on the R-D multiplier
    lambda through AdaLN; during training lambda is drawn per image
    log-uniformly from ``lmb_range`` (default ``[4, 2048]``), so one model traces
    a continuous R-D curve.

The objective is Duan's: ``KL_nats / (3*H*W) + lambda * MSE``, with the MSE
taken against the image scaled to ``[-1, 1]``. Inputs and outputs follow the
upper-bound models' convention instead: ``forward(x)`` takes ``x`` in
``[0, 255]`` and returns ``x_hat`` in ``[0, 255]`` and per-image ``bits``;
``get_losses`` returns ``(loss, bpp, mse)`` with ``mse`` on the ``[0, 255]``
scale. Lambda therefore means Duan's ``lambda`` everywhere (``--lambda`` at
evaluation time).
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdsandwich.layers.convnext_adaln import (
    ConvNeXtAdaLNPatchDown,
    ConvNeXtBlockAdaLN,
    conv_k1s1,
    conv_k3s1,
    patch_downsample,
    patch_upsample,
    sinusoidal_embedding,
)
from rdsandwich.models.upper_bound._common import LN2, MSE_PM1_TO_255, gaussian_kl, tensor_stats

# Architecture presets: (base channels C, latents per scale from 1x1 to 16x16
# feature resolution w.r.t. a 64x64 input). Paper Tables 1 and 3.
PRESETS = {
    "base":    dict(base_channels=128, latents_per_scale=[1, 2, 3, 4, 5]),   # 186.7M
    "c96_l15": dict(base_channels=96,  latents_per_scale=[1, 2, 3, 4, 5]),   # 111.8M
    "c64_l15": dict(base_channels=64,  latents_per_scale=[1, 2, 3, 4, 5]),   # 55.8M
    "c64_l10": dict(base_channels=64,  latents_per_scale=[1, 2, 2, 2, 3]),   # 46.1M
    "c64_l5":  dict(base_channels=64,  latents_per_scale=[1, 1, 1, 1, 1]),   # 34.2M
    # not in the paper: the 15-latent structure at about the ResNet-VAE's size (15.7M)
    "c32_l15": dict(base_channels=32,  latents_per_scale=[1, 2, 3, 4, 5]),   # 18.6M
    "c48_l15": dict(base_channels=48,  latents_per_scale=[1, 2, 3, 4, 5]),   # 41.5M
    # tests / smoke runs only
    "tiny":    dict(base_channels=8, latents_per_scale=[1, 1, 1, 1, 1], z_dim=4,
                    enc_blocks_per_scale=[1, 1, 1, 1, 1], lmb_embed_dim=(32, 32)),
}


# --------------------------------------------------------------------------- #
# Smoothing functions
# --------------------------------------------------------------------------- #
def linear_sqrt(x: torch.Tensor, threshold: float = 6.0) -> torch.Tensor:
    """Eq. 8: ``sign(x) * |x|^(1 - 0.5 tanh|x|)``, ~linear near 0 and ~sqrt for large |x|.

    Above ``threshold`` it switches to the signed square root it converges to,
    for numerical stability (same as the original implementation).

    Computed in fp32: in fp16 the gradient at an exact 0 is NaN (the ``pow``
    branch), and under AMP the near-zero means of a freshly initialized latent
    block underflow to exact zeros, so one NaN would poison every gradient.
    In fp32 the gradient at 0 is 1, as the function is ~identity there.
    """
    x = x.float()
    x_abs = torch.abs(x)
    soft = torch.sign(x) * torch.pow(x_abs, 1 - 0.5 * torch.tanh(x_abs))
    soft = torch.where(x_abs == 0, input=x, other=soft)
    signed_sqrt = torch.sign(x) * torch.sqrt(x_abs + 1e-8)
    return torch.where(x_abs <= threshold, input=soft, other=signed_sqrt)


SOFTPLUS_BETA = math.log(2)


def std_smooth(v: torch.Tensor) -> torch.Tensor:
    """Positive scale from a raw output: softplus with beta = ln 2 (scale 1 at v = 0).
    https://arxiv.org/abs/2203.13751, Sec. 4.2."""
    return F.softplus(v, beta=SOFTPLUS_BETA, threshold=12)


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #
class LatentVariableBlock(nn.Module):
    """One latent variable ``z_i``: prior from the top-down feature, posterior from
    the top-down + encoder features, sample, and residual update of the top-down
    feature (Fig. 2, right)."""

    def __init__(self, width, zdim, embed_dim, enc_width=None, kernel_size=7, mlp_ratio=2,
                 smooth=True):
        super().__init__()
        self.in_channels = width
        self.out_channels = width
        self.smooth = smooth

        block = ConvNeXtBlockAdaLN
        enc_width = enc_width or width
        self.resnet_front = block(width, embed_dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.resnet_end = block(width, embed_dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.posterior0 = block(enc_width, embed_dim, kernel_size=kernel_size)
        self.posterior1 = block(width, embed_dim, kernel_size=kernel_size)
        self.posterior2 = block(width, embed_dim, kernel_size=kernel_size)
        self.post_merge = conv_k1s1(width + enc_width, width)
        self.posterior = conv_k3s1(width, zdim * 2)
        self.prior = conv_k1s1(width, zdim * 2)
        self.z_proj = conv_k1s1(zdim, width)

        self.is_latent_block = True

    def _mean(self, m):
        return linear_sqrt(m) if self.smooth else m

    def transform_prior(self, feature, emb):
        """prior p(z_i | z_<i)"""
        feature = self.resnet_front(feature, emb)
        pm, pv = self.prior(feature).chunk(2, dim=1)
        return feature, self._mean(pm), std_smooth(pv)

    def transform_posterior(self, feature, enc_feature, emb):
        """posterior q(z_i | z_<i, x)"""
        assert feature.shape[2:4] == enc_feature.shape[2:4]
        enc_feature = self.posterior0(enc_feature, emb)
        feature = self.posterior1(feature, emb)
        merged = self.post_merge(torch.cat([feature, enc_feature], dim=1))
        merged = self.posterior2(merged, emb)
        qm, qv = self.posterior(merged).chunk(2, dim=1)
        return self._mean(qm), std_smooth(qv)

    def forward(self, feature, emb, enc_feature, stats: Optional[dict] = None):
        """Returns the updated top-down feature and the elementwise KL (nats)."""
        feature, pm, pv = self.transform_prior(feature, emb)
        qm, qv = self.transform_posterior(feature, enc_feature, emb)
        kl = gaussian_kl(qm, qv, pm, pv)
        z = qm + qv * torch.randn_like(qm)
        if stats is not None:
            stats.update(q_loc=tensor_stats(qm), q_scale=tensor_stats(qv),
                         p_loc=tensor_stats(pm), p_scale=tensor_stats(pv),
                         z=tensor_stats(z), kl=tensor_stats(kl))
        feature = feature + self.z_proj(z)
        feature = self.resnet_end(feature, emb)
        return feature, kl


class FeatureExtractor(nn.Module):
    """Bottom-up encoder; keeps the last feature at every spatial resolution, keyed by height."""

    def __init__(self, blocks):
        super().__init__()
        self.enc_blocks = nn.ModuleList(blocks)

    def forward(self, x, emb):
        feature = x
        enc_features = OrderedDict()
        for block in self.enc_blocks:
            if getattr(block, "requires_embedding", False):
                feature = block(feature, emb)
            else:
                feature = block(feature)
            enc_features[int(feature.shape[2])] = feature
        return enc_features


# --------------------------------------------------------------------------- #
# Config + model
# --------------------------------------------------------------------------- #
@dataclass
class VariableRateLossyVAEConfig:
    base_channels: int = 128
    # Number of latent variables at the 1/64, 1/32, 1/16, 1/8, 1/4 resolutions (top-down).
    latents_per_scale: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    z_dim: int = 32
    # Encoder ConvNeXt blocks at the 1/4, 1/8, 1/16, 1/32, 1/64 resolutions.
    enc_blocks_per_scale: List[int] = field(default_factory=lambda: [6, 6, 6, 4, 4])
    smooth: bool = True
    # Variable rate: lambda ~ log-uniform(lmb_range) per image during training.
    lmb_range: Tuple[float, float] = (4.0, 2048.0)
    # Lambda used when none is given and the model is not training (evaluation).
    lmbda: Optional[float] = None
    lmb_embed_dim: Tuple[int, int] = (256, 256)
    sin_period: int = 64
    # Input normalization (mean / std of ImageNet pixels in [0, 1], as in the original).
    im_shift: float = -0.4546259594901961
    im_scale: float = 3.67572653978347

    @property
    def widths(self) -> List[int]:
        """Feature channels at the 1/4, 1/8, 1/16, 1/32, 1/64 resolutions (Table 1)."""
        c = self.base_channels
        return [2 * c, 4 * c, 5 * c, 6 * c, 6 * c]

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "VariableRateLossyVAEConfig":
        if name not in PRESETS:
            raise ValueError(f"Unknown preset {name!r}; choose from {sorted(PRESETS)}")
        kw = dict(PRESETS[name])
        kw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kw)


class VariableRateLossyVAE(nn.Module):
    MAX_LMB = 8192  # lambda normalization for the sinusoidal embedding
    max_stride = 64  # total downsampling; image sides must be multiples of this

    def __init__(self, cfg: VariableRateLossyVAEConfig):
        super().__init__()
        if len(cfg.latents_per_scale) != 5 or len(cfg.enc_blocks_per_scale) != 5:
            raise ValueError("latents_per_scale and enc_blocks_per_scale need 5 entries each")
        self.cfg = cfg
        widths = cfg.widths
        emb_dim = cfg.lmb_embed_dim[1]

        # Bottom-up path: 1/4 -> 1/64 (64x64 input: 16x16 -> 1x1).
        enc_blocks = [patch_downsample(3, widths[0], rate=4)]
        for s in range(5):
            enc_blocks += [ConvNeXtBlockAdaLN(widths[s], emb_dim)
                           for _ in range(cfg.enc_blocks_per_scale[s])]
            if s < 4:
                enc_blocks.append(ConvNeXtAdaLNPatchDown(widths[s], widths[s + 1], embed_dim=emb_dim))
        self.encoder = FeatureExtractor(enc_blocks)

        # Top-down path: 1/64 -> 1/4, then a 4x sub-pixel upsampling to the image.
        dec_widths = widths[::-1]
        dec_blocks = []
        for j in range(5):
            dec_blocks += [LatentVariableBlock(dec_widths[j], cfg.z_dim, emb_dim,
                                               enc_width=widths[4 - j], smooth=cfg.smooth)
                           for _ in range(cfg.latents_per_scale[j])]
            if j < 4:
                dec_blocks.append(patch_upsample(dec_widths[j], dec_widths[j + 1], rate=2))
            else:
                dec_blocks.append(patch_upsample(dec_widths[j], 3, rate=4))
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.bias = nn.Parameter(torch.zeros(1, dec_widths[0], 1, 1))
        self.num_latents = sum(cfg.latents_per_scale)

        # Lambda embedding: log -> sinusoidal -> MLP.
        self.lmb_embedding = nn.Sequential(
            nn.Linear(cfg.lmb_embed_dim[0], cfg.lmb_embed_dim[1]),
            nn.GELU(),
            nn.Linear(cfg.lmb_embed_dim[1], cfg.lmb_embed_dim[1]),
        )

    # ------------------------------------------------------------------ #
    # Lambda handling
    # ------------------------------------------------------------------ #
    def sample_lmb(self, n: int, device) -> torch.Tensor:
        """Per-image lambda, log-uniform on ``cfg.lmb_range``."""
        low, high = (math.log(v) for v in self.cfg.lmb_range)
        return torch.exp(low + (high - low) * torch.rand(n, device=device))

    def _lmb_tensor(self, lmbda: Union[float, torch.Tensor], n: int, device) -> torch.Tensor:
        lmb = torch.as_tensor(lmbda, dtype=torch.float32, device=device)
        lmb = lmb.expand(n) if lmb.dim() == 0 else lmb
        if lmb.shape != (n,):
            raise ValueError(f"lambda must be a scalar or have shape ({n},), got {tuple(lmb.shape)}")
        low, high = self.cfg.lmb_range
        if bool((lmb < low * (1 - 1e-6)).any() or (lmb > high * (1 + 1e-6)).any()):
            raise ValueError(
                f"lambda={lmb.min().item():g}..{lmb.max().item():g} is outside the training range "
                f"[{low:g}, {high:g}] (lambda is Duan's: KL nats/dim + lambda * MSE on [-1, 1]).")
        return lmb

    def lmb_embedding_of(self, lmb: torch.Tensor) -> torch.Tensor:
        scaled = torch.log(lmb) * self.cfg.sin_period / math.log(self.MAX_LMB)
        emb = sinusoidal_embedding(scaled, dim=self.cfg.lmb_embed_dim[0], max_period=self.cfg.sin_period)
        return self.lmb_embedding(emb)

    # ------------------------------------------------------------------ #
    def forward_end2end(self, im: torch.Tensor, lmb: torch.Tensor, stats: Optional[dict] = None):
        """``im`` in [0, 1] -> (``x_hat`` on the [-1, 1] scale, list of per-latent KL maps in nats)."""
        x = (im + self.cfg.im_shift) * self.cfg.im_scale
        emb = self.lmb_embedding_of(lmb)
        enc_features = self.encoder(x, emb)
        n, _, h, w = enc_features[min(enc_features.keys())].shape
        feature = self.bias.expand(n, -1, h, w)
        kls = []
        for block in self.dec_blocks:
            if getattr(block, "is_latent_block", False):
                block_stats = {} if stats is not None else None
                feature, kl = block(feature, emb, enc_features[int(feature.shape[2])], stats=block_stats)
                kls.append(kl)
                if stats is not None:
                    stats[f"latent{len(kls) - 1}"] = block_stats
            elif getattr(block, "requires_embedding", False):
                feature = block(feature, emb)
            else:
                feature = block(feature)
        return feature, kls

    def forward(self, x: torch.Tensor, lmbda: Union[None, float, torch.Tensor] = None,
                training: Optional[bool] = None, return_stats: bool = False):
        """``x``: images in [0, 255] with sides divisible by 64.

        ``lmbda``: Duan's lambda (scalar or per image). If None: sampled per image when
        training, else ``cfg.lmbda``. ``return_stats=True`` adds per-latent summaries
        under ``out["stats"]`` (for explosion diagnostics).
        """
        if training is None:
            training = self.training
        n, c, h, w = x.shape
        if h % self.max_stride or w % self.max_stride:
            raise ValueError(f"image sides must be multiples of {self.max_stride}, got {h}x{w}")
        if lmbda is None and training:
            lmb = self.sample_lmb(n, x.device)
        else:
            if lmbda is None:
                lmbda = self.cfg.lmbda
                if lmbda is None:
                    raise ValueError("No lambda: pass lmbda=... or set cfg.lmbda for evaluation.")
            lmb = self._lmb_tensor(lmbda, n, x.device)

        im = x / 255.0
        stats = {} if return_stats else None
        x_hat_pm1, kls = self.forward_end2end(im, lmb, stats=stats)

        nats = torch.stack([kl.flatten(1).sum(-1) for kl in kls], dim=0).sum(dim=0)  # [B]
        bits = nats / LN2
        rate = nats / (c * h * w)                                    # nats per sub-pixel
        target = im * 2.0 - 1.0
        mse_pm1 = ((x_hat_pm1 - target) ** 2).flatten(1).mean(-1)   # [B], on [-1, 1]
        loss = (rate + lmb * mse_pm1).mean()

        mses = mse_pm1 * MSE_PM1_TO_255                               # [B], on [0, 255]
        bpp = bits.sum() / (n * h * w)
        psnr = (20 * math.log10(255.0) - 10.0 * torch.log10(mses.clamp_min(1e-12))).mean()
        x_hat = (x_hat_pm1 + 1.0) * 127.5
        out = dict(loss=loss, bpp=bpp, mse=mses.mean(), mses=mses, bits=bits, x_hat=x_hat,
                   psnr=psnr, lmbda=lmb)
        if return_stats:
            stats.update(x_hat=tensor_stats(x_hat), loss=loss.item(), bpp=bpp.item(),
                         mse=mses.mean().item())
            out["stats"] = stats
        return out

    def get_losses(self, x):
        out = self(x)
        return out["loss"], out["bpp"], out["mse"]

    def diagnostics(self, x):
        """Per-latent statistics, for the trainer's explosion report."""
        return self(x, return_stats=True)["stats"]
