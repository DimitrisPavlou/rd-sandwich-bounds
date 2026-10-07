"""Upper-bound evaluators.

* ``evaluate_sampled``: (D, R) with sample mean +/- 95% CI over fresh batches
  from a source (Appendix A.6, 'Upper Bound Estimator'). Needs only
  ``get_losses(x)``.
* ``evaluate_full_images``: per-image bpp / MSE / PSNR over whole,
  variable-size images (Kodak / Tecnick). Needs ``forward(x)`` to return
  ``x_hat`` and per-image ``bits``.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def evaluate_sampled(model: nn.Module, source, batchsize: int, num_batches: int, device=None):
    """Estimate (D, R) with sample mean +/- 95% CI, as in Appendix A.6 ('Upper Bound Estimator')."""
    device = device or next(model.parameters()).device
    model.eval()
    rates, mses = [], []
    for _ in range(num_batches):
        x = source.sample(batchsize).to(device)
        _, rate, mse = model.get_losses(x)
        rates.append(rate.item())
        mses.append(mse.item())

    def mean_ci(vals):
        vals = np.array(vals)
        m, s = vals.mean(), vals.std(ddof=1)
        ci = 1.96 * s / math.sqrt(len(vals))
        return m, s, (m - ci, m + ci)

    r_mean, r_std, r_ci = mean_ci(rates)
    d_mean, d_std, d_ci = mean_ci(mses)
    return dict(rate_mean=r_mean, rate_std=r_std, rate_ci=r_ci,
                mse_mean=d_mean, mse_std=d_std, mse_ci=d_ci)


def _psnr(mse: float) -> float:
    return 20 * math.log10(255.0) - 10 * math.log10(max(mse, 1e-12))


@torch.no_grad()
def evaluate_full_images(model: nn.Module, images: Iterable[torch.Tensor], pad_factor: int,
                         device=None) -> Dict[str, np.ndarray]:
    """Per-image bpp / MSE / PSNR on whole images in [0, 255].

    Each ``[1, 3, H, W]`` image is reflect-padded to a multiple of ``pad_factor``
    (the model's total downsampling), passed through ``model``, and the
    reconstruction is cropped back to ``H x W`` and clipped to [0, 255]. Rate is
    the model's bits over the *original* pixel count.

    Distortion is reported twice from the same forward pass (the latents are
    sampled, so two passes would differ): ``mse`` / ``psnr`` on the clipped
    reconstruction, and ``mse_uint8`` / ``psnr_uint8`` with it also rounded to
    8-bit values. Both are valid upper-bound points (rounding is a function of
    ``x_hat``, so it cannot increase the information); rounding adds ~1/12 to
    the MSE once errors span several grey levels.
    """
    device = device or next(model.parameters()).device
    model.eval()
    res = {k: [] for k in ("bpp", "mse", "psnr", "mse_uint8", "psnr_uint8")}
    for x in images:
        x = x.to(device)
        _, _, H, W = x.shape
        pad_h, pad_w = (-H) % pad_factor, (-W) % pad_factor
        xp = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") if (pad_h or pad_w) else x
        out = model(xp)
        x_hat = torch.clamp(out["x_hat"][:, :, :H, :W], 0.0, 255.0)
        mse = torch.mean((x - x_hat) ** 2).item()
        mse_uint8 = torch.mean((x - torch.round(x_hat)) ** 2).item()
        res["bpp"].append(out["bits"].sum().item() / (H * W))
        res["mse"].append(mse)
        res["psnr"].append(_psnr(mse))
        res["mse_uint8"].append(mse_uint8)
        res["psnr_uint8"].append(_psnr(mse_uint8))
    return {k: np.array(v) for k, v in res.items()}
