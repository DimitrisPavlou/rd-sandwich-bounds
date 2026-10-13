"""Upper-bound evaluators.

* ``load_model``: rebuild a trained model from its checkpoint alone (no model flags).
* ``evaluate_sampled``: (D, R) with sample mean +/- 95% CI over fresh batches
  from a source (Appendix A.6, 'Upper Bound Estimator'). Needs only
  ``get_losses(x)``.
* ``evaluate_full_images``: per-image bpp / MSE / PSNR over whole,
  variable-size images (Kodak / Tecnick). Needs ``forward(x)`` to return
  ``x_hat`` and per-image ``bits``.
"""
from __future__ import annotations

import dataclasses
import math
import os
import re
import warnings
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdsandwich.models.upper_bound.mlp_vae import RDUBConfig, RDUBModel
from rdsandwich.models.upper_bound.ms2020_vae import MS2020VAE, MS2020VAEConfig
from rdsandwich.models.upper_bound.resnet_vae import ResNetVAE, ResNetVAEConfig
from rdsandwich.models.upper_bound.variable_rate_lossy_vae import VariableRateLossyVAE, VariableRateLossyVAEConfig

# The checkpoint's model name (extra["model"], written by train/train_ub.py) -> (model, config) classes.
MODEL_CLASSES = {
    "mlp_vae": (RDUBModel, RDUBConfig),
    "resnet_vae": (ResNetVAE, ResNetVAEConfig),
    "ms2020_vae": (MS2020VAE, MS2020VAEConfig),
    "variable_rate_lossy_vae": (VariableRateLossyVAE, VariableRateLossyVAEConfig),
}
# Config fields added after checkpoints were already being written, with the value those
# older checkpoints (whose saved cfg lacks the field) were trained with.
LEGACY_CONFIG_DEFAULTS = {  # both models had no scale floor before scale_min existed
    "resnet_vae": {"scale_min": 0.0},
    "variable_rate_lossy_vae": {"scale_min": 0.0},
}
_RUN_MODEL = re.compile(r"rdub-model=([a-z0-9_]+)")


def load_model(path: str, device=None, use_ema: bool = True) -> Tuple[nn.Module, Dict[str, Any]]:
    """Rebuild a trained upper-bound model from its checkpoint and load its weights.

    The architecture comes from the checkpoint itself (``extra["cfg"]`` and
    ``extra["model"]``, or the ``rdub-model=<name>`` run directory for checkpoints
    without the latter), so no model flags are needed. ``use_ema`` loads the EMA
    weights when the checkpoint has them.

    Returns the model, on ``device`` and in eval mode, and ``{"model": name,
    "epochs": epochs trained, "weights": "ema" | "raw", "num_params": parameter count}``.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    extra = state.get("extra") or {}
    name = extra.get("model")
    if name is None:
        m = _RUN_MODEL.search(os.path.basename(os.path.dirname(os.path.abspath(path))))
        name = m and m.group(1)
    if name not in MODEL_CLASSES:
        raise ValueError(f"{path}: unknown model {name!r}; expected one of {sorted(MODEL_CLASSES)}")
    if not extra.get("cfg"):
        raise ValueError(f"{path}: the checkpoint has no saved model config")

    model_cls, config_cls = MODEL_CLASSES[name]
    fields = {f.name for f in dataclasses.fields(config_cls)}
    unknown = sorted(set(extra["cfg"]) - fields)
    if unknown:
        warnings.warn(f"{path}: ignoring saved config fields the {name} config no longer has: {unknown}")
    cfg = {**LEGACY_CONFIG_DEFAULTS.get(name, {}), **{k: v for k, v in extra["cfg"].items() if k in fields}}
    model = model_cls(config_cls(**cfg))

    ema = use_ema and "ema_state_dict" in extra
    model.load_state_dict(extra["ema_state_dict"] if ema else state["model_state_dict"])
    model.to(device or "cpu").eval()
    epochs = int(extra["epoch"]) + 1 if "epoch" in extra else None
    return model, {"model": name, "epochs": epochs, "weights": "ema" if ema else "raw",
                   "num_params": sum(p.numel() for p in model.parameters())}


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
                         device=None, lmbda: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Per-image bpp / MSE / PSNR on whole images in [0, 255].

    Each ``[1, 3, H, W]`` image is reflect-padded to a multiple of ``pad_factor``
    (the model's total downsampling), passed through ``model``, and the
    reconstruction is cropped back to ``H x W`` and clipped to [0, 255]. Rate is
    the model's bits over the *original* pixel count. ``lmbda`` is passed to a
    variable-rate model's ``forward``; fixed-rate models have their own.

    Distortion is reported twice from the same forward pass (the latents are
    sampled, so two passes would differ): ``mse`` / ``psnr`` on the clipped
    reconstruction, and ``mse_uint8`` / ``psnr_uint8`` with it also rounded to
    8-bit values. Both are valid upper-bound points (rounding is a function of
    ``x_hat``, so it cannot increase the information); rounding adds ~1/12 to
    the MSE once errors span several grey levels.
    """
    device = device or next(model.parameters()).device
    model.eval()
    kwargs = {} if lmbda is None else {"lmbda": lmbda}
    res = {k: [] for k in ("bpp", "mse", "psnr", "mse_uint8", "psnr_uint8")}
    for x in images:
        x = x.to(device)
        _, _, H, W = x.shape
        pad_h, pad_w = (-H) % pad_factor, (-W) % pad_factor
        xp = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") if (pad_h or pad_w) else x
        out = model(xp, **kwargs)
        x_hat = torch.clamp(out["x_hat"][:, :, :H, :W], 0.0, 255.0)
        mse = torch.mean((x - x_hat) ** 2).item()
        mse_uint8 = torch.mean((x - torch.round(x_hat)) ** 2).item()
        res["bpp"].append(out["bits"].sum().item() / (H * W))
        res["mse"].append(mse)
        res["psnr"].append(_psnr(mse))
        res["mse_uint8"].append(mse_uint8)
        res["psnr_uint8"].append(_psnr(mse_uint8))
    return {k: np.array(v) for k, v in res.items()}
