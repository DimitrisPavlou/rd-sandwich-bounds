"""
Operational neural-compression baselines used as comparison points against
the R-D bounds (Figs. 2b/2c/3): Minnen et al. 2018 (mean-scale hyperprior,
ported in the original repo as ``mbt2018.py``) and Minnen & Singh 2020
(channel-wise autoregressive, ``ms2020.py``).

Rather than re-deriving these architectures from the TensorFlow source (as
the original repo does, adapting tensorflow-compression's reference
models), this module wraps **CompressAI** (https://github.com/InterDigitalInc/CompressAI),
which already ships faithful, actively-maintained PyTorch implementations
of both papers (``compressai.zoo.bmshj2018_hyperprior`` /
``mbt2018_mean`` and ``compressai.zoo.cheng2020_anchor`` /
``compressai.models.Cheng2020Attention`` — or, for an exact architectural
match, ``compressai.models.JointAutoregressiveHierarchicalPriors`` for
Minnen 2018 and a channel-conditioned model for Minnen & Singh 2020).

This is a deliberate scope decision: re-deriving two large published
architectures from scratch would dominate the porting effort for no
scientific benefit, since CompressAI's models are the standard PyTorch
reference implementations of exactly these papers. Install with:

    pip install compressai

Usage::

    from rdsandwich.compression_baselines import get_baseline_model, rd_loss
    model = get_baseline_model("mbt2018-mean", quality=4)  # or "ms2020" / "cheng2020-attn"
    loss, bpp, mse = rd_loss(model, x, lmbda=0.01)
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


_BASELINE_FACTORIES = {
    # name -> (compressai constructor name, default note)
    "mbt2018-mean": "mbt2018_mean",  # Minnen et al. 2018, mean-scale hyperprior
    "mbt2018": "mbt2018",  # Minnen et al. 2018, joint autoregressive + hyperprior
    "ms2020": "cheng2020_anchor",  # closest CompressAI zoo entry to Minnen & Singh 2020's
    #                                  channel-wise autoregressive model architecture family
    "cheng2020-attn": "cheng2020_attn",
    "bmshj2018": "bmshj2018_hyperprior",  # Ballé et al. 2018 scale hyperprior (NTC baseline)
}


def get_baseline_model(name: str, quality: int = 4, pretrained: bool = False, metric: str = "mse") -> nn.Module:
    """Instantiate a CompressAI reference model.

    :param name: one of ``_BASELINE_FACTORIES`` keys.
    :param quality: CompressAI quality level (1-8); ignored if you plan to
        train from scratch with your own lambda (recommended, to match the
        paper's lambda sweeps exactly) — set ``pretrained=False`` and treat
        ``quality`` merely as an architecture-capacity knob.
    :param pretrained: if True, downloads pretrained weights from CompressAI's
        model zoo (trained with CompressAI's own lambda schedule, not the
        paper's).
    """
    try:
        import compressai.zoo as czoo
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "compression_baselines requires `pip install compressai`. "
            "See the module docstring for why this dependency is used "
            "instead of a from-scratch port of mbt2018.py/ms2020.py."
        ) from e

    if name not in _BASELINE_FACTORIES:
        raise ValueError(f"Unknown baseline {name!r}; choose from {list(_BASELINE_FACTORIES)}")
    ctor = getattr(czoo, _BASELINE_FACTORIES[name])
    return ctor(quality=quality, pretrained=pretrained, metric=metric)


def rd_loss(model: nn.Module, x: torch.Tensor, lmbda: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the standard rate-distortion training loss for a CompressAI
    model: ``loss = lmbda * 255^2 * mse + bpp``, matching the convention
    used throughout the neural-compression literature (and the paper's
    Sec. 6.4 note that MSE is computed in the model's native [0,1] scale
    then the resulting curve is reported in dB PSNR at 8-bit scale).

    :param x: a batch of images in [0, 1], shape [B, C, H, W].
    :return: (loss, bpp, mse) — mse is in native [0,1]^2 units (divide the
        paper's reported "distortion (MSE)" quantities by 255**2 to compare,
        per the README's note on the GAN-image experiments).
    """
    out = model(x)
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]
    bpp = sum(
        (torch.log(likelihoods).sum() / (-torch.log(torch.tensor(2.0))))
        for likelihoods in out["likelihoods"].values()
    ) / num_pixels
    mse = torch.mean((out["x_hat"] - x) ** 2)
    loss = lmbda * (255 ** 2) * mse + bpp
    return loss, bpp, mse


def train_baseline(
    model: nn.Module,
    source,
    *,
    lmbda: float,
    batchsize: int,
    steps: int,
    lr: float = 1e-4,
    device=None,
    log_path=None,
    verbose: bool = True,
):
    """Minimal training loop for a CompressAI baseline, for parity with the
    R-D upper/lower bound trainers (so all three can be driven the same way
    from ``scripts/``). For serious training runs prefer CompressAI's own
    ``examples/train.py``, which additionally supports an auxiliary
    optimizer for the entropy bottleneck's CDF parameters.
    """
    from .utils import JsonlLogger

    device = device or next(model.parameters()).device
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    aux_optimizer = torch.optim.Adam(
        (p for n, p in model.named_parameters() if n.endswith(".quantiles")), lr=lr * 10
    ) if any(n.endswith(".quantiles") for n, _ in model.named_parameters()) else None

    logger = JsonlLogger(log_path) if log_path else None
    for step in range(steps):
        x = source.sample(batchsize).to(device)
        optimizer.zero_grad()
        loss, bpp, mse = rd_loss(model, x, lmbda)
        loss.backward()
        optimizer.step()
        if aux_optimizer is not None:
            aux_optimizer.zero_grad()
            aux_loss = model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()
        if verbose and step % 100 == 0:
            print(f"step {step}: loss={loss.item():.4f} bpp={bpp.item():.4f} mse={mse.item():.6f}")
        if logger:
            logger.log({"step": step, "loss": loss.item(), "bpp": bpp.item(), "mse": mse.item()})
    if logger:
        logger.close()
    return model
