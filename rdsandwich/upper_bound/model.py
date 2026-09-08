"""R-D UPPER BOUND model (Section 3 of the paper).

A beta-VAE whose likelihood is induced by the squared-error distortion; per
Corollary A.3.1 the resulting (D, R) point lies above the true R(D) curve for
every lambda. Ported from ``rdub_mlp.py``.

Supported priors ``Q_Z``: 'std_gaussian', 'maf' (see ``rdsandwich.models.flows``),
and factorized Gaussian/logistic mixtures ('gmm_<k>' / 'gsm_<k>' / 'lmm_<k>' /
'lsm_<k>'). Supported posterior ``Q_{Z|X}``: 'gaussian'.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models import MAF, make_mlp
from ..utils import SOFTPLUS_INV_1
from .config import RDUBConfig


def check_no_decoder(decoder_units: Sequence[int]) -> bool:
    """Convention: decoder_units == [0] (or []) with an explicit flag means
    'no decoder network', i.e. Z == Y (identity decoder)."""
    return len(decoder_units) == 1 and decoder_units[0] <= 0


class RDUBModel(nn.Module):
    """Beta-VAE computing a variational upper bound on R(D)."""

    def __init__(self, cfg: RDUBConfig):
        super().__init__()
        self.cfg = cfg
        latent_dim = cfg.latent_dim

        if cfg.posterior_type != "gaussian":
            raise NotImplementedError(
                f"posterior_type={cfg.posterior_type!r} not ported; use 'gaussian' "
                "(see rdsandwich.compression_baselines for NTC-style uniform posteriors)."
            )
        encoder_out_dim = 2 * latent_dim  # loc, raw_scale
        self.encoder = make_mlp(
            cfg.data_dim, list(cfg.encoder_units) + [encoder_out_dim], activation=cfg.encoder_activation
        )

        self.has_decoder = not check_no_decoder(cfg.decoder_units)
        if self.has_decoder:
            self.decoder = make_mlp(latent_dim, list(cfg.decoder_units) + [cfg.data_dim], activation=cfg.decoder_activation)
        else:
            assert cfg.data_dim == latent_dim, "Z==Y requires latent_dim == data_dim"
            self.decoder = None

        self._build_prior()

    # ----------------------------------------------------------------- #
    def _build_prior(self):
        cfg = self.cfg
        pt = cfg.prior_type
        self.prior_kind = pt
        if pt == "std_gaussian":
            pass  # no parameters; N(0, I)
        elif pt == "maf":
            self.maf = MAF(cfg.latent_dim, n_stacks=cfg.maf_stacks, hidden_units=cfg.ar_hidden_units)
        elif pt[:4] in ("gsm_", "gmm_", "lsm_", "lmm_"):
            k = int(pt[4:])
            self.mix_logits = nn.Parameter(torch.randn(cfg.latent_dim, k))
            self.mix_log_scale = nn.Parameter(torch.randn(cfg.latent_dim, k) + 2.0)
            self.is_scale_mixture = "s" in pt
            self.is_gaussian_mixture = pt.startswith("g")
            if not self.is_scale_mixture:
                self.mix_loc = nn.Parameter(torch.randn(cfg.latent_dim, k))
            else:
                self.register_buffer("mix_loc", torch.zeros(cfg.latent_dim, k))
        else:
            raise ValueError(f"Unknown prior_type: {pt!r}")

    def prior_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """log q(z), summed over the latent dimension unless prior_type=='maf'
        (which already returns a joint density)."""
        pt = self.prior_kind
        if pt == "std_gaussian":
            return (-0.5 * (z ** 2 + math.log(2 * math.pi))).sum(-1)
        if pt == "maf":
            return self.maf.log_prob(z)
        # mixture priors: fully factorized across latent_dim, each dim a k-component mixture
        loc = self.mix_loc  # [D, k]
        scale = F.softplus(self.mix_log_scale)  # [D, k]
        log_w = F.log_softmax(self.mix_logits, dim=-1)  # [D, k]
        zz = z.unsqueeze(-1)  # [B, D, 1]
        if self.is_gaussian_mixture:
            log_comp = (-0.5 * (((zz - loc) / scale) ** 2) - torch.log(scale) - 0.5 * math.log(2 * math.pi))
        else:  # logistic mixture
            std_term = (zz - loc) / scale
            log_comp = -std_term - torch.log(scale) - 2 * F.softplus(-std_term)
        log_mix = torch.logsumexp(log_w + log_comp, dim=-1)  # [B, D]
        return log_mix.sum(-1)

    # ----------------------------------------------------------------- #
    def encode_decode(self, x: torch.Tensor):
        enc = self.encoder(x)
        latent_dim = self.cfg.latent_dim
        loc = enc[..., :latent_dim]
        scale = F.softplus(enc[..., latent_dim:] + SOFTPLUS_INV_1)
        eps = torch.randn_like(loc)
        z = loc + scale * eps
        log_q = (-0.5 * ((eps) ** 2 + math.log(2 * math.pi)) - torch.log(scale)).sum(-1)
        log_prior = self.prior_log_prob(z)
        kl = log_q - log_prior  # per-sample rate term (nats), Eq. 9 in the appendix
        y = self.decoder(z) if self.has_decoder else z
        return z, y, kl

    def get_losses(self, x: torch.Tensor):
        _, y, kl = self.encode_decode(x)
        mse = F.mse_loss(y, x)
        rate = kl if self.cfg.nats else kl / math.log(2.0)
        rate = rate.mean()
        if self.cfg.rpd:
            rate = rate / self.cfg.data_dim
        loss = rate + self.cfg.lmbda * mse
        return loss, rate, mse

    def forward(self, x):
        return self.get_losses(x)
