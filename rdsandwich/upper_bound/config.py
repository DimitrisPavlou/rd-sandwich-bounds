"""Hyperparameter config for the R-D upper-bound (beta-VAE) model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class RDUBConfig:
    data_dim: int
    latent_dim: int
    lmbda: float = 0.01
    encoder_units: List[int] = field(default_factory=list)
    decoder_units: List[int] = field(default_factory=list)  # [0] => no decoder (Z == Y)
    encoder_activation: str = "softplus"
    decoder_activation: str = "softplus"
    prior_type: str = "std_gaussian"  # std_gaussian | maf | gmm_<k> | gsm_<k> | lmm_<k> | lsm_<k>
    posterior_type: str = "gaussian"
    ar_hidden_units: List[int] = field(default_factory=lambda: [10, 10])
    maf_stacks: int = 3
    rpd: bool = False  # normalize rate by data_dim (rate "per dimension")
    nats: bool = False  # report rate in nats (else bits)
