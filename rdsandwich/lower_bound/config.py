"""Hyperparameter config for the R-D lower-bound training loop."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RDLBTrainConfig:
    lamb: float
    batchsize: int = 1024  # k
    num_Ck_samples: int = 1  # M
    last_step: int = 3000
    lr: float = 5e-4
    y_steps: int = 500
    y_lr: float = 1e-2
    y_tol: float = 1e-6
    y_init: str = "quick"
    y_quick_topn: int = 10
    y_sequential: bool = False  # False -> vectorized optimize_y (fast, default); True -> per-candidate loop
    cand_chunk: Optional[int] = None  # vectorized only: chunk the [P, k, *dims] term for image data
    beta: float = 0.2  # EMA fraction retained for log_alpha
    log_E_Ck_max_delta: float = 1.0
    chunksize: Optional[int] = None
    checkpoint_interval: int = 1000
