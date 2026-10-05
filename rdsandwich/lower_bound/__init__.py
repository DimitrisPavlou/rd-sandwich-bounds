"""R-D lower bound: config, the C_k algorithm, trainer, and evaluator.

See the paper's Section 4 / Algorithm 1. Trains a ``log u`` network (from
``rdsandwich.models.lower_bound``) by maximizing an unconstrained relaxation of
Csiszár's dual characterization of R(D). Restricted to a squared-error
distortion (the inner global maximization becomes Gaussian-mixture mode-finding).
"""
from .config import RDLBTrainConfig
from .algorithm import batch_mse, compute_Ck_obj, optimize_y, optimize_y_vectorized, run_optimize_y
from .trainer import LowerBoundTrainer
from .evaluate import estimate_R_lower_bound

__all__ = [
    "RDLBTrainConfig",
    "batch_mse", "compute_Ck_obj", "optimize_y", "optimize_y_vectorized", "run_optimize_y",
    "LowerBoundTrainer", "estimate_R_lower_bound",
]
