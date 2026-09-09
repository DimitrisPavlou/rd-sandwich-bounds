"""R-D lower bound: config, log-u model factory, the C_k algorithm, and trainer.

See the paper's Section 4 / Algorithm 1. Trains a ``log u`` network by
maximizing an unconstrained relaxation of Csiszár's dual characterization of
R(D). Restricted to a squared-error distortion (the inner global maximization
becomes Gaussian-mixture mode-finding).
"""
from .config import RDLBTrainConfig
from .model import build_log_u_model
from .algorithm import batch_mse, compute_Ck_obj, optimize_y, optimize_y_vectorized
from .trainer import LowerBoundTrainer, estimate_R_lower_bound

__all__ = [
    "RDLBTrainConfig", "build_log_u_model",
    "batch_mse", "compute_Ck_obj", "optimize_y", "optimize_y_vectorized",
    "LowerBoundTrainer", "estimate_R_lower_bound",
]
