"""R-D upper bound: config, beta-VAE model, and trainer.

See the paper's Section 3. The model computes a variational upper bound on
R(D); the trainer optimizes it by SGD (a gradient-descent version of the
Blahut-Arimoto algorithm).
"""
from .config import RDUBConfig
from .model import RDUBModel, check_no_decoder
from .trainer import UpperBoundTrainer, evaluate_rdub, lr_lambda_schedule, make_lr_scheduler

__all__ = [
    "RDUBConfig", "RDUBModel", "check_no_decoder",
    "UpperBoundTrainer", "evaluate_rdub", "lr_lambda_schedule", "make_lr_scheduler",
]
