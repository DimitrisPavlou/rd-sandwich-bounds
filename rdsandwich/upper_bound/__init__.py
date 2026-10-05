"""R-D upper bound: trainer and evaluators.

See the paper's Section 3. The models (beta-VAEs computing a variational upper
bound on R(D)) live in ``rdsandwich.models.upper_bound``; any model with
``get_losses(x) -> (loss, rate, distortion)`` can be trained here by SGD (a
gradient-descent version of the Blahut-Arimoto algorithm).
"""
from .trainer import UpperBoundTrainer, lr_lambda_schedule, make_lr_scheduler
from .evaluate import evaluate_full_images, evaluate_sampled

__all__ = [
    "UpperBoundTrainer", "lr_lambda_schedule", "make_lr_scheduler",
    "evaluate_full_images", "evaluate_sampled",
]
