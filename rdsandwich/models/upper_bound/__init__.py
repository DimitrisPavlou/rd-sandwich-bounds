"""R-D upper-bound models (beta-VAEs).

Every model here implements ``get_losses(x) -> (loss, rate, distortion)``,
which is all ``rdsandwich.upper_bound.trainer.UpperBoundTrainer`` needs. The image
models additionally return ``x_hat`` and ``bits`` from ``forward(x)``, used by
the full-image evaluator.
"""
