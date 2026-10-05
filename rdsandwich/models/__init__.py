"""Trainable models, grouped by the bound they estimate.

  * ``rdsandwich.models.upper_bound`` -- beta-VAEs whose ``get_losses(x)``
    returns ``(loss, rate, distortion)``; any such model can be trained by
    ``rdsandwich.upper_bound.UpperBoundTrainer``.
  * ``rdsandwich.models.lower_bound`` -- ``log u`` networks for the lower bound.

The generic building blocks these are made of live in ``rdsandwich.layers``.
"""
