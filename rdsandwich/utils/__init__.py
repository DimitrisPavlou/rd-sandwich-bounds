"""Shared infrastructure: logging/checkpointing, seeding, numeric helpers, the
base training loop, and the classical Blahut-Arimoto algorithm.

Logging/checkpointing is in ``io``, seeding and numeric helpers in
``torch_utils``, the base training loop in ``trainer``, epoch-wise LR schedules
in ``lr_schedulers``, and the classical BA algorithm in ``ba``.
"""
