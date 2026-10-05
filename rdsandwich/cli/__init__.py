"""Shared pieces of the command-line scripts in ``train/`` and ``evaluation/``.

``PARSERS`` maps each sweep-runnable script name to its argument parser, so
configs can be validated without launching anything.
"""
from . import lower_bound, upper_bound

PARSERS = {
    "train_ub": upper_bound.build_train_parser,
    "eval_ub": upper_bound.build_eval_parser,
    "train_lb": lower_bound.build_train_parser,
    "eval_lb": lower_bound.build_eval_parser,
}

__all__ = ["PARSERS", "lower_bound", "upper_bound"]
