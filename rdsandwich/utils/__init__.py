"""Shared infrastructure: logging/checkpointing, seeding, numeric helpers, the
base training loop, and the classical Blahut-Arimoto algorithm.

The most-used names are re-exported here so callers can keep writing
``from rdsandwich.utils import JsonlLogger, seed_everything, ...``. The
classical BA algorithm lives in the ``rdsandwich.utils.ba`` submodule.
"""
from .io import (
    JsonlLogger,
    MyJSONEncoder,
    config_dict_to_str,
    get_time_str,
    latest_checkpoint,
    load_checkpoint,
    parse_lamb,
    preprocess_float_dict,
    save_checkpoint,
)
from .torch_utils import (
    SOFTPLUS_INV_1,
    ema_update,
    get_device,
    lower_bound,
    seed_everything,
    upper_bound,
)
from .trainer import BaseTrainer, WarmupReduceLROnPlateau

__all__ = [
    "JsonlLogger", "MyJSONEncoder", "config_dict_to_str", "get_time_str",
    "latest_checkpoint", "load_checkpoint", "parse_lamb", "preprocess_float_dict",
    "save_checkpoint", "SOFTPLUS_INV_1", "ema_update", "get_device", "lower_bound",
    "seed_everything", "upper_bound", "BaseTrainer", "WarmupReduceLROnPlateau",
]
