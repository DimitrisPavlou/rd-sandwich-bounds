"""jsonl logging, run-naming from a config dict, and checkpoint helpers."""
from __future__ import annotations

import datetime
import json
import os
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch


def get_time_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# --------------------------------------------------------------------------- #
# jsonl logging
# --------------------------------------------------------------------------- #
class MyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist() if obj.numel() > 1 else obj.item()
        return super().default(obj)


def preprocess_float_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Convert any scalar tensors/np types in ``d`` to plain Python floats."""
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.item() if v.numel() == 1 else v.detach().cpu().numpy()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


class JsonlLogger:
    """Append-only jsonl logger. One line per call to ``log()``."""

    def __init__(self, path: str, buffering: int = 1):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._f = open(path, mode="a", buffering=buffering)

    def log(self, record: Dict[str, Any]) -> None:
        record = preprocess_float_dict(record)
        self._f.write(json.dumps(record, cls=MyJSONEncoder) + "\n")

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------- #
# Run naming from a config dict
# --------------------------------------------------------------------------- #
_ABBREVIATIONS = {
    "data_dim": "dd",
    "latent_dim": "ld",
    "lmbda": "lamb",
    "lamb": "lamb",
    "encoder_units": "enc",
    "decoder_units": "dec",
    "prior_type": "prior",
    "posterior_type": "post",
    "maf_stacks": "maf",
    "iaf_stacks": "iaf",
    "batchsize": "k",
    "model": "model",
    "units": "units",
    "seed": "seed",
}


def _fmt_val(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.5g}"
    if isinstance(v, (list, tuple)):
        return "_".join(str(x) for x in v)
    return str(v)


def config_dict_to_str(
    args_dict: Dict[str, Any],
    record_keys: Iterable[str] = (),
    prefix: str = "",
    use_abbr: bool = True,
) -> str:
    """Build a filesystem-friendly run name like ``prefix-dd=16-lamb=100``."""
    parts = [prefix] if prefix else []
    for key in record_keys:
        if key not in args_dict:
            continue
        name = _ABBREVIATIONS.get(key, key) if use_abbr else key
        parts.append(f"{name}={_fmt_val(args_dict[key])}")
    return "-".join(parts)


def parse_lamb(path: str, strip_pardir: bool = True) -> str:
    """Extract the ``lamb=...`` component from a checkpoint path/name."""
    name = os.path.basename(path) if strip_pardir else path
    for token in name.replace("/", "-").split("-"):
        if token.startswith("lamb="):
            return token[len("lamb="):]
    raise ValueError(f"Could not find 'lamb=' in {path!r}")


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str, model: torch.nn.Module, optimizer=None, extra: Optional[dict] = None) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        state["extra"] = extra
    torch.save(state, path)
    return path


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, map_location=None) -> dict:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state.get("extra", {})


def latest_checkpoint(dir_path: str, suffix: str = ".pt") -> Optional[str]:
    """Return the most-recently-modified checkpoint file in ``dir_path``, or None."""
    if not os.path.isdir(dir_path):
        return None
    candidates = [
        os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(suffix)
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)
