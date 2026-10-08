"""jsonl logging, run-naming from a config dict, and checkpoint helpers."""
from __future__ import annotations

import datetime
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    """Save atomically: write ``<path>.tmp`` and rename it over ``path``, so a job
    killed mid-save never leaves a truncated checkpoint behind."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        state["extra"] = extra
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, map_location=None,
                    use_ema: bool = False) -> dict:
    """Load weights (and optimizer state) from ``path``; returns the ``extra`` payload.

    ``use_ema=True`` loads the exponential-moving-average weights instead of the raw
    ones when the checkpoint has them (``extra["ema_state_dict"]``), as for evaluation.
    """
    state = torch.load(path, map_location=map_location)
    extra = state.get("extra", {})
    if use_ema and "ema_state_dict" in extra:
        model.load_state_dict(extra["ema_state_dict"])
    else:
        model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return extra


def latest_checkpoint(dir_path: str, suffix: str = ".pt") -> Optional[str]:
    """Return the most-recently-modified checkpoint file in ``dir_path``, or None.

    ``explosion-*`` dumps written by ``BaseTrainer`` on a non-finite loss are not
    checkpoints (no optimizer state, pre-explosion weights) and are skipped.
    """
    if not os.path.isdir(dir_path):
        return None
    candidates = [
        os.path.join(dir_path, f) for f in os.listdir(dir_path)
        if f.endswith(suffix) and not f.startswith("explosion-")
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


# Periodic checkpoints sit next to the final one (``ckpt-lambda=64.pt``) as
# ``ckpt-lambda=64-epoch=0042.pt``, where the number counts completed epochs.
def epoch_checkpoint_path(path: str, epochs_done: int) -> str:
    """``<dir>/<stem>.pt`` -> ``<dir>/<stem>-epoch=NNNN.pt`` (``epochs_done`` = NNNN)."""
    stem, ext = os.path.splitext(path)
    return f"{stem}-epoch={epochs_done:04d}{ext}"


def epoch_checkpoints(path: str) -> List[Tuple[int, str]]:
    """The periodic checkpoints of the run whose final checkpoint is ``path``, as
    ``(epochs_done, file)`` sorted by epoch. Other files in the directory (other
    lambdas, explosion dumps, ``.tmp`` leftovers) are ignored."""
    dir_path = os.path.dirname(path) or "."
    if not os.path.isdir(dir_path):
        return []
    stem, ext = os.path.splitext(os.path.basename(path))
    pattern = re.compile(re.escape(stem) + r"-epoch=(\d+)" + re.escape(ext))
    found = []
    for f in os.listdir(dir_path):
        m = pattern.fullmatch(f)
        if m:
            found.append((int(m.group(1)), os.path.join(dir_path, f)))
    return sorted(found)


def resume_candidates(path: str) -> List[Tuple[int, str]]:
    """Checkpoints of the run whose final checkpoint is ``path``, most-trained first,
    as ``(epochs_done, file)``. Ordered by epoch rather than modification time: a final
    checkpoint from an earlier, shorter run can be older than a newer periodic one.
    The final file has no epoch in its name, so it is read (memory-mapped, without
    loading the weights) from its ``extra["epoch"]``; on a tie it comes first."""
    candidates = [(n, 0, p) for n, p in epoch_checkpoints(path)]
    if os.path.isfile(path):
        try:
            state = torch.load(path, map_location="cpu", mmap=True)
            candidates.append((int(state.get("extra", {}).get("epoch", -1)) + 1, 1, path))
        except Exception as e:  # noqa: BLE001 - an unreadable file is just not a candidate
            print(f"Warning: cannot read {path} ({e!r}); not resuming from it.")
    return [(n, p) for n, _, p in sorted(candidates, reverse=True)]

