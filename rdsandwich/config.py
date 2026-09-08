"""YAML experiment configs and sweep expansion.

Replaces the GNU-``parallel`` one-liners in ``experiments/*.sh`` with a
declarative YAML file. A config has three top-level keys::

    script: train_rdub          # which scripts/<script>.py to run
    fixed:                      # flags identical across every run
      dataset: gaussian
      data_dim: 1000
      ...
    sweep:                      # flags to vary; the Cartesian product is taken
      latent_dim: [400, 500, 600, 800]
      lambda: [0.3, 1, 3, 10, 30, 100, 300]

``expand_sweep`` turns that into one flat params dict per run (the Cartesian
product of the ``sweep`` lists, merged onto ``fixed``), and ``params_to_argv``
renders a params dict into the ``--flag value`` argv list understood by the
``train/train_*.py`` CLIs. Both are pure functions so they can be unit-tested
without launching any training.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Read a YAML experiment config into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"{path!r} must contain a top-level mapping, got {type(config).__name__}")
    return config


def expand_sweep(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand a config's ``fixed`` + ``sweep`` sections into one params dict per run.

    The ``sweep`` values are lists; every combination (their Cartesian product)
    becomes a run, with the swept values merged on top of ``fixed``. A scalar
    (non-list) ``sweep`` value is treated as a single-element list. With no
    ``sweep`` section, a single run using only ``fixed`` is returned.
    """
    fixed = dict(config.get("fixed") or {})
    sweep = config.get("sweep") or {}
    if not sweep:
        return [fixed]

    keys = list(sweep.keys())
    value_lists = [v if isinstance(v, list) else [v] for v in (sweep[k] for k in keys)]
    runs: List[Dict[str, Any]] = []
    for combo in itertools.product(*value_lists):
        run = dict(fixed)
        run.update(dict(zip(keys, combo)))
        runs.append(run)
    return runs


def params_to_argv(params: Dict[str, Any]) -> List[str]:
    """Render a params dict as an argv list for the ``train/train_*.py`` CLIs.

    Conventions matching those CLIs:
      * ``True``  -> a bare ``--flag`` (argparse ``store_true``); ``False`` / ``None`` are omitted.
      * lists/tuples -> ``--flag a,b,c`` (the CLIs split these on commas).
      * everything else -> ``--flag <str(value)>``.
    """
    argv: List[str] = []
    for key, val in params.items():
        flag = f"--{key}"
        if isinstance(val, bool):
            if val:
                argv.append(flag)
        elif val is None:
            continue
        elif isinstance(val, (list, tuple)):
            argv.extend([flag, ",".join(str(v) for v in val)])
        else:
            argv.extend([flag, str(val)])
    return argv
