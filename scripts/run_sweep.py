#!/usr/bin/env python
"""Run a sweep of training runs declared in a YAML config.

This replaces the GNU-``parallel`` one-liners in the README / ``experiments``.
For example, the n=1000 Gaussian upper-bound sweep::

    n=1000; parallel python rdub_mlp.py ... --latent_dim {1} --lambda {2} \\
        ::: 400 500 600 800 ::: 0.3 1 3 10 30 100 300

becomes::

    python scripts/run_sweep.py --config configs/gaussian_ub.yaml

Each (latent_dim, lambda) combination is expanded from the config's ``sweep``
block (see ``rdsandwich.config``) and launched as its own process — the same
"one hyperparameter combo per process" isolation as ``parallel``. Use
``--jobs N`` to run up to N of them concurrently, and ``--dry-run`` to print
the commands without running anything.
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rdsandwich.config import expand_sweep, load_config, params_to_argv

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
TRAIN_DIR = os.path.join(REPO_ROOT, "train")
# CLIs whose arguments are all top-level flags (no argparse sub-commands), so a
# flat params dict maps cleanly onto argv.
ALLOWED_SCRIPTS = {"train_rdub", "train_rdlb", "train_resnet_vae", "train_image_ub"}


def build_commands(config, config_path):
    script = config.get("script")
    if script not in ALLOWED_SCRIPTS:
        raise SystemExit(
            f"config 'script' must be one of {sorted(ALLOWED_SCRIPTS)}, got {script!r} "
            f"(in {config_path})"
        )
    script_path = os.path.join(TRAIN_DIR, f"{script}.py")
    if not os.path.exists(script_path):
        raise SystemExit(f"script not found: {script_path}")
    return [[sys.executable, script_path, *params_to_argv(run)] for run in expand_sweep(config)]


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    p.add_argument("--jobs", "-j", type=int, default=1, help="Number of runs to execute concurrently.")
    p.add_argument("--dry-run", action="store_true", help="Print the commands instead of running them.")
    args = p.parse_args()

    config = load_config(args.config)
    commands = build_commands(config, args.config)
    print(f"{len(commands)} run(s) expanded from {args.config} (script={config['script']})")

    if args.dry_run:
        for cmd in commands:
            print("  " + subprocess.list2cmdline(cmd))
        return

    def run_one(cmd):
        print("START " + subprocess.list2cmdline(cmd))
        return subprocess.run(cmd).returncode

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        returncodes = list(pool.map(run_one, commands))

    failures = [(rc, cmd) for rc, cmd in zip(returncodes, commands) if rc != 0]
    print(f"\nDone: {len(commands) - len(failures)}/{len(commands)} runs succeeded.")
    if failures:
        for rc, cmd in failures:
            print(f"  FAILED (exit {rc}): {subprocess.list2cmdline(cmd)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
