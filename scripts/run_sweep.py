#!/usr/bin/env python
"""Run a sweep of training / evaluation runs declared in a YAML config.

This replaces GNU-``parallel`` one-liners. For example, the n=1000 Gaussian
upper-bound sweep::

    n=1000; parallel python rdub_mlp.py ... --latent_dim {1} --lambda {2} \\
        ::: 400 500 600 800 ::: 0.3 1 3 10 30 100 300

becomes::

    python scripts/run_sweep.py --config configs/gaussian_ub.yaml

Each combination is expanded from the config's ``sweep`` block (see
``rdsandwich.sweep``) and launched as its own process, the same "one
hyperparameter combo per process" isolation as ``parallel``. When ``script`` is
a list (e.g. ``[train_lb, eval_lb]``), its scripts run in order for each
combination, stopping at the first failure. Use ``--jobs N`` to run up to N
combinations concurrently, and ``--dry-run`` to print the commands instead (they
can be pasted into a bash script or a Slurm array).
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from rdsandwich.sweep import expand_sweep, get_scripts, load_config, params_to_argv, script_params

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPTS = {
    "train_ub": os.path.join(REPO_ROOT, "train", "train_ub.py"),
    "train_lb": os.path.join(REPO_ROOT, "train", "train_lb.py"),
    "eval_ub": os.path.join(REPO_ROOT, "evaluation", "eval_ub.py"),
    "eval_lb": os.path.join(REPO_ROOT, "evaluation", "eval_lb.py"),
}


def build_commands(config, config_path):
    """One list of commands per run (several when ``script`` is a list)."""
    scripts = get_scripts(config)
    for script in scripts:
        if script not in SCRIPTS:
            raise SystemExit(f"config 'script' entries must be in {sorted(SCRIPTS)}, got {script!r} "
                             f"(in {config_path})")
    return [
        [[sys.executable, SCRIPTS[s], *params_to_argv(script_params(config, s, run))] for s in scripts]
        for run in expand_sweep(config)
    ]


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    p.add_argument("--jobs", "-j", type=int, default=1, help="Number of runs to execute concurrently.")
    p.add_argument("--dry-run", action="store_true", help="Print the commands instead of running them.")
    args = p.parse_args()

    config = load_config(args.config)
    runs = build_commands(config, args.config)
    print(f"{len(runs)} run(s) expanded from {args.config} (script={config['script']})")

    if args.dry_run:
        for cmds in runs:
            print("  " + " && ".join(subprocess.list2cmdline(c) for c in cmds))
        return

    def run_one(cmds):
        for cmd in cmds:
            print("START " + subprocess.list2cmdline(cmd))
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                return rc, cmd
        return 0, None

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(run_one, runs))

    failures = [(rc, cmd) for rc, cmd in results if rc != 0]
    print(f"\nDone: {len(runs) - len(failures)}/{len(runs)} runs succeeded.")
    if failures:
        for rc, cmd in failures:
            print(f"  FAILED (exit {rc}): {subprocess.list2cmdline(cmd)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
