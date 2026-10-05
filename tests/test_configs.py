"""Every config and template must be accepted by the CLI it targets.

For each YAML file in ``configs/`` and ``configs/templates/``, every expanded run
is rendered to argv exactly as ``scripts/run_sweep.py`` would, and parsed by the
real argument parser of each script it names. This catches renamed or removed
flags, typos, and scripts that no longer exist, without launching anything.
"""
import glob
import os

import pytest

from rdsandwich.cli import PARSERS
from rdsandwich.sweep import expand_sweep, get_scripts, load_config, params_to_argv, script_params

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
CONFIGS = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.yaml"))
                 + glob.glob(os.path.join(CONFIG_DIR, "templates", "*.yaml")))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: os.path.relpath(p, CONFIG_DIR))
def test_config_parses_with_its_scripts(path):
    config = load_config(path)
    scripts = get_scripts(config)
    assert set(scripts) <= set(PARSERS), f"unknown script(s) {scripts}"
    for run in expand_sweep(config):
        for script in scripts:
            argv = params_to_argv(script_params(config, script, run))
            PARSERS[script]().parse_args(argv)  # argparse exits (SystemExit) on any bad flag
