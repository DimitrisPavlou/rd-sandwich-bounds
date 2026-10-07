"""Every config and template must be accepted by the CLI it targets.

For each YAML file under ``configs/`` (``configs/<source>/`` and
``configs/templates/<source>/``, source = gaussian | physics | images), every expanded run
is rendered to argv exactly as ``scripts/run_sweep.py`` would, and parsed by the
real argument parser of each script it names. This catches renamed or removed
flags, typos, and scripts that no longer exist, without launching anything.
"""
import glob
import os

import pytest

from rdsandwich.cli.parsers import PARSERS
from rdsandwich.sweep import expand_sweep, get_scripts, load_config, params_to_argv, script_params

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
CONFIGS = sorted(glob.glob(os.path.join(CONFIG_DIR, "**", "*.yaml"), recursive=True))
SOURCES = ("gaussian", "physics", "images")


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: os.path.relpath(p, CONFIG_DIR))
def test_config_parses_with_its_scripts(path):
    config = load_config(path)
    scripts = get_scripts(config)
    assert set(scripts) <= set(PARSERS), f"unknown script(s) {scripts}"
    for run in expand_sweep(config):
        for script in scripts:
            argv = params_to_argv(script_params(config, script, run))
            PARSERS[script]().parse_args(argv)  # argparse exits (SystemExit) on any bad flag


# --------------------------------------------------------------------------- #
# Templates: one per (source, model, script), listing every flag that model accepts
# --------------------------------------------------------------------------- #
import re

from rdsandwich.cli import lower_bound, upper_bound

TEMPLATE_DIR = os.path.join(CONFIG_DIR, "templates")
# Argument groups of the upper-bound parsers that belong to one model only.
UB_MODEL_GROUPS = {
    "mlp_vae": ["mlp_vae (vector data)"],
    "resnet_vae": ["resnet_vae / ms2020_vae (images)", "resnet_vae"],
    "ms2020_vae": ["resnet_vae / ms2020_vae (images)", "ms2020_vae"],
    "variable_rate_lossy_vae": ["variable_rate_lossy_vae (variable rate)"],
}
_LB = [lower_bound.build_train_parser, lower_bound.build_eval_parser]
TEMPLATES = {  # <source>/<model>_<train|eval>_<ub|lb>.template.yaml -> (parsers it must cover, model)
    **{f"{src}/mlp_vae_{kind}_ub.template.yaml": (
        [upper_bound.build_train_parser if kind == "train" else upper_bound.build_eval_parser], "mlp_vae")
       for src in ("gaussian", "physics") for kind in ("train", "eval")},
    **{f"images/{m}_{kind}_ub.template.yaml": (
        [upper_bound.build_train_parser if kind == "train" else upper_bound.build_eval_parser], m)
       for m in ("resnet_vae", "ms2020_vae", "variable_rate_lossy_vae") for kind in ("train", "eval")},
    "gaussian/mlp_train_eval_lb.template.yaml": (_LB, None),
    "physics/mlp_train_eval_lb.template.yaml": (_LB, None),
    "images/cnn_train_eval_lb.template.yaml": (_LB, None),
}
_NAME_RE = re.compile(r"^[a-z0-9_]+?_(train|eval|train_eval)_(ub|lb)(_[a-z0-9]+)?(\.template)?\.yaml$")
_KEY_RE = re.compile(r"^\s+#?\s*([a-z][A-Za-z0-9_]*):", re.MULTILINE)  # active or commented-out keys


def _required_flags(build_parser, model):
    """Long flag names a template for ``model`` must list: every flag except other models'."""
    other = {g for m, gs in UB_MODEL_GROUPS.items() if m != model for g in gs}
    mine = set(UB_MODEL_GROUPS.get(model, []))
    flags = set()
    for group in build_parser()._action_groups:
        if group.title in other - mine:
            continue
        for action in group._group_actions:
            flags |= {o[2:] for o in action.option_strings if o.startswith("--") and o != "--help"}
    return flags


def test_every_template_file_is_known():
    found = sorted(os.path.relpath(p, TEMPLATE_DIR)
                   for p in glob.glob(os.path.join(TEMPLATE_DIR, "**", "*.yaml"), recursive=True))
    assert found == sorted(TEMPLATES)


def test_configs_live_in_source_folders_with_model_script_bound_names():
    """configs/<source>/<model>_<train|eval|train_eval>_<ub|lb>[_<variant>].yaml (templates alike)."""
    for path in CONFIGS:
        rel = os.path.relpath(path, CONFIG_DIR).split(os.sep)
        if rel[0] == "templates":
            rel = rel[1:]
        assert len(rel) == 2 and rel[0] in SOURCES, f"{path}: not in configs/[templates/]<source>/"
        assert _NAME_RE.match(rel[1]), f"{rel[1]}: not <model>_<train|eval>_<ub|lb>[_<variant>].yaml"


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_template_lists_every_flag_of_its_model(name):
    with open(os.path.join(TEMPLATE_DIR, name), encoding="utf-8") as f:
        text = f.read()
    listed = set(_KEY_RE.findall(text))
    builders, model = TEMPLATES[name]
    for build in builders:
        missing = _required_flags(build, model) - listed
        assert not missing, f"{name} does not list {sorted(missing)}"


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_template_has_the_three_sections_in_order(name):
    with open(os.path.join(TEMPLATE_DIR, name), encoding="utf-8") as f:
        text = f.read()
    heads = re.findall(r"# ===== (\d)\. (\w+)", text)
    assert [h[0] for h in heads] == ["1", "2", "3"], heads
    assert heads[0][1] == "data" and heads[1][1] == "model"
