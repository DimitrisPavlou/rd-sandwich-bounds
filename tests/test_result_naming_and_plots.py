"""Run naming (model keys) and how the training-log plot script reads it."""
import pytest

from evaluation import plot_image_ub_training
from rdsandwich.cli.upper_bound import build_eval_parser, get_runname, model_key

RESNET = ["--latent_channels", "4,8"]


def _args(*argv):
    return build_eval_parser().parse_args(["--dataset", "data/kodak", *argv])


def test_model_keys_and_run_names():
    a = _args("--model", "resnet_vae", "--lambda", "0.01", *RESNET)
    assert model_key(a) == "resnet_vae"
    assert get_runname(a).startswith("rdub-model=resnet_vae-lambda=0.01-")      # unchanged
    a = _args("--model", "resnet_vae", "--image_range", "pm1", "--lambda", "37.6", *RESNET)
    assert model_key(a) == "resnet_vae-range=pm1"
    assert get_runname(a).startswith("rdub-model=resnet_vae-range=pm1-lambda=37.6-")
    a = _args("--model", "variable_rate_lossy_vae", "--lambda", "64")
    assert model_key(a) == "variable_rate_lossy_vae-C=128-L=1_2_3_4_5-lmb=4_2048"
    assert get_runname(a) == "rdub-model=variable_rate_lossy_vae-C=128-L=1_2_3_4_5-lmb=4_2048"
    a = _args("--model", "variable_rate_lossy_vae", "--preset", "c64_l5", "--no_smooth")
    assert model_key(a) == "variable_rate_lossy_vae-C=64-L=1_1_1_1_1-lmb=4_2048-nosmooth"
    assert _args("--model", "ms2020_vae").model == model_key(_args("--model", "ms2020_vae"))


@pytest.mark.parametrize("name, model, lmbda", [
    ("rdub-model=resnet_vae-lambda=0.01-F=256-C=4_8", "resnet_vae", "0.01"),
    ("rdub-model=resnet_vae-range=pm1-lambda=37.6-F=256-C=4_8", "resnet_vae-range=pm1", "37.6"),
    ("rdub-model=ms2020_vae-lambda=1e-05-ld=320", "ms2020_vae", "1e-05"),
])
def test_training_plot_parses_fixed_lambda_runs(name, model, lmbda):
    m = plot_image_ub_training._RUN_RE.match(name)
    assert m and m["model"] == model and m["lmbda"] == lmbda


def test_training_plot_skips_variable_rate_runs():
    name = "rdub-model=variable_rate_lossy_vae-C=128-L=1_2_3_4_5-lmb=4_2048"
    assert plot_image_ub_training._RUN_RE.match(name) is None
