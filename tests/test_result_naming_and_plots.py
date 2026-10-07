"""Run / eval-file naming (model keys) and how the plot scripts read them."""
import numpy as np
import pytest

from evaluation import plot_image_ub_training, plot_qr
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


def _write(results_dir, key, lmbda, dataset, bpp, psnr, mse, uint8=True):
    res = dict(bpp=np.array([bpp]), psnr=np.array([psnr]), mse=np.array([mse]))
    if uint8:
        res.update(psnr_uint8=np.array([psnr - 0.1]), mse_uint8=np.array([mse + 0.1]))
    np.savez(results_dir / f"rdub-model={key}-lambda={lmbda}-dataset={dataset}.npz", **res)


def test_plot_qr_discovers_keys_and_keeps_curves_apart(tmp_path):
    vr = "variable_rate_lossy_vae-C=128-L=1_2_3_4_5-lmb=4_2048"
    _write(tmp_path, "resnet_vae", 0.01, "kodak", 0.3, 33.0, 30.0, uint8=False)  # old format
    _write(tmp_path, "resnet_vae-range=pm1", 37.6, "kodak", 0.31, 33.1, 29.0)
    _write(tmp_path, "resnet_vae-range=pm1", 1e-05, "kodak", 0.01, 20.0, 650.0)  # '-' in lambda
    _write(tmp_path, vr, 8, "kodak", 0.1, 30.0, 60.0)
    _write(tmp_path, vr, 512, "kodak", 1.0, 40.0, 6.5)
    _write(tmp_path, vr, 512, "tecnick", 0.9, 41.0, 5.2)
    assert plot_qr.find_keys(tmp_path, "kodak") == sorted(["resnet_vae", "resnet_vae-range=pm1", vr])
    # "resnet_vae" must not pick up the pm1 files
    assert plot_qr.points_from_npz(tmp_path, "resnet_vae", "kodak") == [(0.3, 33.0, 30.0)]
    assert len(plot_qr.points_from_npz(tmp_path, "resnet_vae-range=pm1", "kodak")) == 2
    assert plot_qr.points_from_npz(tmp_path, vr, "kodak") == [(0.1, 30.0, 60.0), (1.0, 40.0, 6.5)]
    rounded = plot_qr.points_from_npz(tmp_path, vr, "kodak", uint8=True)
    assert rounded == [(0.1, pytest.approx(29.9), pytest.approx(60.1)),
                       (1.0, pytest.approx(39.9), pytest.approx(6.6))]
    st = plot_qr.styles(["resnet_vae", "resnet_vae-range=pm1", vr])
    assert st["resnet_vae"] == plot_qr.MODELS["resnet_vae"]
    assert st["resnet_vae-range=pm1"][0].endswith("[range=pm1]")
    assert st["resnet_vae-range=pm1"][1] != st["resnet_vae"][1]   # same base model: new color
    assert st[vr][1] == plot_qr.MODELS["variable_rate_lossy_vae"][1]  # first of its base: own color


def test_plot_qr_cli_writes_figures(tmp_path, monkeypatch):
    _write(tmp_path, "resnet_vae-range=pm1", 37.6, "kodak", 0.31, 33.1, 29.0)
    _write(tmp_path, "variable_rate_lossy_vae-C=8-L=1_1_1_1_1-lmb=4_2048-tiny", 8, "kodak",
           0.1, 30.0, 60.0)
    for extra in ([], ["--rd"], ["--uint8"]):
        out = tmp_path / f"fig{len(extra)}{''.join(extra)}.png"
        monkeypatch.setattr("sys.argv", ["plot_qr", "--results_dir", str(tmp_path),
                                         "--dataset", "kodak", "--out", str(out), *extra])
        plot_qr.main()
        assert out.exists()


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
