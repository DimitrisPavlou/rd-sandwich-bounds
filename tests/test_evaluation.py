"""The YAML-driven evaluation pipeline: load_model, evaluation/evaluate.py, evaluation/plot.py."""
import glob
import json
import os
import re

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from evaluation import evaluate, plot
from rdsandwich.data.base import build_loader
from rdsandwich.models.upper_bound.resnet_vae import ResNetVAE, ResNetVAEConfig
from rdsandwich.models.upper_bound.variable_rate_lossy_vae import VariableRateLossyVAE, VariableRateLossyVAEConfig
from rdsandwich.upper_bound.evaluate import load_model
from rdsandwich.upper_bound.results import load_yaml_config
from rdsandwich.upper_bound.trainer import UpperBoundTrainer
from rdsandwich.utils.io import final_checkpoint, save_checkpoint

VR_RUN = "rdub-model=variable_rate_lossy_vae-C=8-L=1_1_1_1_1-lmb=4_2048-tiny"
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "templates")


class _Images:
    def sample(self, b):
        return torch.rand(b, 3, 64, 64) * 255


def _resnet(lmbda):
    return ResNetVAE(ResNetVAEConfig(latent_channels=[4, 8], num_filters=8, lmbda=lmbda))


def _vr():
    return VariableRateLossyVAE(VariableRateLossyVAEConfig.from_preset("tiny"))


def _train(model, name, run_dir, ckpt_name, epochs=1, **kw):
    """Train for a step per epoch with the real trainer, so the checkpoints are the real format."""
    torch.manual_seed(0)
    trainer = UpperBoundTrainer(
        model, build_loader(_Images(), 2), optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        epochs=epochs, steps_per_epoch=1, ckpt_path=os.path.join(run_dir, ckpt_name), verbose=False,
        checkpoint_metadata={"model": name}, **kw)
    trainer.train()
    return trainer


def _write_images(folder, n=2, size=(48, 80)):
    """Images whose sides are not multiples of the models' strides, to exercise the padding."""
    os.makedirs(folder)
    rng = np.random.default_rng(0)
    for i in range(n):
        Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8)).save(f"{folder}/im{i}.png")
    return str(folder)


def _run(script, monkeypatch, config, *extra):
    monkeypatch.setattr("sys.argv", [script.__name__, "--config", str(config), *extra])
    script.main()


# --------------------------------------------------------------------------- #
# load_model / final_checkpoint
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name, build", [("resnet_vae", lambda: _resnet(64.0)), ("variable_rate_lossy_vae", _vr)])
def test_load_model_rebuilds_the_trained_model_from_the_checkpoint_alone(tmp_path, name, build):
    trainer = _train(build(), name, tmp_path, "ckpt.pt", ema_decay=0.5, ema_warmup=1)
    model, meta = load_model(os.path.join(tmp_path, "ckpt.pt"))
    assert meta == {"model": name, "epochs": 1, "weights": "ema",
                    "num_params": sum(p.numel() for p in trainer.model.parameters())}
    assert type(model) is type(trainer.model) and vars(model.cfg) == vars(trainer.model.cfg)
    assert not model.training
    for k, v in trainer.ema_model.state_dict().items():
        assert torch.equal(model.state_dict()[k], v)
    raw, meta = load_model(os.path.join(tmp_path, "ckpt.pt"), use_ema=False)
    assert meta["weights"] == "raw"
    for k, v in trainer.model.state_dict().items():
        assert torch.equal(raw.state_dict()[k], v)


def test_load_model_fills_legacy_fields_and_reads_the_model_from_the_run_name(tmp_path):
    """Old checkpoints: no extra["model"], no scale_min (trained without a floor), a removed field."""
    model = _vr()
    cfg = {k: v for k, v in vars(model.cfg).items() if k != "scale_min"}
    path = os.path.join(tmp_path, VR_RUN, "ckpt.pt")
    save_checkpoint(path, model, extra={"cfg": {**cfg, "removed_field": 1}, "epoch": 9})
    with pytest.warns(UserWarning, match="removed_field"):
        loaded, meta = load_model(path)
    assert meta["model"] == "variable_rate_lossy_vae" and meta["epochs"] == 10 and meta["weights"] == "raw"
    assert loaded.cfg.scale_min == 0.0


def test_final_checkpoint_goes_by_epoch_not_file_time(tmp_path):
    _train(_resnet(8.0), "resnet_vae", tmp_path, "ckpt-lambda=8.pt", epochs=3, checkpoint_interval=1)
    os.utime(os.path.join(tmp_path, "ckpt-lambda=8-epoch=0001.pt"))   # newest file, least trained
    # epoch 3 is in both the final file and ckpt-lambda=8-epoch=0003.pt; on a tie the final one wins
    assert final_checkpoint(str(tmp_path)) == os.path.join(tmp_path, "ckpt-lambda=8.pt")
    save_checkpoint(os.path.join(tmp_path, "other.pt"), _resnet(8.0))
    with pytest.raises(ValueError, match="one checkpoint series"):
        final_checkpoint(str(tmp_path))


# --------------------------------------------------------------------------- #
# evaluation/evaluate.py
# --------------------------------------------------------------------------- #
def test_lambda_rules():
    vr, fixed = _vr(), _resnet(64.0)
    assert evaluate.lambdas_for(vr, "c.pt", [8.0, 64.0]) == [8.0, 64.0]
    assert evaluate.lambdas_for(fixed, "c.pt", None) == [64.0]
    with pytest.raises(SystemExit, match="set `lambdas`"):
        evaluate.lambdas_for(vr, "c.pt", None)
    with pytest.raises(SystemExit, match="outside"):
        evaluate.lambdas_for(vr, "c.pt", [2.0, 64.0])
    with pytest.raises(SystemExit, match="remove `lambdas`"):
        evaluate.lambdas_for(fixed, "c.pt", [8.0])


def test_a_curve_takes_one_checkpoint_per_run(tmp_path):
    _train(_resnet(8.0), "resnet_vae", tmp_path, "ckpt-lambda=8.pt", epochs=2, checkpoint_interval=1)
    with pytest.raises(SystemExit, match="same run"):
        evaluate.resolve_checkpoints([os.path.join(tmp_path, "ckpt-lambda=*.pt")])
    assert evaluate.resolve_checkpoints([str(tmp_path)]) == [final_checkpoint(str(tmp_path))]
    with pytest.raises(SystemExit, match="matches nothing"):
        evaluate.resolve_checkpoints([os.path.join(tmp_path, "nothing*")])


def test_evaluate_then_plot(tmp_path, monkeypatch):
    for lmbda in (8, 64):
        _train(_resnet(float(lmbda)), "resnet_vae", tmp_path / "ckpts" / f"rdub-model=resnet_vae-lambda={lmbda}",
               f"ckpt-lambda={lmbda}.pt")
    _train(_vr(), "variable_rate_lossy_vae", tmp_path / "ckpts" / VR_RUN, "ckpt.pt")
    datasets = {"kodak": _write_images(tmp_path / "kodak"), "tecnick": _write_images(tmp_path / "tecnick", n=1)}
    results = str(tmp_path / "results")

    def eval_config(name, **kw):
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(dict(name=name, datasets=datasets, results_dir=results, device="cpu", **kw)))
        return path

    fixed = eval_config("resnet", checkpoints=[str(tmp_path / "ckpts" / "rdub-model=resnet_vae-lambda=*")])
    vr = eval_config("vr", label="VR", checkpoints=[str(tmp_path / "ckpts" / VR_RUN)], lambdas=[8, 64])
    vr_one = eval_config("vr_one", checkpoints=[str(tmp_path / "ckpts" / VR_RUN)], lambdas=[64])
    for config in (fixed, vr, vr_one):
        _run(evaluate, monkeypatch, config)

    res = json.loads((tmp_path / "results" / "resnet_kodak.json").read_text())
    assert res["label"] == "resnet" and res["images"] == ["im0.png", "im1.png"]
    assert [pt["lmbda"] for pt in res["points"]] == [8.0, 64.0]
    assert all(pt["model"] == "resnet_vae" and len(pt["per_image"]["bpp"]) == 2 for pt in res["points"])
    assert res["points"][0]["checkpoint"].endswith("ckpt-lambda=8.pt")
    vr_res = json.loads((tmp_path / "results" / "vr_kodak.json").read_text())
    assert vr_res["label"] == "VR" and [pt["lmbda"] for pt in vr_res["points"]] == [8.0, 64.0]
    # the seed is reset per point: lambda=64 is the same alone as after lambda=8
    one = json.loads((tmp_path / "results" / "vr_one_kodak.json").read_text())["points"][0]
    assert one["per_image"] == vr_res["points"][1]["per_image"]
    assert len(json.loads((tmp_path / "results" / "vr_tecnick.json").read_text())["points"][0]["per_image"]["bpp"]) == 1

    # existing results are skipped, --force redoes them
    mtime = os.path.getmtime(tmp_path / "results" / "vr_kodak.json")
    _run(evaluate, monkeypatch, vr)
    assert os.path.getmtime(tmp_path / "results" / "vr_kodak.json") == mtime

    plot_config = tmp_path / "plot.yaml"
    plot_config.write_text(yaml.safe_dump(dict(
        out_dir=str(tmp_path / "plots"), curves=["resnet", "vr", "missing"], datasets=["kodak", "tecnick"],
        results_dir=results, distortion="uint8", labels={"resnet": "ResNet-VAE"})))
    _run(plot, monkeypatch, plot_config)
    n = res["points"][0]["num_params"]
    assert n == sum(p.numel() for p in _resnet(8.0).parameters())
    assert plot.load_curves(load_yaml_config(plot.PlotConfig, str(plot_config)), "kodak")[0]["label"] == \
        f"ResNet-VAE, {n / 1e6:.1f}M params"
    assert plot.param_count([{"num_params": 2e6}, {"num_params": 18.6e6}]) == "2.0-18.6M params"
    assert plot.param_count([{}]) == ""
    assert sorted(os.listdir(tmp_path / "plots")) == [
        "combined.png", "qr_kodak.png", "qr_tecnick.png", "rd_kodak.png", "rd_tecnick.png"]


def test_bad_configs_exit_with_a_message(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("name: x\ncheckpoints: []\ndatasets: {}\nlambda: [1]\n")   # typo: lambda
    with pytest.raises(SystemExit, match="lambda"):
        load_yaml_config(evaluate.EvalConfig, str(path))
    path.write_text("out_dir: p\ncurves: [a]\ndatasets: [kodak]\nkinds: [qr, dr]\n")
    with pytest.raises(SystemExit, match="kinds"):
        load_yaml_config(plot.PlotConfig, str(path))


# --------------------------------------------------------------------------- #
# Templates: they load, and list every key
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pattern, cls", [("eval/*.yaml", evaluate.EvalConfig), ("plots/*.yaml", plot.PlotConfig)])
def test_templates_load_and_list_every_key(pattern, cls):
    paths = glob.glob(os.path.join(TEMPLATE_DIR, pattern))
    assert paths
    for path in paths:
        load_yaml_config(cls, path)
        with open(path, encoding="utf-8") as f:
            listed = set(re.findall(r"^#?\s*([a-z][a-z0-9_]*):", f.read(), re.MULTILINE))
        missing = {field for field in cls.__dataclass_fields__} - listed
        assert not missing, f"{path} does not list {sorted(missing)}"
