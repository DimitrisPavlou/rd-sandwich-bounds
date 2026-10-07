import os

from rdsandwich.sweep import expand_sweep, get_scripts, load_config, params_to_argv, script_params


def test_expand_sweep_cartesian_product():
    config = {
        "fixed": {"dataset": "gaussian", "data_dim": 1000},
        "sweep": {"latent_dim": [400, 500, 600, 800], "lambda": [0.3, 1, 3, 10, 30, 100, 300]},
    }
    runs = expand_sweep(config)
    assert len(runs) == 4 * 7
    # fixed keys are present on every run
    assert all(r["dataset"] == "gaussian" and r["data_dim"] == 1000 for r in runs)
    # every (latent_dim, lambda) combination appears exactly once
    combos = {(r["latent_dim"], r["lambda"]) for r in runs}
    assert len(combos) == 4 * 7
    assert (400, 0.3) in combos and (800, 300) in combos


def test_expand_sweep_no_sweep_section():
    assert expand_sweep({"fixed": {"a": 1}}) == [{"a": 1}]


def test_expand_sweep_scalar_sweep_value():
    runs = expand_sweep({"fixed": {}, "sweep": {"lambda": 10}})
    assert runs == [{"lambda": 10}]


def test_params_to_argv_types():
    argv = params_to_argv({
        "dataset": "gaussian",
        "data_dim": 1000,
        "rpd": True,
        "nats": False,          # omitted
        "device": None,          # omitted
        "units": [100, 100, 1],
        "lambda": 0.3,
    })
    assert argv == [
        "--dataset", "gaussian",
        "--data_dim", "1000",
        "--rpd",
        "--units", "100,100,1",
        "--lambda", "0.3",
    ]


def test_load_and_expand_gaussian_config():
    import math

    path = os.path.join(os.path.dirname(__file__), "..", "configs", "gaussian/mlp_vae_train_ub.yaml")
    config = load_config(path)
    assert config["script"] == "train_ub"
    runs = expand_sweep(config)
    # length == Cartesian product of the sweep list lengths (robust to config edits)
    expected = math.prod(len(v) if isinstance(v, list) else 1 for v in config["sweep"].values())
    assert len(runs) == expected
    # lr must parse as a float (5.0e-4), not the string "5e-4"
    assert isinstance(runs[0]["lr"], float) and abs(runs[0]["lr"] - 5e-4) < 1e-12


def test_script_list_and_script_args():
    config = {
        "script": ["train_lb", "eval_lb"],
        "fixed": {"dataset": "gaussian", "num_Ck_samples": 1},
        "sweep": {"lamb": [1, 10]},
        "script_args": {"train_lb": {"lr": 5e-4}, "eval_lb": {"num_Ck_samples": 5}},
    }
    assert get_scripts(config) == ["train_lb", "eval_lb"]
    assert get_scripts({"script": "train_ub"}) == ["train_ub"]
    run = expand_sweep(config)[0]
    assert script_params(config, "train_lb", run) == {
        "dataset": "gaussian", "num_Ck_samples": 1, "lamb": 1, "lr": 5e-4}
    # script_args override fixed/sweep, for that script only
    assert script_params(config, "eval_lb", run)["num_Ck_samples"] == 5
    assert "lr" not in script_params(config, "eval_lb", run)


def test_run_sweep_index_selects_one_run():
    import pytest
    from scripts.run_sweep import select_runs

    runs = [["a"], ["b"], ["c"]]
    assert select_runs(runs, None) == runs
    assert select_runs(runs, 1) == [["b"]]
    with pytest.raises(SystemExit, match="out of range"):
        select_runs(runs, 3)
