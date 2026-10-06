import math

import torch

from rdsandwich.layers.mlp import make_mlp
from rdsandwich.lower_bound.trainer import LowerBoundTrainer
from rdsandwich.lower_bound.config import RDLBTrainConfig
from rdsandwich.lower_bound.algorithm import compute_Ck_obj, optimize_y
from rdsandwich.lower_bound.evaluate import estimate_R_lower_bound
from rdsandwich.data.gaussian import GaussianSource
from rdsandwich.data.base import build_loader


def test_optimize_y_quick_runs():
    torch.manual_seed(0)
    log_u = make_mlp(4, [16, 1], activation="selu")
    x = torch.randn(8, 4)
    res = optimize_y(log_u, x, lamb=1.0, num_steps=5, init="quick", quick_topn=3)
    assert res["opt_y"].shape == (4,)


def test_optimize_y_exhaustive_runs():
    log_u = make_mlp(4, [16, 1], activation="selu")
    x = torch.randn(6, 4)
    res = optimize_y(log_u, x, lamb=1.0, num_steps=5, init="exhaustive")
    assert res["opt_y"].shape == (4,)


def test_compute_ck_obj_finite():
    log_u = make_mlp(4, [16, 1], activation="selu")
    x = torch.randn(8, 4)
    y = torch.randn(4)
    log_ck, log_u_x = compute_Ck_obj(log_u, x, y, lamb=1.0)
    assert torch.isfinite(log_ck)
    assert log_u_x.shape == (8,)


def test_lower_bound_trainer_smoke():
    torch.manual_seed(0)
    log_u = make_mlp(2, [8, 1], activation="selu")
    source = GaussianSource.standard(2)
    cfg = RDLBTrainConfig(lamb=1.0, batchsize=16, num_Ck_samples=1, last_step=3, y_steps=3, y_init="quick", y_quick_topn=4)
    loader = build_loader(source, cfg.batchsize)
    optimizer = torch.optim.Adam(log_u.parameters(), lr=cfg.lr)
    trainer = LowerBoundTrainer(log_u, loader, optimizer=optimizer, cfg=cfg, verbose=False)
    trainer.train()  # should not raise


def test_estimate_R_lower_bound_exhaustive_finite():
    # Regression test for the autograd/no_grad conflict: estimate_R_lower_bound is
    # decorated with @torch.no_grad(), but optimize_y's hill-climb needs grad. The
    # 'exhaustive' y_init is the path the README/eval script prescribe for the
    # final lower-bound number, so exercise it end-to-end here.
    torch.manual_seed(0)
    log_u = make_mlp(2, [8, 1], activation="selu")
    source = GaussianSource.standard(2)
    cfg = RDLBTrainConfig(lamb=1.0, batchsize=16, num_Ck_samples=1, y_steps=3, y_init="exhaustive")
    res = estimate_R_lower_bound(log_u, source, lamb=1.0, cfg=cfg)
    assert math.isfinite(res["R_"])


def test_lower_bound_trainer_on_finite_dataset():
    """The LB loop pulls batches with _next_batch(); it must also work on finite
    (map-style) datasets such as .npy arrays or image folders, not only on
    infinite synthetic sources."""
    from rdsandwich.data.array import ArraySource

    torch.manual_seed(0)
    data = ArraySource(torch.randn(64, 2).numpy())
    model = make_mlp(2, [8, 1])
    cfg = RDLBTrainConfig(lamb=1.0, batchsize=16, num_Ck_samples=1, last_step=6, y_steps=5)
    trainer = LowerBoundTrainer(
        model, build_loader(data, batch_size=cfg.batchsize),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3), cfg=cfg, verbose=False,
    )
    trainer.train()  # 7 steps x 1 batch > 4 batches per pass: wraps around the loader
