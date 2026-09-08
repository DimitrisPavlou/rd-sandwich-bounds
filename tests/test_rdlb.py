import math

import torch

from rdsandwich.models import make_mlp
from rdsandwich.lower_bound import (
    LowerBoundTrainer,
    RDLBTrainConfig,
    compute_Ck_obj,
    estimate_R_lower_bound,
    optimize_y,
)
from rdsandwich.dataloader import GaussianSource, build_loader


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
