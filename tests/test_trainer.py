"""Tests for the BaseTrainer extensions (grad clip, NaN guard, plateau LR, resume)."""
import os

import pytest
import torch
import torch.nn as nn

from rdsandwich.dataloader import build_loader
from rdsandwich.utils import BaseTrainer, WarmupReduceLROnPlateau


class _Src:
    def sample(self, b):
        return torch.randn(b, 4)


class _Trainer(BaseTrainer):
    def train_step(self, x):
        pred = self.model(x)
        loss = (pred ** 2).mean()
        return loss, {"aux": float(loss.item())}


def _make(model, **kw):
    loader = build_loader(_Src(), batch_size=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    return loader, opt


def test_basic_train_and_history():
    m = nn.Linear(4, 4)
    loader, opt = _make(m)
    tr = _Trainer(m, loader, optimizer=opt, epochs=3, steps_per_epoch=4, verbose=False)
    hist = tr.train()
    assert set(hist) >= {"loss", "aux"}
    assert len(hist["loss"]) == 3


def test_grad_clip_and_terminate_on_nan():
    m = nn.Linear(4, 4)
    loader, opt = _make(m)
    tr = _Trainer(m, loader, optimizer=opt, epochs=1, steps_per_epoch=2,
                  grad_clip=1.0, verbose=False)
    tr.train()  # should not raise


def test_periodic_checkpoint_and_resume(tmp_path):
    m = nn.Linear(4, 4)
    loader, opt = _make(m)
    ckpt = os.path.join(tmp_path, "ckpt.pt")
    tr = _Trainer(m, loader, optimizer=opt, epochs=4, steps_per_epoch=2,
                  ckpt_path=ckpt, checkpoint_interval=2, verbose=False)
    tr.train()
    assert os.path.exists(ckpt)

    m2 = nn.Linear(4, 4)
    loader2, opt2 = _make(m2)
    tr2 = _Trainer(m2, loader2, optimizer=opt2, epochs=6, steps_per_epoch=2,
                   ckpt_path=ckpt, resume=True, verbose=False)
    assert tr2.start_epoch == 4


def test_plateau_scheduler_respects_warmup():
    m = nn.Linear(4, 4)
    loader, opt = _make(m)
    sch = WarmupReduceLROnPlateau(opt, factor=0.5, patience=0, warmup=2, min_lr=1e-8)
    tr = _Trainer(m, loader, optimizer=opt, epochs=1, steps_per_epoch=2,
                  scheduler=sch, verbose=False)
    # feed a non-improving metric during warmup -> lr unchanged
    lr0 = opt.param_groups[0]["lr"]
    sch.step(10.0)  # epoch 0 (warmup)
    sch.step(10.0)  # epoch 1 (warmup)
    assert opt.param_groups[0]["lr"] == pytest.approx(lr0)
    sch.step(10.0)  # epoch 2 (active), no improvement -> reduce
    sch.step(10.0)
    assert opt.param_groups[0]["lr"] < lr0


def _poison_grad_trainer(tmp_path, **kw):
    """Trainer whose loss is finite but whose gradient becomes NaN on step 1."""
    m = nn.Linear(4, 4)
    loader, opt = _make(m)

    class T(BaseTrainer):
        n = 0

        def train_step(self, x):
            loss = (self.model(x) ** 2).mean()
            if self.n == 1:  # one-shot hook: only this step's weight grad is NaN
                handle = []

                def poison(g):
                    handle[0].remove()
                    return torch.full_like(g, float("nan"))
                handle.append(self.model.weight.register_hook(poison))
            self.n += 1
            return loss, {}

    tr = T(m, loader, optimizer=opt, epochs=1, steps_per_epoch=4, verbose=False,
           ckpt_path=os.path.join(tmp_path, "ckpt.pt"), **kw)
    return m, tr


def test_nonfinite_grad_aborts_with_report(tmp_path):
    _, tr = _poison_grad_trainer(tmp_path)
    with pytest.raises(FloatingPointError, match="Non-finite gradient"):
        tr.train()
    reports = [f for f in os.listdir(tmp_path) if f.startswith("explosion-") and f.endswith(".json")]
    assert len(reports) == 1
    import json
    with open(os.path.join(tmp_path, reports[0])) as f:
        rep = json.load(f)
    assert rep["kind"] == "non-finite gradients"
    assert [r["param"] for r in rep["nonfinite_grads"]] == ["weight"]
    assert rep["num_nonfinite_params"] == 0          # weights were not updated with NaN
    assert os.path.exists(os.path.join(tmp_path, reports[0][:-5] + ".pt"))


def test_nonfinite_grad_skip_keeps_weights_finite(tmp_path):
    m, tr = _poison_grad_trainer(tmp_path, max_nonfinite_skips=5)
    hist = tr.train()
    assert all(torch.isfinite(p).all() for p in m.parameters())
    assert hist["nonfinite_grad_steps"] == [1]


def test_grad_norm_logging_and_clip_frac():
    m = nn.Linear(4, 4)
    loader, opt = _make(m)
    tr = _Trainer(m, loader, optimizer=opt, epochs=2, steps_per_epoch=3,
                  grad_clip=1e-8, verbose=False)
    hist = tr.train()
    assert all(g > 0 for g in hist["grad_norm_mean"])
    assert hist["clip_frac"] == [1.0, 1.0]          # tiny threshold -> every step clipped
    tr2 = _Trainer(m, loader, optimizer=opt, epochs=1, steps_per_epoch=3,
                   grad_clip=1e12, verbose=False)
    assert tr2.train()["clip_frac"] == [0.0]
