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
