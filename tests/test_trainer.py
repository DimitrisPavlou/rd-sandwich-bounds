"""Tests for the BaseTrainer extensions (grad clip, NaN guard, plateau LR, resume, mixed precision)."""
import os

import pytest
import torch
import torch.nn as nn

from rdsandwich.data.base import build_loader
from rdsandwich.utils.lr_schedulers import WarmupReduceLROnPlateau
from rdsandwich.utils.trainer import BaseTrainer


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
    # One file per periodic save, plus the final checkpoint.
    assert sorted(os.listdir(tmp_path)) == ["ckpt-epoch=0002.pt", "ckpt-epoch=0004.pt", "ckpt.pt"]
    e2 = torch.load(os.path.join(tmp_path, "ckpt-epoch=0002.pt"), weights_only=False)
    assert e2["extra"]["epoch"] == 1 and len(e2["extra"]["history"]["loss"]) == 2

    m2 = nn.Linear(4, 4)
    loader2, opt2 = _make(m2)
    tr2 = _Trainer(m2, loader2, optimizer=opt2, epochs=6, steps_per_epoch=2,
                   ckpt_path=ckpt, resume=True, verbose=False)
    assert tr2.start_epoch == 4
    assert torch.equal(m2.weight, m.weight)
    hist = tr2.train()
    assert len(hist["loss"]) == 6                       # history carried across the resume
    assert torch.load(ckpt, weights_only=False)["extra"]["epoch"] == 5


def _ckpt_trainer(tmp_path, epochs, **kw):
    m = nn.Linear(4, 4)
    loader, opt = _make(m)
    return _Trainer(m, loader, optimizer=opt, epochs=epochs, steps_per_epoch=2,
                    ckpt_path=os.path.join(tmp_path, "ckpt-lambda=64.pt"), verbose=False, **kw)


def test_resume_picks_highest_epoch_not_newest_file(tmp_path):
    _ckpt_trainer(tmp_path, 3, checkpoint_interval=1).train()     # epoch 1..3 + final (epoch 3)
    # Then a longer run is resumed and killed after its epoch-5 save: the final file is
    # from epoch 3, and we make it the most recently modified file.
    tr = _ckpt_trainer(tmp_path, 5, checkpoint_interval=1, resume=True)
    for epoch in (3, 4):
        tr._save(epoch, {}, os.path.join(tmp_path, f"ckpt-lambda=64-epoch={epoch + 1:04d}.pt"))
    os.utime(os.path.join(tmp_path, "ckpt-lambda=64.pt"))
    # Other runs' files and explosion dumps in the same directory are not candidates.
    tr._save(98, {}, os.path.join(tmp_path, "ckpt-lambda=6-epoch=0099.pt"))
    tr._save(98, {}, os.path.join(tmp_path, "ckpt-lambda=64-epoch=0099.pt.tmp"))
    tr._save(98, {}, os.path.join(tmp_path, "explosion-e98-s0-2026.pt"))
    assert _ckpt_trainer(tmp_path, 10, resume=True).start_epoch == 5


def test_resume_falls_back_past_a_corrupt_checkpoint(tmp_path):
    _ckpt_trainer(tmp_path, 4, checkpoint_interval=2).train()
    os.remove(os.path.join(tmp_path, "ckpt-lambda=64.pt"))
    (tmp_path / "ckpt-lambda=64-epoch=0004.pt").write_bytes(b"truncated")
    assert _ckpt_trainer(tmp_path, 6, resume=True).start_epoch == 2
    (tmp_path / "ckpt-lambda=64-epoch=0002.pt").write_bytes(b"truncated")
    with pytest.raises(RuntimeError, match="none of the checkpoints"):
        _ckpt_trainer(tmp_path, 6, resume=True)


def test_resume_without_checkpoints_starts_fresh(tmp_path):
    assert _ckpt_trainer(tmp_path, 2, resume=True).start_epoch == 0


def test_every_periodic_checkpoint_is_kept(tmp_path):
    _ckpt_trainer(tmp_path, 6, checkpoint_interval=2).train()
    tr = _ckpt_trainer(tmp_path, 10, checkpoint_interval=2, resume=True)
    tr.train()
    assert sorted(os.listdir(tmp_path)) == [f"ckpt-lambda=64-epoch={n:04d}.pt" for n in (2, 4, 6, 8, 10)] \
        + ["ckpt-lambda=64.pt"]


def test_resume_past_epochs_keeps_the_loaded_epoch(tmp_path):
    _ckpt_trainer(tmp_path, 4, checkpoint_interval=2).train()
    tr = _ckpt_trainer(tmp_path, 2, resume=True)               # asks for fewer epochs than done
    tr.train()                                                 # runs nothing
    final = torch.load(os.path.join(tmp_path, "ckpt-lambda=64.pt"), weights_only=False)
    assert final["extra"]["epoch"] == 3


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


def test_latest_checkpoint_skips_explosion_dumps(tmp_path):
    import time

    from rdsandwich.utils.io import latest_checkpoint

    (tmp_path / "ckpt-lambda=0.01.pt").write_bytes(b"x")
    time.sleep(0.01)
    (tmp_path / "explosion-e3-s7-2026.pt").write_bytes(b"x")  # newer, but not a checkpoint
    assert latest_checkpoint(str(tmp_path)).endswith("ckpt-lambda=0.01.pt")


def test_save_checkpoint_is_atomic(tmp_path):
    from rdsandwich.utils.io import save_checkpoint

    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, nn.Linear(2, 2))
    assert os.listdir(tmp_path) == ["ckpt.pt"]                 # no .tmp left behind


# --------------------------------------------------------------------------- #
# Mixed precision (--amp bf16 | fp16)
# --------------------------------------------------------------------------- #
_UB_ARGS = ["--dataset", "gaussian", "--model", "mlp_vae"]

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="autocast is CUDA-only")
requires_bf16 = pytest.mark.skipif(not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
                                   reason="needs a bf16-capable GPU")


@pytest.mark.parametrize("flags, amp", [([], None), (["--amp"], "bf16"),
                                        (["--amp", "bf16"], "bf16"), (["--amp", "fp16"], "fp16")])
def test_amp_flag(flags, amp):
    from rdsandwich.cli.upper_bound import build_train_parser

    assert build_train_parser().parse_args(_UB_ARGS + flags).amp == amp


def test_amp_flag_rejects_unknown_dtype_and_keeps_the_next_flag():
    from rdsandwich.cli.upper_bound import build_train_parser

    parser = build_train_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(_UB_ARGS + ["--amp", "fp32"])
    args = parser.parse_args(_UB_ARGS + ["--amp", "--compile"])
    assert args.amp == "bf16" and args.compile


def test_amp_is_validated_and_cuda_only():
    m = nn.Linear(4, 4)
    loader, opt = _make(m)
    with pytest.raises(ValueError, match="amp must be"):
        _Trainer(m, loader, optimizer=opt, epochs=1, steps_per_epoch=1, amp="fp32", verbose=False)
    tr = _Trainer(m, loader, optimizer=opt, epochs=1, steps_per_epoch=1, amp="bf16", verbose=False)
    assert tr.amp is None and not tr.grad_scaling              # on CPU: plain fp32


def test_resume_warns_on_a_precision_change(tmp_path, capsys):
    _ckpt_trainer(tmp_path, 2).train()
    path = os.path.join(tmp_path, "ckpt-lambda=64.pt")
    assert torch.load(path, weights_only=False)["extra"]["amp"] is None
    _ckpt_trainer(tmp_path, 3, resume=True)
    assert "Warning" not in capsys.readouterr().out            # fp32 -> fp32
    # A checkpoint from before "amp" was recorded: only its scaler state says fp16.
    state = torch.load(path, weights_only=False)
    del state["extra"]["amp"]
    state["extra"]["scaler"] = {"scale": 65536.0}
    torch.save(state, path)
    _ckpt_trainer(tmp_path, 3, resume=True)
    assert "trained with fp16; continuing in fp32" in capsys.readouterr().out


@requires_bf16
def test_bf16_has_no_scaler_and_never_applies_a_nonfinite_grad(tmp_path):
    m, tr = _poison_grad_trainer(tmp_path, amp="bf16", device=torch.device("cuda"))
    assert tr.amp == "bf16" and not tr.scaler.is_enabled()
    with pytest.raises(FloatingPointError, match="Non-finite gradient"):
        tr.train()
    assert all(torch.isfinite(p).all() for p in m.parameters())


@requires_bf16
def test_bf16_nonfinite_grad_skip_keeps_weights_finite(tmp_path):
    m, tr = _poison_grad_trainer(tmp_path, amp="bf16", device=torch.device("cuda"), max_nonfinite_skips=5)
    hist = tr.train()
    assert all(torch.isfinite(p).all() for p in m.parameters())
    assert hist["nonfinite_grad_steps"] == [1]


@requires_cuda
def test_fp16_scaler_state_is_saved_and_restored(tmp_path):
    tr = _ckpt_trainer(tmp_path, 2, amp="fp16", device=torch.device("cuda"))
    assert tr.grad_scaling and tr.scaler.is_enabled()
    tr.train()
    extra = torch.load(os.path.join(tmp_path, "ckpt-lambda=64.pt"), weights_only=False)["extra"]
    assert extra["amp"] == "fp16" and "scaler" in extra
    resumed = _ckpt_trainer(tmp_path, 3, amp="fp16", device=torch.device("cuda"), resume=True)
    assert resumed.scaler.get_scale() == tr.scaler.get_scale()
