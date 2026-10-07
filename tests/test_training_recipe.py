"""The shared-recipe options: const-cos LR, EMA of the weights, h-flip and reflect padding
of training patches, and the variable-rate checkpoint name."""
import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

from rdsandwich.cli.common import load_dataset
from rdsandwich.cli.upper_bound import build_eval_parser, build_train_parser, checkpoint_filename
from rdsandwich.data.base import build_loader
from rdsandwich.data.image import ImageFolderDataset, _reflect_pad_to
from rdsandwich.upper_bound.trainer import UpperBoundTrainer
from rdsandwich.utils.lr_schedulers import const_cos_factor, make_const_cos_scheduler
from rdsandwich.utils.io import load_checkpoint


# --------------------------------------------------------------------------- #
# const-cos LR schedule
# --------------------------------------------------------------------------- #
def test_const_cos_factor_shape():
    f = [const_cos_factor(e, 54) for e in range(54)]
    assert all(v == 1.0 for v in f[:28])                     # first half (+ cosine start = 1)
    assert f[-1] == pytest.approx(0.01)                      # last epoch: final factor
    assert all(a >= b for a, b in zip(f, f[1:]))             # never increases
    assert f[40] == pytest.approx(0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * 13 / 26)))
    assert [const_cos_factor(e, 1) for e in range(1)] == [1.0]   # degenerate lengths are safe
    assert const_cos_factor(1, 2) == pytest.approx(1.0)


def test_const_cos_scheduler_steps_per_epoch():
    p = nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([p], lr=2e-4)
    sched = make_const_cos_scheduler(opt, epochs=10)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[:6] == [2e-4] * 6 and lrs[-1] == pytest.approx(2e-6)


# --------------------------------------------------------------------------- #
# EMA
# --------------------------------------------------------------------------- #
class _ToyUB(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 4)
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def get_losses(self, x):
        self.count += 1
        mse = ((self.net(x) - x) ** 2).mean()
        rate = self.net.weight.abs().mean()
        return rate + 10.0 * mse, rate, mse


class _VecSrc:
    def sample(self, b):
        return torch.randn(b, 4)


def _trainer(model, ckpt=None, **kw):
    return UpperBoundTrainer(
        model, build_loader(_VecSrc(), batch_size=8),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2), epochs=kw.pop("epochs", 2),
        steps_per_epoch=5, ckpt_path=ckpt, verbose=False, **kw)


def test_ema_update_formula():
    torch.manual_seed(0)
    model = _ToyUB()
    trainer = _trainer(model, ema_decay=0.9999, ema_warmup=100)
    ema_w = trainer.ema_model.net.weight.clone()
    with torch.no_grad():
        model.net.weight.add_(1.0)
        model.count.fill_(7)
    trainer.global_step = 50
    trainer._ema_update()
    d = 0.9999 * (1 - math.exp(-50 / 100))
    assert torch.allclose(trainer.ema_model.net.weight, d * ema_w + (1 - d) * model.net.weight)
    assert trainer.ema_model.count.item() == 7                # integer buffers are copied
    assert not any(p.requires_grad for p in trainer.ema_model.parameters())


def test_ema_is_saved_resumed_and_loaded_for_eval(tmp_path):
    torch.manual_seed(0)
    ckpt = str(tmp_path / "ckpt.pt")
    model = _ToyUB()
    trainer = _trainer(model, ckpt, ema_decay=0.99, ema_warmup=1)
    trainer.train()
    assert trainer.global_step == 10                          # 2 epochs x 5 steps
    extra = torch.load(ckpt, weights_only=False)["extra"]
    assert extra["global_step"] == 10 and "ema_state_dict" in extra
    ema_w = trainer.ema_model.net.weight.detach().clone()
    assert not torch.allclose(ema_w, model.net.weight)        # the average lags the weights

    resumed = _trainer(_ToyUB(), ckpt, ema_decay=0.99, ema_warmup=1, epochs=3, resume=True)
    assert resumed.start_epoch == 2 and resumed.global_step == 10
    assert torch.equal(resumed.ema_model.net.weight, ema_w)
    resumed.train()
    assert resumed.global_step == 15

    fresh = _ToyUB()
    load_checkpoint(ckpt, fresh, use_ema=True)
    assert torch.equal(fresh.net.weight, resumed.ema_model.net.weight)
    load_checkpoint(ckpt, fresh)                              # default: raw weights
    assert torch.equal(fresh.net.weight, resumed.model.net.weight)


def test_no_ema_by_default(tmp_path):
    ckpt = str(tmp_path / "ckpt.pt")
    trainer = _trainer(_ToyUB(), ckpt)
    trainer.train()
    assert trainer.ema_model is None
    extra = torch.load(ckpt, weights_only=False)["extra"]
    assert "ema_state_dict" not in extra and extra["global_step"] == 10
    fresh = _ToyUB()
    load_checkpoint(ckpt, fresh, use_ema=True)                # falls back to the raw weights
    assert torch.equal(fresh.net.weight, trainer.model.net.weight)


# --------------------------------------------------------------------------- #
# Training patches: reflect padding and h-flip
# --------------------------------------------------------------------------- #
def _random_image(h, w, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("h, w", [(200, 640), (640, 200), (100, 70), (1, 300), (300, 300)])
def test_reflect_pad_matches_torchvision_random_crop_padding(h, w):
    """Same padding as RandomCrop(256, pad_if_needed=True, padding_mode='reflect') on a PIL
    image (torchvision pads each short side on both ends by 256 - side), for any size."""
    arr = _random_image(h, w)
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    pad_w, pad_h = max(256 - w, 0), max(256 - h, 0)
    ref = np.asarray(TF.pad(Image.fromarray(arr), [pad_w, pad_h], padding_mode="reflect"))
    ours = _reflect_pad_to(t, 256).permute(1, 2, 0).numpy()
    assert ours.shape == ref.shape and np.array_equal(ours, ref)
    assert min(ours.shape[:2]) >= 256


def _folder(tmp_path, shapes):
    for i, (h, w) in enumerate(shapes):
        Image.fromarray(_random_image(h, w, seed=i)).save(tmp_path / f"{i}.png")
    return str(tmp_path)


def test_small_image_modes(tmp_path):
    folder = _folder(tmp_path, [(200, 640)])
    original = set(np.unique(_random_image(200, 640, seed=0)).tolist())
    padded = ImageFolderDataset(folder, patchsize=256, small_image_mode="reflect_pad")
    resized = ImageFolderDataset(folder, patchsize=256)               # default: resize
    for _ in range(5):
        p = padded[0]
        assert p.shape == (3, 256, 256)
        assert set(np.unique(p.numpy().astype(np.uint8)).tolist()) <= original  # only real pixels
        assert resized[0].shape == (3, 256, 256)
    with pytest.raises(ValueError, match="small_image_mode"):
        ImageFolderDataset(folder, small_image_mode="stretch")


def test_hflip_mirrors_about_half_the_patches(tmp_path):
    folder = _folder(tmp_path, [(64, 96)])
    image = torch.from_numpy(_random_image(64, 96)).permute(2, 0, 1).float()
    flipping = ImageFolderDataset(folder, patchsize=None, hflip=True)
    torch.manual_seed(0)
    n_flipped = 0
    for _ in range(200):
        p = flipping[0]
        assert torch.equal(p, image) or torch.equal(p, image.flip(-1))
        n_flipped += torch.equal(p, image.flip(-1))
    assert 60 < n_flipped < 140
    plain = ImageFolderDataset(folder, patchsize=None)
    assert all(torch.equal(plain[0], image) for _ in range(20))
    assert all(torch.equal(x[0], image) for x in flipping.all_images())   # eval: never flipped


def test_cli_data_flags_reach_the_dataset(tmp_path):
    folder = _folder(tmp_path, [(64, 64)])
    args = build_train_parser().parse_args(
        ["--model", "resnet_vae", "--dataset", folder, "--patchsize", "32",
         "--hflip", "--small_image_mode", "reflect_pad"])
    ds = load_dataset(args, device="cpu")
    assert ds.hflip and ds.small_image_mode == "reflect_pad"
    ds = load_dataset(build_train_parser().parse_args(["--model", "resnet_vae", "--dataset", folder]),
                      device="cpu")
    assert not ds.hflip and ds.small_image_mode == "resize"


# --------------------------------------------------------------------------- #
# CLI: flags and checkpoint names
# --------------------------------------------------------------------------- #
def test_recipe_flags_and_checkpoint_names():
    p = build_train_parser()
    a = p.parse_args(["--model", "variable_rate_lossy_vae", "--dataset", "x",
                      "--lr_schedule", "const-cos", "--ema", "0.9999"])
    assert a.lr_schedule == "const-cos" and a.ema == 0.9999 and a.ema_warmup == 10_000
    assert checkpoint_filename(a) == "ckpt.pt"
    a = p.parse_args(["--model", "resnet_vae", "--dataset", "x", "--lambda", "64"])
    assert a.ema is None and checkpoint_filename(a) == "ckpt-lambda=64.pt"
    e = build_eval_parser().parse_args(["--model", "resnet_vae", "--dataset", "x", "--no_ema"])
    assert e.no_ema
