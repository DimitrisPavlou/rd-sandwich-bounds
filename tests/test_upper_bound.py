"""Tests for the model-agnostic UpperBoundTrainer options and the UB evaluators."""
import numpy as np
import torch
import torch.nn as nn

from rdsandwich.data import build_loader, get_dataset
from rdsandwich.models.upper_bound import ResNetVAE, ResNetVAEConfig
from rdsandwich.upper_bound import UpperBoundTrainer, evaluate_full_images, evaluate_sampled


class _ToyUB(nn.Module):
    """Smallest possible upper-bound model: anything with get_losses(x) is trainable."""

    def __init__(self, dim=4):
        super().__init__()
        self.net = nn.Linear(dim, dim)

    def get_losses(self, x):
        mse = ((self.net(x) - x) ** 2).mean()
        rate = self.net.weight.abs().mean()
        return rate + 10.0 * mse, rate, mse


class _ImgSrc:
    def sample(self, b):
        return torch.rand(b, 3, 32, 32) * 255.0


class _VecSrc:
    def sample(self, b):
        return torch.randn(b, 4)


def _tiny_resnet_vae():
    return ResNetVAE(ResNetVAEConfig(latent_channels=[4, 8], num_filters=8))


def test_any_get_losses_model_trains_and_saves_metadata(tmp_path):
    torch.manual_seed(0)
    model = _ToyUB()
    ckpt = tmp_path / "ckpt.pt"
    trainer = UpperBoundTrainer(
        model, build_loader(_VecSrc(), batch_size=16),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2), epochs=2, steps_per_epoch=5,
        ckpt_path=str(ckpt), verbose=False, checkpoint_metadata={"model": "toy"},
    )
    history = trainer.train()
    assert set(history) >= {"loss", "rate", "mse"}
    extra = torch.load(ckpt, weights_only=False)["extra"]
    assert extra["model"] == "toy" and extra["cfg"] is None



def test_external_model_on_array_file_anywhere(tmp_path):
    """A model defined outside rdsandwich, trained on a .npy file outside data/,
    through the same get_dataset + build_loader + UpperBoundTrainer path as train_ub."""
    torch.manual_seed(0)
    path = tmp_path / "elsewhere" / "my_data.npy"
    path.parent.mkdir()
    np.save(path, np.random.randn(256, 4).astype(np.float32))

    dataset = get_dataset(str(path))
    model = _ToyUB(dim=4)
    trainer = UpperBoundTrainer(
        model, build_loader(dataset, batch_size=32),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2), epochs=4, steps_per_epoch=0,
        verbose=False,
    )
    history = trainer.train()
    assert len(history["loss"]) == 4
    assert history["loss"][-1] < history["loss"][0]


def test_channels_last_training_step():
    torch.manual_seed(0)
    model = _tiny_resnet_vae()
    trainer = UpperBoundTrainer(
        model, build_loader(_ImgSrc(), batch_size=2),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4), epochs=1, steps_per_epoch=1,
        verbose=False, channels_last=True,
    )
    conv_weights = [p for p in trainer.model.parameters() if p.dim() == 4]
    assert all(w.is_contiguous(memory_format=torch.channels_last) for w in conv_weights)
    history = trainer.train()
    assert torch.isfinite(torch.tensor(history["loss"])).all()


def test_diagnostic_forward_uses_model_diagnostics():
    model = _tiny_resnet_vae()
    trainer = UpperBoundTrainer(
        model, build_loader(_ImgSrc(), batch_size=1),
        optimizer=torch.optim.Adam(model.parameters()), epochs=1, steps_per_epoch=1, verbose=False,
    )
    with torch.no_grad():
        stats = trainer.diagnostic_forward(_ImgSrc().sample(1))
    assert {"level0", "level1"} <= set(stats)

    toy = _ToyUB()
    trainer = UpperBoundTrainer(
        toy, build_loader(_VecSrc(), batch_size=1),
        optimizer=torch.optim.Adam(toy.parameters()), epochs=1, steps_per_epoch=1, verbose=False,
    )
    with torch.no_grad():
        assert trainer.diagnostic_forward(_VecSrc().sample(2)) == {}


def test_evaluate_full_images_pads_and_crops():
    torch.manual_seed(0)
    model = _tiny_resnet_vae()
    # Odd sizes: not multiples of the 2**len(latent_channels) = 4 downsampling factor.
    images = [torch.rand(1, 3, 30, 34) * 255.0, torch.rand(1, 3, 33, 29) * 255.0]
    res = evaluate_full_images(model, images, pad_factor=4)
    assert {k: v.shape for k, v in res.items()} == {"bpp": (2,), "mse": (2,), "psnr": (2,)}
    assert (res["bpp"] > 0).all() and (res["mse"] >= 0).all()


def test_evaluate_sampled_reports_ci():
    res = evaluate_sampled(_ToyUB(), _VecSrc(), batchsize=8, num_batches=4)
    lo, hi = res["rate_ci"]
    assert lo <= res["rate_mean"] <= hi
