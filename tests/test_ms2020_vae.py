"""Tests for the Minnen & Singh 2020 beta-VAE image R-D upper bound."""
import torch

from rdsandwich.ms2020_vae import MS2020VAE, MS2020VAEConfig


def _small(**kw):
    base = dict(latent_depth=40, hyperprior_depth=24, num_filters=32,
                num_slices=5, max_support_slices=3, hyper_synth_out=48)
    base.update(kw)
    return MS2020VAEConfig(**base)


def test_forward_backward_finite():
    torch.manual_seed(0)
    m = MS2020VAE(_small(lmbda=0.02))
    x = torch.rand(2, 3, 64, 64) * 255.0
    out = m(x)
    assert out["x_hat"].shape == x.shape
    assert out["bits"].shape == (2,)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_loss_is_bpp_plus_lambda_mse():
    torch.manual_seed(0)
    m = MS2020VAE(_small(lmbda=0.05))
    x = torch.rand(2, 3, 64, 64) * 255.0
    out = m(x)
    assert torch.allclose(out["loss"], out["bpp"] + 0.05 * out["mse"])


def test_latent_depth_must_divide_slices():
    try:
        MS2020VAE(MS2020VAEConfig(latent_depth=41, num_slices=10))
    except ValueError:
        return
    raise AssertionError("expected ValueError for indivisible latent_depth")


def test_channel_conditioning_shapes():
    """cc/lrp transforms must accept the growing conditioning input per slice."""
    cfg = _small()
    m = MS2020VAE(cfg)
    sd = m.slice_depth
    cond = cfg.hyper_synth_out
    for i in range(cfg.num_slices):
        n_support = min(i, cfg.max_support_slices)
        # first conv of cc_mean[i] must accept cond + n_support*sd channels
        first = m.cc_mean[i].net[0][0]
        assert first.in_channels == cond + n_support * sd
        first_lrp = m.lrp[i].net[0][0]
        assert first_lrp.in_channels == cond + n_support * sd + sd


def test_eval_runs_under_no_grad():
    m = MS2020VAE(_small()).eval()
    with torch.no_grad():
        out = m(torch.rand(1, 3, 64, 64) * 255.0)
    assert torch.isfinite(out["bpp"]) and torch.isfinite(out["mse"])


def test_overfits_a_fixed_batch():
    # The IGDN synthesis transform is divergence-prone at high LR, so (as in the
    # paper's 1e-4 schedule) training uses a modest LR + gradient clipping.
    torch.manual_seed(0)
    m = MS2020VAE(_small(lmbda=0.02))
    x = torch.rand(2, 3, 64, 64) * 255.0
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    first = None
    for it in range(300):
        opt.zero_grad()
        out = m(x)
        assert torch.isfinite(out["loss"]), f"diverged at step {it}"
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if it == 0:
            first = out["mse"].item()
    assert out["mse"].item() < 0.5 * first
