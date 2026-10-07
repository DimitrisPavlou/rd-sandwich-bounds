"""Tests for variable_rate_lossy_vae, Duan et al.'s variable-rate ResNet-VAE image R-D upper bound."""
import math

import pytest
import torch

from rdsandwich.cli.upper_bound import build_model, build_train_parser, downsampling_factor, get_runname
from rdsandwich.models.upper_bound.variable_rate_lossy_vae import (
    MSE_PM1_TO_255,
    VariableRateLossyVAE,
    VariableRateLossyVAEConfig,
    linear_sqrt,
)
from rdsandwich.upper_bound.evaluate import evaluate_full_images


def _tiny(**kw):
    return VariableRateLossyVAE(VariableRateLossyVAEConfig.from_preset("tiny", **kw))


def _images(n=2, h=64, w=128):
    return torch.randint(0, 256, (n, 3, h, w)).float()


def test_forward_backward_finite():
    torch.manual_seed(0)
    m = _tiny().train()
    x = _images()
    out = m(x)
    assert out["x_hat"].shape == x.shape
    assert out["bits"].shape == (2,) and out["mses"].shape == (2,)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_loss_is_duan_objective_and_metrics_are_on_255_scale():
    """loss = mean(KL nats/dim + lambda * MSE on [-1, 1]); logged bpp / MSE are bits/pixel and [0, 255]."""
    torch.manual_seed(0)
    m = _tiny().eval()
    x = _images()
    lmb = torch.tensor([16.0, 512.0])
    out = m(x, lmbda=lmb)
    n, c, h, w = x.shape
    rate = out["bits"] * math.log(2) / (c * h * w)
    mse_pm1 = out["mses"] / MSE_PM1_TO_255
    assert torch.allclose(out["loss"], (rate + lmb * mse_pm1).mean())
    assert torch.allclose(out["bpp"], out["bits"].sum() / (n * h * w))
    # mses is the MSE between x_hat (on [0, 255]) and x
    assert torch.allclose(out["mses"], ((out["x_hat"] - x) ** 2).flatten(1).mean(-1), rtol=1e-4)
    assert torch.allclose(out["mse"], out["mses"].mean())
    # get_losses -> (loss, bpp, mse) of the same forward pass
    m.cfg.lmbda = 16.0
    torch.manual_seed(1)
    loss, bpp, mse = m.get_losses(x)
    torch.manual_seed(1)
    ref = m(x, lmbda=16.0)
    assert torch.allclose(loss, ref["loss"]) and torch.allclose(bpp, ref["bpp"])
    assert torch.allclose(mse, ref["mse"])


def test_get_losses_samples_lambda_in_training():
    torch.manual_seed(0)
    m = _tiny().train()
    loss, bpp, mse = m.get_losses(_images())
    assert loss.dim() == bpp.dim() == mse.dim() == 0
    out = m(_images(n=64, h=64, w=64))
    low, high = m.cfg.lmb_range
    assert ((out["lmbda"] >= low) & (out["lmbda"] <= high)).all()
    assert out["lmbda"].std() > 0  # one lambda per image


def test_sample_lmb_is_log_uniform():
    torch.manual_seed(0)
    m = _tiny()
    lmb = m.sample_lmb(200_000, device="cpu")
    low, high = m.cfg.lmb_range
    assert lmb.min() >= low and lmb.max() <= high
    assert torch.median(lmb).item() == pytest.approx(math.sqrt(low * high), rel=0.02)


def test_eval_lambda_handling():
    m = _tiny().eval()
    x = _images(n=1)
    with pytest.raises(ValueError, match="No lambda"):
        m(x)
    with pytest.raises(ValueError, match="outside the training range"):
        m(x, lmbda=0.01)  # an old [0, 255]-convention lambda
    m.cfg.lmbda = 64.0
    with torch.no_grad():
        out = m(x)  # uses cfg.lmbda
    assert torch.allclose(out["lmbda"], torch.tensor([64.0]))


def test_rejects_sides_not_multiple_of_64():
    with pytest.raises(ValueError, match="multiples of 64"):
        _tiny().train()(_images(h=64, w=96))


def test_parameter_counts_match_paper_table3():
    expected = {"base": 186.7, "c96_l15": 111.8, "c64_l15": 55.8, "c64_l10": 46.1, "c64_l5": 34.2,
                "c32_l15": 18.6}  # the last one is not in the paper
    for preset, millions in expected.items():
        with torch.device("meta"):
            m = VariableRateLossyVAE(VariableRateLossyVAEConfig.from_preset(preset))
        n = sum(p.numel() for p in m.parameters())
        assert round(n / 1e6, 1) == millions, preset


def test_state_dict_names_match_original():
    """Names follow lvae/models/rd so Duan's checkpoints load unchanged."""
    with torch.device("meta"):
        model = VariableRateLossyVAE(VariableRateLossyVAEConfig.from_preset("base"))
    keys = set(model.state_dict())
    for k in ["bias", "lmb_embedding.0.weight", "lmb_embedding.2.bias",
              "encoder.enc_blocks.0.weight",                      # 4x patch embedding
              "encoder.enc_blocks.7.downsapmle.weight",           # ConvNeXtAdaLNPatchDown (sic)
              "encoder.enc_blocks.1.embedding_layer.1.weight",    # AdaLN
              "encoder.enc_blocks.1.mlp.fc1.weight", "encoder.enc_blocks.1.gamma",
              "dec_blocks.0.posterior0.mlp.fc2.weight", "dec_blocks.0.post_merge.weight",
              "dec_blocks.0.prior.weight", "dec_blocks.0.z_proj.weight",
              "dec_blocks.1.0.weight",                            # first patch_upsample
              "dec_blocks.19.0.weight"]:                          # final 4x upsample to RGB
        assert k in keys, k
    assert len([k for k in keys if k.endswith("post_merge.weight")]) == 15
    assert not any(k.startswith("dec_blocks.20.") for k in keys)  # 15 latents + 5 upsamplers


def test_linear_sqrt_values_and_bounded_gradient():
    a = torch.linspace(-1e4, 1e4, 200_001, requires_grad=True)
    y = linear_sqrt(a)
    assert torch.allclose(y, -linear_sqrt(-a))                       # odd
    big = a.detach().abs() > 100
    assert torch.allclose(y[big].abs(), a.detach()[big].abs().sqrt(), rtol=1e-4)  # ~sqrt
    y.sum().backward()
    assert torch.isfinite(a.grad).all() and a.grad.max() < 1.5      # bounded slope (peak ~1.24)
    assert a.grad[-1].item() == pytest.approx(0.5 / math.sqrt(1e4), rel=1e-3)  # sqrt regime
    z = torch.zeros(1, requires_grad=True)
    linear_sqrt(z).sum().backward()
    assert z.grad.item() == pytest.approx(1.0)                       # ~identity at 0


def test_smooth_flag_only_changes_the_means():
    torch.manual_seed(0)
    smooth = _tiny()
    plain = _tiny(smooth=False)
    plain.load_state_dict(smooth.state_dict())
    block_s = next(b for b in smooth.dec_blocks if getattr(b, "is_latent_block", False))
    block_p = next(b for b in plain.dec_blocks if getattr(b, "is_latent_block", False))
    feat = torch.randn(1, block_s.in_channels, 1, 2) * 50
    emb = smooth.lmb_embedding_of(torch.tensor([64.0]))
    _, pm_s, pv_s = block_s.transform_prior(feat, emb)
    _, pm_p, pv_p = block_p.transform_prior(feat, emb)
    assert torch.allclose(pv_s, pv_p)
    assert torch.allclose(pm_s, linear_sqrt(pm_p))


def test_fits_a_fixed_batch_at_fixed_lambda():
    torch.manual_seed(0)
    m = _tiny().train()
    # smooth, memorizable images (random noise is incompressible)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 64), torch.linspace(0, 1, 64), indexing="ij")
    x = torch.stack([xx, yy, (xx + yy) / 2]).unsqueeze(0).repeat(2, 1, 1, 1) * 255.0
    x[1] = 255.0 - x[1]
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        out = m(x, lmbda=256.0)
        out["loss"].backward()
        opt.step()
        losses.append(out["loss"].item())
    assert sum(losses[-5:]) / 5 < 0.5 * sum(losses[:5]) / 5


def test_full_image_evaluation_pads_and_crops():
    torch.manual_seed(0)
    m = _tiny()
    m.cfg.lmbda = 64.0
    images = [_images(n=1, h=70, w=90), _images(n=1, h=64, w=64)]
    res = evaluate_full_images(m, images, pad_factor=64)
    assert res["bpp"].shape == res["psnr"].shape == (2,)
    assert (res["bpp"] > 0).all()


def test_cli_builds_model_and_run_name():
    p = build_train_parser()
    args = p.parse_args(["--model", "variable_rate_lossy_vae", "--dataset", "x", "--preset", "tiny",
                         "--lambda", "64", "--no_smooth"])
    m = build_model(args)
    assert isinstance(m, VariableRateLossyVAE) and not m.cfg.smooth and m.cfg.lmbda == 64.0
    assert downsampling_factor(args) == 64
    assert get_runname(args) == (
        "rdub-model=variable_rate_lossy_vae-C=8-L=1_1_1_1_1-lmb=4_2048-nosmooth-tiny")
    args = p.parse_args(["--model", "variable_rate_lossy_vae", "--dataset", "x"])
    assert get_runname(args) == "rdub-model=variable_rate_lossy_vae-C=128-L=1_2_3_4_5-lmb=4_2048"
    args = p.parse_args(["--model", "variable_rate_lossy_vae", "--dataset", "x",
                         "--preset", "c64_l5",
                         "--latents_per_scale", "1,1,1,2,2"])
    assert build_model(args).cfg.latents_per_scale == [1, 1, 1, 2, 2]


def test_linear_sqrt_gradient_is_finite_in_fp16():
    """Under AMP, near-zero means underflow to exact (signed) zeros in fp16; the gradient
    there must be 1, not NaN (which would poison every gradient of the model)."""
    a = torch.tensor([0.0, -0.0, 1e-3, 2.0, -7.0], dtype=torch.float16, requires_grad=True)
    linear_sqrt(a).sum().backward()
    assert torch.isfinite(a.grad).all()
    assert a.grad[0].item() == pytest.approx(1.0) and a.grad[1].item() == pytest.approx(1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA autocast")
def test_amp_training_step_has_finite_gradients():
    torch.manual_seed(0)
    m = _tiny().cuda().train()
    x = _images(n=2).cuda()
    for scale in (2.0 ** 16, 2.0 ** 8):   # GradScaler's initial scale, and a reduced one
        m.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            loss, _, _ = m.get_losses(x)
        (loss * scale).backward()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        # overflow (inf) at a high scale is normal: the GradScaler skips the step and lowers
        # the scale. NaN is not: it does not go away when the scale is lowered.
        assert grads and not any(torch.isnan(g).any() for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
