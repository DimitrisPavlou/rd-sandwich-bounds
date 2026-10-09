"""Under autocast, the layers whose output feeds the rate or the distortion stay fp32 (``call_fp32``),
while the rest of the network still runs in the autocast dtype. CPU autocast is enough to test this:
``call_fp32`` disables autocast on any device."""
import torch
import torch.nn as nn

from rdsandwich.layers.deep_factorized import DeepFactorized
from rdsandwich.models.upper_bound.resnet_vae import ResNetVAE, ResNetVAEConfig
from rdsandwich.models.upper_bound.variable_rate_lossy_vae import VariableRateLossyVAE, VariableRateLossyVAEConfig
from rdsandwich.utils.torch_utils import call_fp32


def _bf16_autocast():
    return torch.autocast("cpu", dtype=torch.bfloat16)


def _output_dtypes(model, x, names):
    """Run ``model(x)`` under bf16 autocast; return (outputs, {module name: output dtype})."""
    dtypes = {}

    def record(name):
        def hook(module, inputs, output):
            dtypes.setdefault(name, output.dtype)
        return hook

    modules = dict(model.named_modules())
    handles = [modules[n].register_forward_hook(record(n)) for n in names]
    with _bf16_autocast():
        out = model(x)
    for h in handles:
        h.remove()
    return out, dtypes


def test_call_fp32_keeps_a_layer_fp32_under_autocast():
    conv, x = nn.Conv2d(4, 4, 3, padding=1), torch.randn(1, 4, 8, 8)
    with _bf16_autocast():
        assert conv(x.float()).dtype == torch.bfloat16          # .float() alone is not enough
        assert call_fp32(conv, x.bfloat16()).dtype == torch.float32
    assert torch.equal(call_fp32(conv, x), conv(x))             # no-op outside autocast


def test_deep_factorized_log_prob_is_unaffected_by_autocast():
    torch.manual_seed(0)
    prior, z = DeepFactorized(4), torch.randn(2, 4, 3, 3)
    with _bf16_autocast():
        under_autocast = prior.log_prob_nchw(z)
    assert torch.equal(under_autocast, prior.log_prob_nchw(z))


def test_variable_rate_lossy_vae_heads_stay_fp32():
    torch.manual_seed(0)
    model = VariableRateLossyVAE(VariableRateLossyVAEConfig.from_preset("tiny")).train()
    heads = [n for n in dict(model.named_modules()) if n.endswith((".prior", ".posterior"))]
    trunk = "encoder.enc_blocks.1.conv_dw"
    out, dtypes = _output_dtypes(model, torch.rand(2, 3, 64, 64) * 255, heads + [trunk])
    assert all(dtypes[n] == torch.float32 for n in heads)
    assert dtypes[trunk] == torch.bfloat16                      # the rest still runs in bf16
    assert out["x_hat"].dtype == torch.float32


def test_resnet_vae_heads_stay_fp32():
    torch.manual_seed(0)
    cfg = ResNetVAEConfig(num_filters=16, ar_prior_levels=2, ar_slices=2, image_range="pm1")
    model = ResNetVAE(cfg).train()
    heads = [n for n in dict(model.named_modules()) if n.endswith((".gen_net", ".inf_net", ".td_inf_net"))]
    top_encoder, trunk = f"encoders.{len(model.encoders) - 1}", "encoders.0"
    out, dtypes = _output_dtypes(model, torch.rand(2, 3, 64, 64) * 255, heads + [top_encoder, trunk])
    assert all(dtypes[n] == torch.float32 for n in heads + [top_encoder])
    assert dtypes[trunk] == torch.bfloat16
    assert out["x_hat"].dtype == torch.float32
