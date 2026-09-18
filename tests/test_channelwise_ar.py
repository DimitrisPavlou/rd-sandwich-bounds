"""Tests for the channel-wise autoregressive transform."""
import torch

from rdsandwich.models import ChannelwiseARTransform


def test_shape_preserved_and_first_slice_zero():
    ar = ChannelwiseARTransform(input_num_channels=16, num_slices=8)
    x = torch.randn(2, 16, 6, 6)
    out = ar(x)
    assert out.shape == x.shape
    # first slice (slice_depth channels) is predicted as zeros
    assert torch.count_nonzero(out[:, : ar.slice_depth]) == 0


def test_autoregressive_causality():
    torch.manual_seed(0)
    ar = ChannelwiseARTransform(input_num_channels=16, num_slices=8)
    x = torch.randn(2, 16, 5, 5)
    out = ar(x)
    x2 = x.clone()
    x2[:, -ar.slice_depth:] += 10.0  # perturb only the last slice
    out2 = ar(x2)
    # early output slices must not depend on later input slices
    early = 4 * ar.slice_depth
    assert torch.allclose(out[:, :early], out2[:, :early])


def test_num_slices_clamped_to_channels():
    ar = ChannelwiseARTransform(input_num_channels=4, num_slices=8)
    assert ar.num_slices == 4
    assert ar.slice_depth == 1


def test_gradients_flow():
    ar = ChannelwiseARTransform(input_num_channels=8, num_slices=4)
    x = torch.randn(2, 8, 4, 4, requires_grad=True)
    ar(x).sum().backward()
    assert torch.isfinite(x.grad).all()
