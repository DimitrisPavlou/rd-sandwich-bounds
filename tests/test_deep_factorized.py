"""Tests for the deep factorized density (Ballé 2018, App. 6.1)."""
import torch

from rdsandwich.models import DeepFactorized


def test_density_normalizes_to_one():
    torch.manual_seed(0)
    df = DeepFactorized(channels=2)
    xs = torch.linspace(-8000, 8000, 200001).unsqueeze(-1).repeat(1, 2)
    with torch.no_grad():
        p = df.log_prob(xs).exp()
    dx = (xs[1, 0] - xs[0, 0]).item()
    mass = (p.sum(0) * dx)
    assert torch.allclose(mass, torch.ones(2), atol=2e-2), mass


def test_cumulative_spans_unit_interval():
    torch.manual_seed(1)
    df = DeepFactorized(channels=1)
    ends = torch.tensor([[-1e4], [1e4]]).permute(1, 0).unsqueeze(1)
    logit, _ = df._cumulative_and_derivative(ends)
    c = torch.sigmoid(logit).flatten()
    assert c[0] < 1e-3 and c[1] > 1 - 1e-3


def test_gradients_flow_to_input_and_params():
    df = DeepFactorized(channels=3)
    z = torch.randn(4, 3, requires_grad=True)
    df.log_prob(z).sum().backward()
    assert torch.isfinite(z.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in df.parameters())


def test_nchw_reduction_shape():
    df = DeepFactorized(channels=3)
    out = df.log_prob_nchw(torch.randn(2, 3, 5, 5))
    assert out.shape == (2,)


def test_can_fit_standard_normal():
    torch.manual_seed(0)
    df = DeepFactorized(channels=1)
    opt = torch.optim.Adam(df.parameters(), lr=1e-2)
    for _ in range(1500):
        loss = -df.log_prob(torch.randn(4096, 1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # Gaussian differential entropy is 0.5*log(2*pi*e) ~= 1.419 nats.
    assert abs(loss.item() - 1.419) < 0.1, loss.item()
