import numpy as np

from rdsandwich.utils.ba import discretize_and_run_ba


def test_ba_runs_and_rate_nonnegative():
    rng = np.random.RandomState(0)
    samples = rng.randn(2000, 2)
    records, log_Q, log_q_y = discretize_and_run_ba(samples, lamb=1.0, bins=20, steps=100, tol=1e-5)
    assert records[-1]["R"] >= -1e-6
    assert records[-1]["D"] >= 0


def test_ba_higher_lambda_gives_lower_distortion():
    rng = np.random.RandomState(1)
    samples = rng.randn(2000, 2)
    records_lo, *_ = discretize_and_run_ba(samples, lamb=0.1, bins=20, steps=100, tol=1e-5)
    records_hi, *_ = discretize_and_run_ba(samples, lamb=10.0, bins=20, steps=100, tol=1e-5)
    assert records_hi[-1]["D"] <= records_lo[-1]["D"]
