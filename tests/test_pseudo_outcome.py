"""Tests for AIPCW pseudo-outcome computation."""

import jax.numpy as jnp
import numpy as np
import pytest

from air_cate._types import NuisanceEstimates, SurvivalData
from air_cate.pseudo_outcome import compute_pseudo_outcomes


def _make_oracle_data(n=5000, M=20, t_end=5.0, rng_seed=0):
    """Generate data with known (oracle) nuisance functions.

    Uses Weibull(shape=1.5, scale=2.0) for event times in both arms (no treatment effect),
    and Exponential(rate=0.1) for censoring.
    """
    rng = np.random.default_rng(rng_seed)
    p = 3
    X = rng.standard_normal((n, p)).astype(np.float32)
    W = rng.binomial(1, 0.5, n).astype(np.float32)

    shape, scale = 1.5, 2.0
    D = scale * (-np.log(rng.uniform(0, 1, n))) ** (1.0 / shape)
    cens_rate = 0.1
    C = rng.exponential(1.0 / cens_rate, n)
    Z = np.minimum(D, C).astype(np.float32)
    Delta = (D <= C).astype(np.float32)

    data = SurvivalData(
        X=jnp.array(X), W=jnp.array(W), Z=jnp.array(Z), Delta=jnp.array(Delta)
    )

    t_grid = jnp.linspace(0, t_end, M + 1)
    delta = t_end / M

    # Oracle nuisances: true Weibull survival and true censoring survival
    t_np = np.array(t_grid)
    S_event_grid = np.exp(-((t_np[None, :] / scale) ** shape)).astype(np.float32)  # (1, M+1)
    S_event_grid = np.broadcast_to(S_event_grid, (n, M + 1)).copy()

    S_event_at_Z = np.exp(-((Z / scale) ** shape)).astype(np.float32)

    S_cens_grid = np.exp(-cens_rate * t_np[None, :]).astype(np.float32)
    S_cens_grid = np.broadcast_to(S_cens_grid, (n, M + 1)).copy()

    S_cens_at_Z = np.exp(-cens_rate * Z).astype(np.float32)

    e_hat = np.full(n, 0.5, dtype=np.float32)

    nuisances = NuisanceEstimates(
        e_hat=jnp.array(e_hat),
        S_event_0_grid=jnp.array(S_event_grid),
        S_event_1_grid=jnp.array(S_event_grid),
        S_event_0_at_Z=jnp.array(S_event_at_Z),
        S_event_1_at_Z=jnp.array(S_event_at_Z),
        S_cens_0_grid=jnp.array(S_cens_grid),
        S_cens_1_grid=jnp.array(S_cens_grid),
        S_cens_0_at_Z=jnp.array(S_cens_at_Z),
        S_cens_1_at_Z=jnp.array(S_cens_at_Z),
    )

    true_S = np.exp(-((t_np / scale) ** shape)).astype(np.float32)

    return data, nuisances, t_grid, delta, true_S


def test_pseudo_outcome_at_zero():
    """Y_tilde(0) = 1 for all subjects (everyone alive at baseline)."""
    data, nuisances, t_grid, delta, _ = _make_oracle_data(n=500, M=20)
    Y_tilde, _, _ = compute_pseudo_outcomes(data, nuisances, t_grid, delta)
    np.testing.assert_allclose(np.array(Y_tilde[:, 0]), 1.0, atol=0.05)


def test_pseudo_outcome_shapes():
    """Output shapes are correct."""
    n, M = 100, 20
    data, nuisances, t_grid, delta, _ = _make_oracle_data(n=n, M=M)
    Y_tilde, S_marg, e_clipped = compute_pseudo_outcomes(data, nuisances, t_grid, delta)

    assert Y_tilde.shape == (n, M + 1)
    assert S_marg.shape == (n, M + 1)
    assert e_clipped.shape == (n,)


def test_pseudo_outcome_unbiasedness_oracle():
    """With oracle nuisances, mean(Y_tilde(t)) ≈ mean(S†_W(t|X)) for several grid points."""
    data, nuisances, t_grid, delta, true_S = _make_oracle_data(n=10000, M=50, rng_seed=42)
    Y_tilde, _, _ = compute_pseudo_outcomes(data, nuisances, t_grid, delta)

    # Check at several grid points
    for m_idx in [5, 10, 20, 30, 40]:
        mean_pseudo = float(jnp.mean(Y_tilde[:, m_idx]))
        true_val = float(true_S[m_idx])
        np.testing.assert_allclose(
            mean_pseudo, true_val, atol=0.05,
            err_msg=f"Unbiasedness failed at grid index {m_idx}: "
                    f"mean(Y_tilde)={mean_pseudo:.4f}, S_true={true_val:.4f}"
        )


def test_pseudo_outcome_no_nans():
    """No NaN values in pseudo-outcomes."""
    data, nuisances, t_grid, delta, _ = _make_oracle_data(n=500, M=20)
    Y_tilde, S_marg, e_clipped = compute_pseudo_outcomes(data, nuisances, t_grid, delta)

    assert not jnp.any(jnp.isnan(Y_tilde)), "NaN in Y_tilde"
    assert not jnp.any(jnp.isnan(S_marg)), "NaN in S_marg"
    assert not jnp.any(jnp.isnan(e_clipped)), "NaN in e_clipped"
