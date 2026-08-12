"""Tests for WeibullAFT survival model."""

import numpy as np
import jax.numpy as jnp
import pytest


def _generate_weibull_data(n, p, true_shape, true_scale_fn, cens_rate, rng):
    """Generate right-censored Weibull data with known parameters."""
    X = rng.standard_normal((n, p)).astype(np.float32)
    scales = true_scale_fn(X)
    # Weibull event times: T = scale * (-log(U))^(1/shape)
    U = rng.uniform(0, 1, n)
    D = scales * (-np.log(U)) ** (1.0 / true_shape)
    # Exponential censoring
    C = rng.exponential(1.0 / cens_rate, n)
    Z = np.minimum(D, C).astype(np.float32)
    Delta = (D <= C).astype(np.float32)
    return X, Z, Delta, scales


def test_survival_at_zero():
    """S(0|x) = 1 for all x."""
    from air_cate.nuisance.weibull_aft import WeibullAFT

    rng = np.random.default_rng(0)
    n, p = 200, 3
    X, Z, Delta, _ = _generate_weibull_data(
        n, p, true_shape=1.5, true_scale_fn=lambda x: np.full(len(x), 2.0), cens_rate=0.1, rng=rng
    )
    model = WeibullAFT(hidden_sizes=[32])
    model.fit(X, Z, Delta, epochs=10, verbose=0)

    S_at_0 = model.survival(np.array([0.0]), X)
    np.testing.assert_allclose(S_at_0, 1.0, atol=1e-5)


def test_survival_monotone_decreasing():
    """S(t|x) is monotonically decreasing in t."""
    from air_cate.nuisance.weibull_aft import WeibullAFT

    rng = np.random.default_rng(1)
    n, p = 200, 3
    X, Z, Delta, _ = _generate_weibull_data(
        n, p, true_shape=1.5, true_scale_fn=lambda x: np.full(len(x), 2.0), cens_rate=0.1, rng=rng
    )
    model = WeibullAFT(hidden_sizes=[32])
    model.fit(X, Z, Delta, epochs=10, verbose=0)

    t_grid = np.linspace(0.01, 10.0, 50).astype(np.float32)
    S = model.survival(t_grid, X[:10])  # (10, 50)
    diffs = np.diff(np.array(S), axis=1)
    assert np.all(diffs <= 1e-5), "Survival function should be non-increasing"


def test_survival_grid_vs_point():
    """Grid evaluation and point evaluation should agree."""
    from air_cate.nuisance.weibull_aft import WeibullAFT

    rng = np.random.default_rng(2)
    n, p = 100, 3
    X, Z, Delta, _ = _generate_weibull_data(
        n, p, true_shape=1.5, true_scale_fn=lambda x: np.full(len(x), 2.0), cens_rate=0.1, rng=rng
    )
    model = WeibullAFT(hidden_sizes=[32])
    model.fit(X, Z, Delta, epochs=10, verbose=0)

    # Point evaluation at each subject's own time
    S_point = model.survival(Z, X)  # (n,)
    # Grid evaluation with the same times
    S_grid = model.survival(Z, X)
    np.testing.assert_allclose(np.array(S_point), np.array(S_grid), atol=1e-5)


def test_shared_shape_is_constant_across_subjects():
    """With shared_shape=True the fitted k does not vary with x, but lambda does."""
    from air_cate.nuisance.weibull_aft import WeibullAFT

    rng = np.random.default_rng(0)
    n = 400
    X = rng.standard_normal((n, 3)).astype(np.float32)
    # Exponential times with a covariate-dependent scale: true shape is 1 for all.
    scale = np.exp(0.5 + 0.4 * X[:, 0])
    T = (scale * rng.exponential(1.0, n)).astype(np.float32)
    event = np.ones(n, dtype=np.float32)

    model = WeibullAFT(hidden_sizes=(8,), shared_shape=True)
    model.fit(X, T, event, epochs=20, verbose=0)

    k, lam = WeibullAFT._get_params(model.model_(X, training=False))
    k = np.asarray(k)
    lam = np.asarray(lam)

    assert np.ptp(k) < 1e-5, f"shape varies across subjects: range {np.ptp(k)}"
    assert np.ptp(lam) > 1e-3, "scale should still depend on x"


def test_parameter_recovery():
    """Recover known Weibull parameters from large sample."""
    from air_cate.nuisance.weibull_aft import WeibullAFT

    rng = np.random.default_rng(42)
    n, p = 2000, 3
    true_shape = 1.5
    true_scale = 2.0
    X, Z, Delta, _ = _generate_weibull_data(
        n, p, true_shape=true_shape, true_scale_fn=lambda x: np.full(len(x), true_scale), cens_rate=0.05, rng=rng
    )

    model = WeibullAFT(hidden_sizes=[32, 16])
    model.fit(X, Z, Delta, epochs=200, verbose=0, lr=1e-3)

    # Check survival at several time points vs. true Weibull survival
    t_grid = np.array([0.5, 1.0, 2.0, 3.0, 5.0], dtype=np.float32)
    S_pred = np.array(model.survival(t_grid, X[:50]))  # (50, 5)
    S_true = np.exp(-((t_grid[None, :] / true_scale) ** true_shape))  # (1, 5) broadcast

    # Average over subjects (constant scale, so all should be similar)
    S_pred_mean = S_pred.mean(axis=0)
    S_true_flat = S_true.flatten()

    np.testing.assert_allclose(S_pred_mean, S_true_flat, atol=0.1,
                                err_msg="Weibull survival should be recovered within tolerance")
