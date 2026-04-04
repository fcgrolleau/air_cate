"""Tests for cross-fitting orchestrator."""

import numpy as np
import jax.numpy as jnp
import pytest

from air_cate._types import SurvivalData
from air_cate.nuisance.cross_fit import cross_fit_nuisances


def _make_simple_data(n=300, p=3, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    prob = 1 / (1 + np.exp(-0.5 * X[:, 0]))
    W = rng.binomial(1, prob).astype(np.float32)
    # Weibull event times
    scale = np.exp(1.0 + 0.3 * X[:, 0] + 0.5 * W * X[:, 1])
    D = scale * (-np.log(rng.uniform(0, 1, n))) ** (1.0 / 1.5)
    C = rng.exponential(5.0, n)
    Z = np.minimum(D, C).astype(np.float32)
    Delta = (D <= C).astype(np.float32)
    return SurvivalData(
        X=jnp.array(X), W=jnp.array(W), Z=jnp.array(Z), Delta=jnp.array(Delta)
    )


def test_cross_fit_shapes():
    """Output shapes match data dimensions."""
    data = _make_simple_data(n=200, p=3)
    t_grid = jnp.linspace(0, 5.0, 21)  # M=20

    nuisances = cross_fit_nuisances(
        data, t_grid, n_folds=2,
        training_config={"epochs": 5, "verbose": 0},
    )

    n = 200
    M_plus_1 = 21
    assert nuisances.e_hat.shape == (n,)
    assert nuisances.S_event_0_grid.shape == (n, M_plus_1)
    assert nuisances.S_event_1_grid.shape == (n, M_plus_1)
    assert nuisances.S_event_0_at_Z.shape == (n,)
    assert nuisances.S_event_1_at_Z.shape == (n,)
    assert nuisances.S_cens_0_grid.shape == (n, M_plus_1)
    assert nuisances.S_cens_1_grid.shape == (n, M_plus_1)
    assert nuisances.S_cens_0_at_Z.shape == (n,)
    assert nuisances.S_cens_1_at_Z.shape == (n,)


def test_cross_fit_no_nans():
    """No NaN values in nuisance estimates."""
    data = _make_simple_data(n=200, p=3)
    t_grid = jnp.linspace(0, 5.0, 21)

    nuisances = cross_fit_nuisances(
        data, t_grid, n_folds=2,
        training_config={"epochs": 5, "verbose": 0},
    )

    for name in nuisances._fields:
        arr = getattr(nuisances, name)
        assert not jnp.any(jnp.isnan(arr)), f"NaN found in {name}"


def test_cross_fit_propensity_range():
    """Propensity estimates are in (0, 1)."""
    data = _make_simple_data(n=200, p=3)
    t_grid = jnp.linspace(0, 5.0, 21)

    nuisances = cross_fit_nuisances(
        data, t_grid, n_folds=2,
        training_config={"epochs": 5, "verbose": 0},
    )

    assert jnp.all(nuisances.e_hat > 0)
    assert jnp.all(nuisances.e_hat < 1)
