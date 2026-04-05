"""Tests for oracle nuisance builder and estimator oracle path."""

import jax.numpy as jnp
import numpy as np
import pytest

from conftest import WeibullDGP, WeibullMixtureDGP


def test_oracle_nuisances_weibull_shapes():
    """Oracle nuisances have correct shapes for WeibullDGP."""
    from air_cate.nuisance.oracle import build_oracle_nuisances
    from air_cate.utils import build_time_grid

    X, W, Z, Delta, t_end = WeibullDGP.generate(n=50, rng_seed=0)
    time_grid, _ = build_time_grid(t_end, M=20)
    nuis = build_oracle_nuisances(WeibullDGP, X, Z, time_grid)

    assert nuis.e_hat.shape == (50,)
    assert nuis.S_event_0_grid.shape == (50, 21)
    assert nuis.S_event_1_grid.shape == (50, 21)
    assert nuis.S_event_0_at_Z.shape == (50,)
    assert nuis.S_event_1_at_Z.shape == (50,)
    assert nuis.S_cens_0_grid.shape == (50, 21)
    assert nuis.S_cens_1_grid.shape == (50, 21)
    assert nuis.S_cens_0_at_Z.shape == (50,)
    assert nuis.S_cens_1_at_Z.shape == (50,)


def test_oracle_nuisances_mixture_shapes():
    """Oracle nuisances have correct shapes for WeibullMixtureDGP."""
    from air_cate.nuisance.oracle import build_oracle_nuisances
    from air_cate.utils import build_time_grid

    X, W, Z, Delta, t_end = WeibullMixtureDGP.generate(n=50, rng_seed=0)
    time_grid, _ = build_time_grid(t_end, M=20)
    nuis = build_oracle_nuisances(WeibullMixtureDGP, X, Z, time_grid)

    assert nuis.e_hat.shape == (50,)
    assert nuis.S_event_0_grid.shape == (50, 21)
    assert nuis.S_event_1_at_Z.shape == (50,)


def test_oracle_survivals_in_01():
    """All oracle survival values are in [0, 1]."""
    from air_cate.nuisance.oracle import build_oracle_nuisances
    from air_cate.utils import build_time_grid

    for dgp_class in [WeibullDGP, WeibullMixtureDGP]:
        X, W, Z, Delta, t_end = dgp_class.generate(n=100, rng_seed=1)
        time_grid, _ = build_time_grid(t_end, M=30)
        nuis = build_oracle_nuisances(dgp_class, X, Z, time_grid)

        for field in nuis:
            arr = np.asarray(field)
            assert np.all(arr >= -1e-6), f"Negative survival in {dgp_class.__name__}"
            assert np.all(arr <= 1.0 + 1e-6), f"Survival > 1 in {dgp_class.__name__}"


def test_oracle_propensity_matches_dgp():
    """Oracle propensity matches DGP propensity function."""
    from air_cate.nuisance.oracle import build_oracle_nuisances
    from air_cate.utils import build_time_grid

    X, W, Z, Delta, t_end = WeibullDGP.generate(n=50, rng_seed=0)
    time_grid, _ = build_time_grid(t_end, M=10)
    nuis = build_oracle_nuisances(WeibullDGP, X, Z, time_grid)

    expected_e = WeibullDGP.propensity(X)
    np.testing.assert_allclose(np.asarray(nuis.e_hat), expected_e, atol=1e-5)


def test_estimator_with_oracle_nuisances():
    """AIREstimator runs with pre-computed oracle nuisances (skips cross-fitting)."""
    from air_cate.estimator import AIREstimator
    from air_cate.nuisance.oracle import build_oracle_nuisances
    from air_cate.utils import build_time_grid

    X, W, Z, Delta, t_end = WeibullDGP.generate(n=200, rng_seed=0)
    M = 10
    time_grid, _ = build_time_grid(t_end, M)
    nuis = build_oracle_nuisances(WeibullDGP, X, Z, time_grid)

    est = AIREstimator(
        K=2, M=M, eps=0.01,
        cate_epochs=10,
    )
    est.fit(X, W, Z, Delta, t_end, nuisances=nuis, verbose=0)

    tau = est.predict_tau(X[:10])
    assert tau.shape == (10, M + 1)
