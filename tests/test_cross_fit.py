"""Tests for cross-fitting orchestrator."""

import numpy as np
import jax.numpy as jnp
import pytest

from air_cate._types import SurvivalData
from air_cate.nuisance.cross_fit import cross_fit_nuisances


def _make_simple_data(n=300, p=3, rng_seed=0, t_end=None):
    """Synthetic right-censored data. With `t_end`, also truncate at t_end.

    Truncating is what puts an atom in the censoring variable min(C, t_end), the
    case `admin_as_censored` exists for.
    """
    rng = np.random.default_rng(rng_seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    prob = 1 / (1 + np.exp(-0.5 * X[:, 0]))
    W = rng.binomial(1, prob).astype(np.float32)
    # Weibull event times
    scale = np.exp(1.0 + 0.3 * X[:, 0] + 0.5 * W * X[:, 1])
    D = scale * (-np.log(rng.uniform(0, 1, n))) ** (1.0 / 1.5)
    C = rng.exponential(5.0, n)
    if t_end is None:
        Z = np.minimum(D, C).astype(np.float32)
        Delta = (D <= C).astype(np.float32)
    else:
        Z = np.minimum(D, np.minimum(C, t_end)).astype(np.float32)
        Delta = (D < np.minimum(C, t_end)).astype(np.float32)
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


def test_cross_fit_does_not_mutate_training_config():
    """The caller's training_config dict is left untouched.

    AIREstimatorCV reuses one config dict across many cross-fitting runs, so the
    internal setdefault calls must operate on a copy.
    """
    data = _make_simple_data(n=200, p=3)
    t_grid = jnp.linspace(0, 5.0, 21)

    config = {"epochs": 5, "verbose": 0}
    cross_fit_nuisances(data, t_grid, n_folds=2, training_config=config)

    assert config == {"epochs": 5, "verbose": 0}, (
        f"caller's config was mutated: {config}"
    )


def test_admin_as_censored_requires_t_end():
    """The correction cannot identify administrative rows without t_end."""
    data = _make_simple_data(n=100, p=3)
    t_grid = jnp.linspace(0, 5.0, 21)

    with pytest.raises(ValueError, match="t_end"):
        cross_fit_nuisances(
            data, t_grid, n_folds=2, admin_as_censored=True,
            training_config={"epochs": 2, "verbose": 0},
        )


def test_admin_as_censored_changes_only_censoring():
    """The correction moves S_cens and leaves the event models alone.

    The event models see the same rows and the same indicator either way, and
    they are built before the censoring model within each fold, so their Keras
    weight initializations are drawn from the same global RNG states. Any change
    there would mean the flag had leaked out of the censoring fit.
    """
    t_end = 5.0
    data = _make_simple_data(n=300, p=3, t_end=t_end)
    t_grid = jnp.linspace(0, t_end, 21)
    admin = np.asarray(data.Z) >= t_end - 1e-6
    assert admin.sum() > 10, "test data has too few administrative rows to be a test"

    train_cfg = {"epochs": 5, "verbose": 0}
    import keras

    out = {}
    for flag in (False, True):
        keras.utils.set_random_seed(0)
        out[flag] = cross_fit_nuisances(
            data, t_grid, n_folds=2, admin_as_censored=flag, t_end=t_end,
            training_config=train_cfg,
        )

    for field in ("S_event_0_grid", "S_event_1_grid", "e_hat"):
        np.testing.assert_allclose(
            getattr(out[False], field), getattr(out[True], field),
            err_msg=f"{field} changed, so the flag is not censoring-only",
        )
    assert not jnp.allclose(out[False].S_cens_0_grid, out[True].S_cens_0_grid)


def test_censoring_config_overrides_weibull_config():
    """`censoring_config` applies to the censoring models only."""
    data = _make_simple_data(n=300, p=3)
    t_grid = jnp.linspace(0, 5.0, 21)

    nuisances = cross_fit_nuisances(
        data, t_grid, n_folds=2,
        weibull_config={"hidden_sizes": (8,)},
        censoring_config={"hidden_sizes": (8,), "shared_shape": True},
        training_config={"epochs": 5, "verbose": 0},
    )

    # A shared shape cannot make S_cens degenerate or non-monotone.
    S = np.asarray(nuisances.S_cens_0_grid)
    assert not np.any(np.isnan(S))
    assert np.all(np.diff(S, axis=1) <= 1e-6)
