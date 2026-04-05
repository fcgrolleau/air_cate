"""Tests for survival forest baseline estimators."""

import numpy as np

from conftest import WeibullDGP


def test_t_learner_output_shape():
    """T-learner predict_tau returns correct shape and stores grid attributes."""
    from air_cate.baselines import SurvivalForestTLearner

    X, W, Z, Delta, t_end = WeibullDGP.generate(n=200, rng_seed=0)
    est = SurvivalForestTLearner(M=10, n_estimators=20, min_samples_leaf=20)
    est.fit(X, W, Z, Delta, t_end)

    tau = est.predict_tau(X[:10])
    assert tau.shape == (10, 11)  # M+1 = 11
    assert est.time_grid_ is not None
    assert est.delta_ is not None


def test_s_learner_output_shape():
    """S-learner predict_tau returns correct shape."""
    from air_cate.baselines import SurvivalForestSLearner

    X, W, Z, Delta, t_end = WeibullDGP.generate(n=200, rng_seed=0)
    est = SurvivalForestSLearner(M=10, n_estimators=20, min_samples_leaf=20)
    est.fit(X, W, Z, Delta, t_end)

    tau = est.predict_tau(X[:10])
    assert tau.shape == (10, 11)


def test_t_learner_tau_bounded():
    """T-learner tau = S1 - S0 must be in [-1, 1]."""
    from air_cate.baselines import SurvivalForestTLearner

    X, W, Z, Delta, t_end = WeibullDGP.generate(n=300, rng_seed=1)
    est = SurvivalForestTLearner(M=20, n_estimators=50, min_samples_leaf=15)
    est.fit(X, W, Z, Delta, t_end)

    tau = est.predict_tau(X[:50])
    assert np.all(tau >= -1.0 - 1e-6)
    assert np.all(tau <= 1.0 + 1e-6)
