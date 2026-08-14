"""Tests for the grf causal survival forest baseline.

Every test needing R is skipped when Rscript or the grf package is missing, so the
suite still runs on a machine without them.
"""

import numpy as np
import pytest

from air_cate.baselines_csf import CausalSurvivalForest, grf_available
from conftest import WeibullDGP

# Probed once at collection rather than per test -- each probe spawns an R process.
requires_grf = pytest.mark.skipif(
    not grf_available(), reason="R with the grf package is not available"
)


@requires_grf
def test_output_shape_and_grid():
    """predict_tau returns (n, M+1) and the grid attributes are populated."""
    X, W, Z, Delta, t_end = WeibullDGP.generate(n=300, rng_seed=0)
    est = CausalSurvivalForest(M=5, num_trees=200)
    est.fit(X, W, Z, Delta, t_end)

    tau = est.predict_tau(X[:10])
    assert tau.shape == (10, 6)  # M+1 = 6
    assert est.time_grid_ is not None
    assert est.delta_ == pytest.approx(t_end / 5)
    assert np.all(np.isfinite(tau))


@requires_grf
def test_tau_is_zero_at_baseline():
    """tau(0|x) = 0 exactly: S_1(0) = S_0(0) = 1, so the first column is imposed."""
    X, W, Z, Delta, t_end = WeibullDGP.generate(n=300, rng_seed=0)
    est = CausalSurvivalForest(M=5, num_trees=200)
    est.fit(X, W, Z, Delta, t_end)

    tau = est.predict_tau(X[:10])
    assert np.all(tau[:, 0] == 0.0)


@requires_grf
def test_diagnostics_populated():
    """The failure count and out-of-range fraction are recorded for reporting."""
    X, W, Z, Delta, t_end = WeibullDGP.generate(n=300, rng_seed=2)
    est = CausalSurvivalForest(M=5, num_trees=200)
    est.fit(X, W, Z, Delta, t_end)
    est.predict_tau(X[:20])

    assert 0 <= est.n_failed_horizons_ <= 5
    assert 0.0 <= est.frac_out_of_range_ <= 1.0


@requires_grf
def test_tau_roughly_bounded():
    """Predictions should sit near [-1, 1] without being clipped there.

    The bound is deliberately loose: unlike S1 - S0 from the forest baselines, an
    orthogonalized causal-survival-forest estimate carries no range guarantee, and
    this module does not clip. A gross excursion still signals a wiring bug.
    """
    X, W, Z, Delta, t_end = WeibullDGP.generate(n=500, rng_seed=1)
    est = CausalSurvivalForest(M=10, num_trees=300)
    est.fit(X, W, Z, Delta, t_end)

    tau = est.predict_tau(X[:50])
    assert np.abs(tau).max() < 2.0


@requires_grf
def test_predict_before_fit_raises():
    X = np.zeros((5, 3))
    with pytest.raises(RuntimeError, match="fit must be called"):
        CausalSurvivalForest(M=5).predict_tau(X)


@requires_grf
def test_covariate_mismatch_raises():
    X, W, Z, Delta, t_end = WeibullDGP.generate(n=200, rng_seed=0)
    est = CausalSurvivalForest(M=5, num_trees=100)
    est.fit(X, W, Z, Delta, t_end)

    with pytest.raises(ValueError, match="columns"):
        est.predict_tau(X[:10, :2])


@requires_grf
def test_seed_makes_predictions_reproducible():
    """An explicit seed pins grf, so two fits on one sample agree exactly."""
    X, W, Z, Delta, t_end = WeibullDGP.generate(n=300, rng_seed=0)

    taus = []
    for _ in range(2):
        est = CausalSurvivalForest(M=5, num_trees=200, seed=1234)
        est.fit(X, W, Z, Delta, t_end)
        taus.append(est.predict_tau(X[:20]))

    np.testing.assert_allclose(taus[0], taus[1])


@requires_grf
@pytest.mark.slow
def test_every_method_shares_one_time_grid():
    """CSF is the only method that refits per horizon; it must still land on the
    common grid.

    The simulation study compares iMSE, the derivative MSE and the extrema count
    across methods by summing over grid columns, so a method whose grid differed --
    even by an endpoint convention -- would make every one of those comparisons
    meaningless while every individual fit still looked healthy. This pins the
    requirement as a test rather than an assumption: one draw, every estimator,
    identical `time_grid_` and `delta_`.
    """
    from air_cate.baselines import SurvivalForestSLearner, SurvivalForestTLearner
    from air_cate.estimator import AIREstimator
    from air_cate.utils import build_time_grid

    M = 5
    X, W, Z, Delta, t_end = WeibullDGP.generate(n=300, rng_seed=0)
    expected_grid, expected_delta = build_time_grid(t_end, M)

    estimators = {
        "CSF": CausalSurvivalForest(M=M, num_trees=100),
        "RSF-T": SurvivalForestTLearner(M=M, n_estimators=20),
        "RSF-S": SurvivalForestSLearner(M=M, n_estimators=20),
        # Trained to nothing on purpose: this asserts the grid, not the fit.
        "AIR": AIREstimator(
            K=2, M=M, n_folds=2, encoder_hidden_sizes=(8,),
            nuisance_training_config={"epochs": 2, "verbose": 0},
            cate_epochs=2,
        ),
    }

    for name, est in estimators.items():
        est.fit(X, W, Z, Delta, t_end, verbose=0)
        np.testing.assert_allclose(
            np.asarray(est.time_grid_), np.asarray(expected_grid),
            err_msg=f"{name} does not use the shared time grid",
        )
        assert est.delta_ == pytest.approx(float(expected_delta)), name


@requires_grf
@pytest.mark.slow
def test_marginal_curve_tracks_truth():
    """The averaged tau_hat(t) curve should track the averaged truth.

    This is the check that the estimand is wired correctly end to end: a
    misindexed horizon or an RMST-vs-survival-probability mixup would leave the
    marginal curve visibly off even though every shape assertion still passed.
    """
    X, W, Z, Delta, t_end = WeibullDGP.generate(n=2000, rng_seed=0)
    est = CausalSurvivalForest(M=10, num_trees=500)
    est.fit(X, W, Z, Delta, t_end)

    X_test = WeibullDGP.generate(n=500, rng_seed=7)[0]
    tau_hat = est.predict_tau(X_test)
    tau_true = WeibullDGP.true_tau(np.asarray(est.time_grid_), X_test)

    mean_hat = tau_hat.mean(axis=0)
    mean_true = tau_true.mean(axis=0)
    assert np.abs(mean_hat - mean_true).max() < 0.06
