"""Tests for PropensityMLP."""

import numpy as np
import pytest


def test_propensity_output_range():
    """Predictions are in (0, 1)."""
    from air_cate.nuisance.propensity import PropensityMLP

    rng = np.random.default_rng(0)
    n, p = 500, 5
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p)
    prob = 1 / (1 + np.exp(-X @ beta))
    W = rng.binomial(1, prob).astype(np.float32)

    model = PropensityMLP(hidden_sizes=[32, 16])
    model.fit(X.astype(np.float32), W, epochs=20, verbose=0)
    preds = model.predict(X.astype(np.float32))

    assert preds.shape == (n,)
    assert np.all(preds > 0) and np.all(preds < 1)


def test_propensity_calibration():
    """Brier score is reasonable on logistic synthetic data."""
    from air_cate.nuisance.propensity import PropensityMLP

    rng = np.random.default_rng(42)
    n, p = 2000, 5
    X = rng.standard_normal((n, p)).astype(np.float32)
    beta = np.array([0.5, -0.3, 0.0, 0.0, 0.0], dtype=np.float32)
    prob = 1 / (1 + np.exp(-X @ beta))
    W = rng.binomial(1, prob).astype(np.float32)

    model = PropensityMLP(hidden_sizes=[32, 16])
    model.fit(X, W, epochs=50, verbose=0)
    preds = model.predict(X)

    brier = np.mean((preds - W) ** 2)
    # Brier score for a well-calibrated model should be close to E[p(1-p)]
    optimal_brier = np.mean(prob * (1 - prob))
    assert brier < optimal_brier + 0.05, f"Brier score {brier:.4f} too high"
