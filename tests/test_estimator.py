"""Integration tests for AIREstimator."""

import numpy as np
import pytest

from conftest import WeibullDGP, NullDGP


@pytest.mark.slow
def test_estimator_weibull_dgp(weibull_dgp_data):
    """AIREstimator recovers heterogeneous CATE from Weibull DGP."""
    from air_cate.estimator import AIREstimator

    X, W, Z, Delta, t_end = weibull_dgp_data

    est = AIREstimator(
        K=4,
        M=50,
        n_folds=3,
        eps=0.01,
        encoder_hidden_sizes=(32, 16),
        nuisance_training_config={"epochs": 50, "verbose": 0},
        cate_epochs=150,
        cate_lr=1e-3,
        cate_batch_size=256,
    )
    est.fit(X, W, Z, Delta, t_end, verbose=0)

    # Predict on a test set
    rng = np.random.default_rng(99)
    X_test = rng.uniform(-1, 1, (500, X.shape[1])).astype(np.float32)

    tau_hat = est.predict_tau(X_test)
    t_grid = np.array(est.time_grid_)
    tau_true = WeibullDGP.true_tau(t_grid, X_test)

    # Integrated MSE
    delta = est.delta_
    imse = np.mean(np.sum(delta * (tau_hat - tau_true) ** 2, axis=1))
    print(f"Weibull DGP integrated MSE: {imse:.4f}")

    # Structural constraints
    assert tau_hat.shape == (500, len(t_grid))
    np.testing.assert_allclose(tau_hat[:, 0], 0.0, atol=1e-4, err_msg="tau(0) should be 0")
    assert np.all(tau_hat >= -1.0 - 1e-4) and np.all(tau_hat <= 1.0 + 1e-4), "tau out of [-1, 1]"

    # MSE check — generous threshold for moderate sample size
    assert imse < 0.3, f"Integrated MSE {imse:.4f} too high (threshold 0.3)"


@pytest.mark.slow
def test_estimator_null_dgp(null_dgp_data):
    """AIREstimator produces near-zero CATE under null treatment effect."""
    from air_cate.estimator import AIREstimator

    X, W, Z, Delta, t_end = null_dgp_data

    est = AIREstimator(
        K=2,
        M=50,
        n_folds=3,
        eps=0.01,
        encoder_hidden_sizes=(32, 16),
        nuisance_training_config={"epochs": 50, "verbose": 0},
        cate_epochs=100,
        cate_lr=1e-3,
        cate_batch_size=256,
    )
    est.fit(X, W, Z, Delta, t_end, verbose=0)

    rng = np.random.default_rng(99)
    X_test = rng.uniform(-1, 1, (300, X.shape[1])).astype(np.float32)
    tau_hat = est.predict_tau(X_test)

    max_abs = np.max(np.abs(tau_hat))
    print(f"Null DGP max |tau_hat|: {max_abs:.4f}")
    assert max_abs < 0.2, f"max |tau_hat| = {max_abs:.4f}, expected < 0.2 under null"


@pytest.mark.slow
def test_estimator_gamma_output(weibull_dgp_data):
    """predict_gamma returns cumulative tau and has correct shape."""
    from air_cate.estimator import AIREstimator

    X, W, Z, Delta, t_end = weibull_dgp_data

    est = AIREstimator(
        K=2, M=30, n_folds=2, eps=0.01,
        encoder_hidden_sizes=(16,),
        nuisance_training_config={"epochs": 20, "verbose": 0},
        cate_epochs=50,
    )
    est.fit(X, W, Z, Delta, t_end, verbose=0)

    X_test = X[:50]
    gamma = est.predict_gamma(X_test)
    tau = est.predict_tau(X_test)

    assert gamma.shape == tau.shape
    # Gamma should be cumulative sum of delta * tau
    expected_gamma = np.cumsum(est.delta_ * tau, axis=1)
    np.testing.assert_allclose(gamma, expected_gamma, atol=1e-5)


def test_estimator_predict_shapes():
    """Quick smoke test for predict shapes (no slow marker)."""
    from air_cate.estimator import AIREstimator

    rng = np.random.default_rng(0)
    n, p = 200, 5
    X = rng.uniform(-1, 1, (n, p)).astype(np.float32)
    W = rng.binomial(1, 0.5, n).astype(np.float32)
    Z = rng.exponential(2.0, n).astype(np.float32)
    Delta = rng.binomial(1, 0.7, n).astype(np.float32)

    est = AIREstimator(
        K=2, M=10, n_folds=2, eps=0.01,
        encoder_hidden_sizes=(16,),
        nuisance_training_config={"epochs": 5, "verbose": 0},
        cate_epochs=10,
    )
    est.fit(X, W, Z, Delta, t_end=5.0, verbose=0)

    tau = est.predict_tau(X[:10])
    gamma = est.predict_gamma(X[:10])

    assert tau.shape == (10, 11)  # M=10, so M+1=11 grid points
    assert gamma.shape == (10, 11)


def test_estimator_records_history():
    """fit() exposes the per-epoch training loss trajectory on history_."""
    from air_cate.estimator import AIREstimator

    rng = np.random.default_rng(0)
    n, p = 200, 5
    X = rng.uniform(-1, 1, (n, p)).astype(np.float32)
    W = rng.binomial(1, 0.5, n).astype(np.float32)
    Z = rng.exponential(2.0, n).astype(np.float32)
    Delta = rng.binomial(1, 0.7, n).astype(np.float32)

    est = AIREstimator(
        K=2, M=10, n_folds=2, eps=0.01,
        encoder_hidden_sizes=(16,),
        nuisance_training_config={"epochs": 5, "verbose": 0},
        cate_epochs=10,
        patience=None,  # train the full 10 epochs
        rng_seed=0,
    )
    est.fit(X, W, Z, Delta, t_end=5.0, verbose=0)

    assert est.history_ is not None
    assert len(est.history_["train_loss"]) == 10
    assert est.history_["val_loss"] is None, "no validation split in AIREstimator"
    assert est.history_["monitor"] == "train"
    assert np.all(np.isfinite(est.history_["train_loss"]))

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = est.plot_convergence()
    try:
        curve = next(
            line for line in ax.lines if len(line.get_ydata()) == 10
        )
        np.testing.assert_allclose(
            curve.get_ydata(), est.history_["train_loss"], rtol=1e-12
        )
        assert ax.get_ylabel() == "Training AIR CATE loss"
    finally:
        plt.close(fig)


def test_plot_convergence_before_fit_raises():
    """The error names the fix, rather than surfacing a None dereference."""
    from air_cate.estimator import AIREstimator

    with pytest.raises(RuntimeError, match="Call fit"):
        AIREstimator().plot_convergence()
