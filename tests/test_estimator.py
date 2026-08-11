"""Integration tests for AIREstimator."""

import math

import numpy as np
import pytest

from conftest import WeibullDGP


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

    import keras  # after air_cate, which sets KERAS_BACKEND

    X, W, Z, Delta, t_end = null_dgp_data

    # Seed globally, not only through `rng_seed`: that argument reaches just the CATE
    # stage, so the propensity MLP and the four AFT initializations would still be
    # redrawn every run and the assertions below would be a lottery rather than a test.
    keras.utils.set_random_seed(0)

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
        rng_seed=0,
    )
    est.fit(X, W, Z, Delta, t_end, verbose=0)

    rng = np.random.default_rng(99)
    X_test = rng.uniform(-1, 1, (300, X.shape[1])).astype(np.float32)
    tau_abs = np.abs(est.predict_tau(X_test))

    # Assert on the RMS rather than the max. max over 300 x 51 = 15300 predictions is
    # an extreme-value statistic, and X_test is uniform on [-1, 1]^d, so its corners
    # sit outside the training support -- the max mostly measures extrapolation there,
    # not whether the estimated CATE is near zero. Measured over 15 estimator seeds at
    # n=1000 (all with true tau == 0, so every nonzero value is artefact):
    #
    #              mean      sd   worst   mean+4sd
    #   max       0.187   0.077   0.384      0.495
    #   rms       0.059   0.019   0.087      0.135
    #
    # A historical `max < 0.2` assertion therefore failed on 6 of those 15 draws;
    # holding onto max would mean a threshold near 0.5, i.e. half of tau's entire
    # range. `rms < 0.15` sits ~4.5 sd above the mean and is the stricter test.
    rms = float(np.sqrt((tau_abs**2).mean()))
    max_abs = float(tau_abs.max())
    print(f"Null DGP rms |tau_hat|: {rms:.4f} (max {max_abs:.4f})")
    assert rms < 0.15, f"rms |tau_hat| = {rms:.4f}, expected < 0.15 under null"
    # Secondary guard: a spurious effect saturating toward tau's +-1 bound at a few
    # points would barely move an RMS taken over 15300 of them.
    assert max_abs < 0.5, f"max |tau_hat| = {max_abs:.4f}, expected < 0.5 under null"


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
    assert est.history_["val_loss"] is None, "cate_val_frac defaults to 0, so no split"
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


def test_cate_val_frac_makes_early_stopping_real():
    """A validation split must actually reach train_cate and drive the monitor.

    At the default cate_val_frac=0 the monitored quantity is the *training* R-loss,
    which falls monotonically, so `patience` never fires and every fit runs to
    cate_epochs. That inert stopping rule cost ~35% in iMSE on the IST-3 DGP (oracle
    nuisances, 20 paired seeds at n=2000: 0.192 at 200 epochs vs 0.142 at 100). This
    pins the wiring that fixes it: a split is drawn, val_loss is recorded, and the
    monitor resolves to "val" rather than silently staying on "train".
    """
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
        patience=None,  # keep all 10 epochs so the histories stay comparable
        cate_val_frac=0.25,
        rng_seed=0,
    )
    est.fit(X, W, Z, Delta, t_end=5.0, verbose=0)

    assert est.history_["monitor"] == "val", "the split did not reach train_cate"
    assert est.history_["val_loss"] is not None
    assert len(est.history_["val_loss"]) == 10
    assert np.all(np.isfinite(est.history_["val_loss"]))
    # Restart ranking must switch to the held-out score, not stay on the train loss.
    assert est.restart_losses_ == [est.history_["best_val_loss"]]
    # Predictions still cover every subject: only the CATE *fit* is subsetted.
    assert est.predict_tau(X).shape == (n, 11)


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_cate_val_frac_out_of_range_raises(bad):
    """Reject at construction, not with an opaque indexing error mid-fit."""
    from air_cate.estimator import AIREstimator

    with pytest.raises(ValueError, match="cate_val_frac"):
        AIREstimator(cate_val_frac=bad)


def test_cate_val_frac_too_small_for_n_raises():
    """A fraction that rounds to zero held-out rows must fail, not silently no-op.

    Without the guard, n_val = 0 yields an empty validation set whose R-loss is NaN,
    which the refuse-NaN-model check in train_cate would then report as a training
    failure -- a confusing symptom several layers from the actual cause.
    """
    from air_cate.estimator import AIREstimator

    rng = np.random.default_rng(0)
    n, p = 20, 3
    X = rng.uniform(-1, 1, (n, p)).astype(np.float32)
    W = rng.binomial(1, 0.5, n).astype(np.float32)
    Z = rng.exponential(2.0, n).astype(np.float32)
    Delta = rng.binomial(1, 0.7, n).astype(np.float32)

    est = AIREstimator(
        K=2, M=5, n_folds=2, eps=0.01,
        encoder_hidden_sizes=(8,),
        nuisance_training_config={"epochs": 2, "verbose": 0},
        cate_epochs=2, patience=None,
        cate_val_frac=0.01,  # 0.01 * 20 = 0.2 -> rounds to 0
        rng_seed=0,
    )
    with pytest.raises(ValueError, match="leaves 0 validation"):
        est.fit(X, W, Z, Delta, t_end=5.0, verbose=0)


def test_cate_refit_full_matches_budget_in_steps_not_epochs():
    """The refit must reproduce the selected budget in optimizer *steps*.

    The stopping point is measured with ceil(n_fit / B) updates per epoch; the refit
    sees ceil(n / B), a factor 1/(1 - cate_val_frac) more. Reusing the epoch count
    verbatim would train ~25% longer at cate_val_frac=0.2 than the held-out block ever
    endorsed -- reintroducing the over-training the split exists to prevent. Batch 32
    with n=200 makes the two rates differ (5 vs 7 steps/epoch) so the conversion is
    actually exercised; at the default batch 256 both round to 1 and the test is vacuous.
    """
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
        cate_batch_size=32,
        patience=None,  # spend all 10 epochs, so best_epoch is chosen on merit
        cate_val_frac=0.25,
        cate_refit_full=True,
        rng_seed=0,
    )
    est.fit(X, W, Z, Delta, t_end=5.0, verbose=0)

    n_fit = n - int(round(0.25 * n))
    steps_sel = math.ceil(n_fit / 32)      # 150 -> 5
    steps_full = math.ceil(n / 32)         # 200 -> 7
    assert steps_sel < steps_full, "batch size chosen so the two rates differ"
    selected_epochs = est.history_["best_epoch"] + 1
    expected = max(1, round(selected_epochs * steps_sel / steps_full))
    assert est.refit_epochs_ == expected

    # history_ still describes the *selection* fit -- that is what plot_convergence
    # draws, and what earned the epoch count.
    assert est.history_["monitor"] == "val"
    # The refit holds nothing out, so it has no val curve to monitor.
    assert est.refit_history_ is not None
    assert est.refit_history_ is not est.history_
    assert est.refit_history_["monitor"] == "train"
    assert est.refit_history_["val_loss"] is None
    assert len(est.refit_history_["train_loss"]) == est.refit_epochs_
    assert np.all(np.isfinite(est.refit_history_["train_loss"]))
    assert est.predict_tau(X).shape == (n, 11)


def test_cate_refit_full_leaves_attributes_none_when_off():
    """Off by default: the refit attributes exist but stay None, so callers can test
    `refit_history_ is None` instead of hasattr."""
    from air_cate.estimator import AIREstimator

    est = AIREstimator()
    assert est.refit_history_ is None
    assert est.refit_epochs_ is None


def test_cate_refit_full_without_split_raises():
    """Reject at construction: with nothing held out there is no honest stopping point
    to refit at, and the failure should not wait until the first fit."""
    from air_cate.estimator import AIREstimator

    with pytest.raises(ValueError, match="cate_refit_full"):
        AIREstimator(cate_refit_full=True)  # cate_val_frac defaults to 0.0


def test_plot_convergence_before_fit_raises():
    """The error names the fix, rather than surfacing a None dereference."""
    from air_cate.estimator import AIREstimator

    with pytest.raises(RuntimeError, match="Call fit"):
        AIREstimator().plot_convergence()
