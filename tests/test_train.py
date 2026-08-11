"""Tests for the shared CATE training loop."""

import jax.numpy as jnp
import numpy as np
import pytest

from air_cate.cate_model.mixture import CATEMixture
from air_cate.cate_model.train import train_cate
from air_cate.r_loss import r_loss

N = 50
P = 3
M = 8
# Deliberately does not divide N, so every epoch ends in a short batch of 18 rows.
BATCH_SIZE = 32


def _heavy_tailed_inputs(rng_seed=0):
    """Small problem whose per-subject loss has a few dominant contributions.

    The R-loss is a mean over subjects, so a heavy tail is what makes any
    mis-weighting of the minibatch average visible.
    """
    rng = np.random.default_rng(rng_seed)
    X = rng.uniform(-1, 1, (N, P)).astype(np.float32)
    S_marg = np.tile(np.linspace(1.0, 0.2, M + 1, dtype=np.float32), (N, 1))

    # Most subjects contribute little; five contribute ~20x more.
    scale = np.full(N, 0.1, dtype=np.float32)
    scale[:5] = 2.0
    Y_tilde = S_marg + scale[:, None] * rng.normal(size=(N, M + 1)).astype(np.float32)

    W = rng.binomial(1, 0.5, N).astype(np.float32)
    e_clipped = np.full(N, 0.5, dtype=np.float32)
    time_grid = jnp.asarray(np.linspace(0.0, 4.0, M + 1), dtype=jnp.float32)
    delta = 4.0 / M
    return X, Y_tilde, S_marg, W, e_clipped, time_grid, delta


def _train_frozen(rng_seed, epochs=3):
    """Run train_cate with lr=0, so the parameters never move.

    Every epoch then evaluates the *same* model, and the only thing that differs
    between epochs -- or between rng_seeds -- is the minibatch shuffle.
    """
    # Imported here, not at module scope: air_cate.__init__ sets KERAS_BACKEND=jax,
    # and importing keras before that happens picks up the TensorFlow backend.
    import keras

    X, Y_tilde, S_marg, W, e_clipped, time_grid, delta = _heavy_tailed_inputs()

    # Seed with a CONSTANT, not rng_seed: the model's weights come from Keras's
    # global RNG, so without this the two runs below would differ by their weights
    # as well as their shuffle, and the comparison would test nothing.
    keras.utils.set_random_seed(0)
    model = CATEMixture(K=2, encoder_hidden_sizes=(4,))

    history = train_cate(
        model, X=X, Y_tilde=Y_tilde, S_marg=S_marg, W=W, e_clipped=e_clipped,
        time_grid=time_grid, delta=delta,
        epochs=epochs, lr=0.0, weight_decay=0.0, batch_size=BATCH_SIZE,
        patience=None, rng_seed=rng_seed, verbose=0,
    )

    tau = model({"X": jnp.asarray(X), "time_grid": time_grid})
    full_data_loss = float(
        r_loss(
            tau,
            jnp.asarray(Y_tilde), jnp.asarray(S_marg),
            jnp.asarray(W), jnp.asarray(e_clipped), delta,
        )
    )
    return history, full_data_loss


def test_epoch_train_loss_equals_the_full_data_loss():
    """Size-weighting the minibatch average makes each subject count exactly once.

    Regression guard: averaging batch means with equal weight over-weights the
    final short batch (18 of 50 rows here would carry half the weight), which
    biases the epoch loss away from the loss actually being minimized.
    """
    history, full_data_loss = _train_frozen(rng_seed=0)

    assert N % BATCH_SIZE != 0, "the test needs a short final batch to be meaningful"
    for epoch_loss in history["train_loss"]:
        assert epoch_loss == np.float32(full_data_loss).item() or abs(
            epoch_loss - full_data_loss
        ) < 1e-4, f"{epoch_loss} != {full_data_loss}"


def test_epoch_train_loss_is_shuffle_invariant():
    """With frozen parameters the epoch loss must not depend on the shuffle.

    This is the defect that made the convergence grid unreadable: reshuffling moved
    heavy-tailed subjects in and out of the over-weighted final batch, injecting
    noise an order of magnitude larger than the real epoch-to-epoch movement.
    """
    history_a, _ = _train_frozen(rng_seed=0)
    history_b, _ = _train_frozen(rng_seed=12345)

    # Across epochs within a run (each epoch is a fresh permutation) ...
    assert np.ptp(history_a["train_loss"]) < 1e-4
    # ... and across runs with entirely different shuffles.
    np.testing.assert_allclose(
        history_a["train_loss"], history_b["train_loss"], atol=1e-4
    )


def test_a_non_finite_loss_raises_instead_of_returning_a_nan_model():
    """Training must fail loudly rather than hand back a model full of NaN.

    The failure this guards against was silent and therefore expensive: a NaN
    reaching the weights left `best_weights` unset, the `is not None` check then
    skipped weight restoration, and the caller received a model whose predictions
    were entirely NaN. Scored in a simulation that tolerates NaN when plotting, it
    surfaced only as iMSE = nan in roughly 10% of fits -- with no warning at all,
    and it would have quietly contaminated a 999-seed run.
    """
    import keras

    X, Y_tilde, S_marg, W, e_clipped, time_grid, delta = _heavy_tailed_inputs()

    # One NaN pseudo-outcome is enough: the R-loss sums over subjects, so it makes
    # every epoch's loss non-finite from the very first batch.
    Y_tilde = Y_tilde.copy()
    Y_tilde[0, 0] = np.nan

    keras.utils.set_random_seed(0)
    model = CATEMixture(K=2, encoder_hidden_sizes=(4,))

    with pytest.raises(RuntimeError, match="no usable weights"):
        train_cate(
            model, X=X, Y_tilde=Y_tilde, S_marg=S_marg, W=W, e_clipped=e_clipped,
            time_grid=time_grid, delta=delta,
            epochs=3, lr=1e-3, weight_decay=0.0, batch_size=BATCH_SIZE,
            patience=None, rng_seed=0, verbose=0,
        )
