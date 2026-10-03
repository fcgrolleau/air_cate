"""Tests for the auxiliary loss hook on the CATE training loop.

The hook exists because there was previously no way to regularize the CATE stage:
`CATEMixture.compute_loss` is never called (`train_cate` builds its own objective),
and `stateless_call` is invoked without `return_losses=True` while `model.losses` is
never read, so an `add_loss` penalty is silently dropped. A penalty that is accepted
and then ignored is the worst possible failure here -- every penalized run would come
back looking inert, and the conclusion drawn would be about the penalty rather than
about the plumbing. `test_a_weight_dependent_penalty_enters_the_gradient` is the test
that distinguishes those two cases, and it is the reason this file exists.
"""

import jax.numpy as jnp
import numpy as np

from air_cate.cate_model.mixture import CATEMixture
from air_cate.cate_model.train import train_cate

N = 60
P = 4
M = 8
N_VAL = 20
# Deliberately does not divide N - N_VAL, so every epoch ends in a short batch and the
# size-weighted aux average is exercised on unequal batches.
BATCH_SIZE = 32
CONST = 7.0

_ARRAY_KEYS = ("X", "Y_tilde", "S_marg", "W", "e_clipped")


def _inputs(rng_seed=0):
    """A small problem with a non-degenerate R-loss."""
    rng = np.random.default_rng(rng_seed)
    X = rng.uniform(-1, 1, (N, P)).astype(np.float32)
    S_marg = np.tile(np.linspace(1.0, 0.2, M + 1, dtype=np.float32), (N, 1))
    Y_tilde = S_marg + 0.3 * rng.normal(size=(N, M + 1)).astype(np.float32)
    W = rng.binomial(1, 0.5, N).astype(np.float32)
    e_clipped = np.full(N, 0.5, dtype=np.float32)
    time_grid = jnp.asarray(np.linspace(0.0, 4.0, M + 1), dtype=jnp.float32)
    return dict(
        X=X, Y_tilde=Y_tilde, S_marg=S_marg, W=W, e_clipped=e_clipped,
        time_grid=time_grid, delta=4.0 / M,
    )


def _fit(aux_loss_fn=None, aux_in_monitor=False, epochs=6, lr=1e-2, seed=0):
    """One train_cate run with a validation block, so `monitor` resolves to 'val'."""
    # Imported here, not at module scope: air_cate.__init__ sets KERAS_BACKEND=jax,
    # and importing keras before that happens picks up the TensorFlow backend.
    import keras

    d = _inputs()
    val = {k: d[k][:N_VAL] for k in _ARRAY_KEYS}
    tr = {k: d[k][N_VAL:] for k in _ARRAY_KEYS}

    # Constant seed: the weights come from Keras's global RNG, so without this two
    # runs would differ by their initialization as well as by the hook.
    keras.utils.set_random_seed(seed)
    model = CATEMixture(K=2, encoder_hidden_sizes=(4,), normalize_time=True)

    history = train_cate(
        model, X=tr["X"], Y_tilde=tr["Y_tilde"], S_marg=tr["S_marg"],
        W=tr["W"], e_clipped=tr["e_clipped"],
        time_grid=d["time_grid"], delta=d["delta"],
        epochs=epochs, lr=lr, weight_decay=0.0, batch_size=BATCH_SIZE,
        validation_data=val, patience=None, rng_seed=0, verbose=0,
        aux_loss_fn=aux_loss_fn, aux_in_monitor=aux_in_monitor,
    )
    tau = np.asarray(model({"X": jnp.asarray(d["X"]), "time_grid": d["time_grid"]}))
    return history, model, tau


def _constant_penalty(model, trainable_vars, non_trainable_vars, X, time_grid, tau):
    """A penalty with zero gradient, so it cannot move the weight trajectory."""
    return jnp.asarray(CONST, dtype=tau.dtype)


def _tau_ridge(coef):
    """A weight-DEPENDENT penalty: shrinks tau, so it must change the trajectory."""

    def penalty(model, trainable_vars, non_trainable_vars, X, time_grid, tau):
        return jnp.asarray(coef, dtype=tau.dtype) * jnp.mean(tau ** 2)

    return penalty


def test_aux_none_leaves_the_new_keys_inert():
    """With no hook, the added history keys must not change any existing behaviour."""
    history, _, _ = _fit(aux_loss_fn=None)

    assert history["aux_in_monitor"] is False
    assert history["train_aux"] == [0.0] * len(history["train_loss"])
    assert history["val_aux"] == [0.0] * len(history["val_loss"])

    # The monitored series is the one best_epoch is chosen on. Unhooked, it must be
    # the pure validation loss, element for element.
    assert history["monitor"] == "val"
    np.testing.assert_array_equal(history["monitored_loss"], history["val_loss"])
    assert history["best_epoch"] == int(np.argmin(history["monitored_loss"]))


def test_a_constant_penalty_is_reported_without_touching_the_pure_losses():
    """A zero-gradient penalty must be reported, and must change nothing else.

    This pins the division of labour: `train_aux`/`val_aux` carry the penalty while
    `train_loss`/`val_loss` stay pure R-loss, which is what lets downstream readers
    (plot_convergence, save_refit_history) keep working and what makes runs with
    different penalty strengths comparable at all.
    """
    plain, _, tau_plain = _fit(aux_loss_fn=None)
    hooked, _, tau_hooked = _fit(aux_loss_fn=_constant_penalty, aux_in_monitor=True)

    # A constant has zero gradient, so the optimizer followed the same trajectory.
    # Not asserted bitwise: the None and hooked paths are separately jitted functions,
    # so XLA may legitimately order the same arithmetic differently.
    np.testing.assert_allclose(hooked["train_loss"], plain["train_loss"], rtol=1e-5)
    np.testing.assert_allclose(hooked["val_loss"], plain["val_loss"], rtol=1e-5)
    np.testing.assert_allclose(tau_hooked, tau_plain, rtol=1e-5, atol=1e-6)

    # The penalty is reported, on both blocks.
    np.testing.assert_allclose(hooked["train_aux"], CONST, rtol=1e-6)
    np.testing.assert_allclose(hooked["val_aux"], CONST, rtol=1e-6)

    # ... and reaches the monitored series, since aux_in_monitor=True.
    assert hooked["aux_in_monitor"] is True
    np.testing.assert_allclose(
        hooked["monitored_loss"],
        np.asarray(hooked["val_loss"]) + CONST,
        rtol=1e-6,
    )


def test_aux_in_monitor_controls_whether_the_penalty_reaches_best_epoch():
    """aux_in_monitor is the switch between the two monitored series.

    It matters because `restore_best` reads the monitored series: optimizing
    R + penalty while monitoring pure R hands back the least-penalized epoch, which
    silently undoes the penalty.
    """
    off, _, _ = _fit(aux_loss_fn=_constant_penalty, aux_in_monitor=False)
    on, _, _ = _fit(aux_loss_fn=_constant_penalty, aux_in_monitor=True)

    # Off: the penalty is still computed and reported, but excluded from monitoring.
    assert off["aux_in_monitor"] is False
    np.testing.assert_allclose(off["train_aux"], CONST, rtol=1e-6)
    np.testing.assert_array_equal(off["monitored_loss"], off["val_loss"])

    # On: included.
    np.testing.assert_allclose(
        np.asarray(on["monitored_loss"]) - np.asarray(on["val_loss"]),
        CONST,
        rtol=1e-6,
    )

    # A constant shift cannot move an argmin, so best_epoch must agree. Any other
    # result would mean the two series are not the same curve plus a constant.
    assert off["best_epoch"] == on["best_epoch"]


def test_a_weight_dependent_penalty_enters_the_gradient():
    """The test that distinguishes a live hook from a silently-dropped one.

    `add_loss` fails exactly here: accepted, recorded, never differentiated. A ridge
    on tau has a non-zero gradient wherever tau is non-zero, so a live hook must both
    move the weights and shrink the fitted tau.
    """
    plain, model_plain, tau_plain = _fit(aux_loss_fn=None, epochs=25)
    ridged, model_ridged, tau_ridged = _fit(
        aux_loss_fn=_tau_ridge(5e3), aux_in_monitor=True, epochs=25
    )

    # Precondition: without it the comparison below would pass vacuously.
    assert np.mean(tau_plain ** 2) > 1e-8, "unpenalized tau is already ~0"

    # The weights actually moved somewhere else.
    moved = [
        not np.allclose(a.value, b.value, rtol=1e-4, atol=1e-6)
        for a, b in zip(model_plain.trainable_variables, model_ridged.trainable_variables)
    ]
    assert any(moved), "a weight-dependent penalty left every variable unchanged"

    # And it moved them in the direction the penalty asks for.
    assert np.mean(tau_ridged ** 2) < np.mean(tau_plain ** 2)

    # The penalty is non-trivial in size, so the comparison is not about rounding.
    assert ridged["train_aux"][0] > 0.0


def test_aux_loss_fn_reaches_both_the_selection_fit_and_the_refit():
    """AIREstimator must forward the hook to the refit as well as the selection fit.

    Under cate_refit_full the refit model IS the estimator that gets returned, so a
    hook that reached only the selection fit would hand back an unpenalized model
    sitting behind a penalized history -- the same asymmetry `cate_model_args` already
    guards against for normalize_time.
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
        patience=None,
        cate_val_frac=0.25,
        cate_refit_full=True,
        rng_seed=0,
        aux_loss_fn=_constant_penalty,
        aux_in_monitor=True,
    )
    est.fit(X, W, Z, Delta, t_end=5.0, verbose=0)

    assert est.refit_history_ is not None, "the refit must have run"
    for label, history in (("selection", est.history_), ("refit", est.refit_history_)):
        assert history["aux_in_monitor"] is True, label
        np.testing.assert_allclose(
            history["train_aux"], CONST, rtol=1e-6,
            err_msg=f"the {label} fit did not receive the penalty",
        )
