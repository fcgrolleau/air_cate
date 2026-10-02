"""Custom JAX training loop for CATE models, with optional validation tracking."""

import jax
import jax.numpy as jnp
import keras
import numpy as np

from air_cate.r_loss import r_loss


def _extract_vars(keras_vars):
    """Extract JAX arrays from Keras variables (JIT boundary crossing)."""
    return [jnp.array(v.value) for v in keras_vars]


def _assign_vars(keras_vars, values):
    """Assign JAX arrays back to Keras variables."""
    for var, val in zip(keras_vars, values):
        var.assign(val)


def train_cate(
    model,
    X: np.ndarray,
    Y_tilde: np.ndarray,
    S_marg: np.ndarray,
    W: np.ndarray,
    e_clipped: np.ndarray,
    time_grid: jnp.ndarray,
    delta: float,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    validation_data: dict | None = None,
    monitor: str = "auto",
    patience: int | None = 10,
    restore_best: bool = True,
    rng_seed: int | None = None,
    verbose: int = 0,
    aux_loss_fn=None,
    aux_in_monitor: bool = False,
) -> dict:
    """Minimize the empirical R-loss over a CATE model via minibatch AdamW.

    Args:
        model: CATEMixture or CATETensorProduct instance (built or unbuilt).
        X: Covariates (n, p).
        Y_tilde: Pseudo-outcomes (n, M+1).
        S_marg: Marginal survival (n, M+1).
        W: Treatment indicators (n,).
        e_clipped: Clipped propensity scores (n,).
        time_grid: (M+1,) time grid.
        delta: Grid step size (scalar).
        epochs: Maximum number of epochs.
        lr: Learning rate.
        batch_size: Minibatch size.
        weight_decay: AdamW weight decay.
        validation_data: Optional dict with keys X, Y_tilde, S_marg, W, e_clipped
            for a held-out block. Evaluated once per epoch as a single forward pass.
        monitor: Which loss drives early stopping and best-epoch selection.
            "train", "val", or "auto" (= "val" if validation_data else "train").
        patience: Epochs without improvement before stopping. None disables
            early stopping entirely (train the full `epochs`); best-epoch
            selection and weight restoration still apply.
        restore_best: Restore the weights from the best epoch when done.
        rng_seed: Seed for minibatch shuffling. None uses fresh entropy.
        verbose: Verbosity level (0=silent).
        aux_loss_fn: Optional auxiliary penalty added to the optimized objective,
            called as

                aux_loss_fn(model, trainable_vars, non_trainable_vars,
                            X_batch, time_grid, tau_pred) -> scalar

            and expected to return its terms already multiplied by their own
            coefficients. `model` is passed in rather than captured so that one hook
            object stays valid across restarts and across the full-data refit, each
            of which builds a fresh model; inside the jitted step the model's
            `v.value` is stale, so use it only for structure and take every numeric
            input from `trainable_vars`. None (the default) leaves the objective, the
            returned history and the emitted HLO identical to the unhooked loop.
        aux_in_monitor: Add the penalty to the monitored series that drives early
            stopping and best-epoch selection. Leave False to monitor the pure
            R-loss; set True whenever a penalty is actually in use. This is not a
            symmetry question: a penalty deliberately moves the fit off the R-loss
            optimum, so a pure-R monitor *rises* as the penalty bites and
            `restore_best` then hands back the least-penalized epoch -- silently
            undoing the penalty and making every penalised run look inert. The
            monitored values are not comparable across penalty strengths, which is
            why `train_loss` and `val_loss` below stay pure R-loss.

    Returns:
        history: dict with keys
            train_loss: list of per-epoch training R-loss, averaged over
                minibatches weighted by batch size (so each subject counts once
                per epoch). Still a running average over the epoch's parameter
                trajectory rather than the loss at a single parameter value.
            val_loss: list of per-epoch validation R-loss, or None.
            train_aux: list of per-epoch auxiliary penalty, size-weighted exactly
                like train_loss. All zeros when aux_loss_fn is None.
            val_aux: list of per-epoch validation penalty, or None.
            monitored_loss: the series actually minimized over to choose
                best_epoch. Element-wise equal to train_loss or val_loss when
                aux_in_monitor is False.
            aux_in_monitor: the resolved flag, so an archived history records which
                series best_epoch came from.
            best_epoch: 0-indexed epoch minimizing the monitored loss.
            best_train_loss: training loss at best_epoch (pure R-loss).
            best_val_loss: validation loss at best_epoch (pure R-loss), or None.
            stopped_epoch: 0-indexed final epoch actually run.
            monitor: the resolved monitor string.
    """
    if monitor == "auto":
        monitor = "val" if validation_data is not None else "train"
    if monitor not in ("train", "val"):
        raise ValueError(f"monitor must be 'train', 'val', or 'auto', got {monitor!r}")
    if monitor == "val" and validation_data is None:
        raise ValueError("monitor='val' requires validation_data")

    n = X.shape[0]
    X_jax = jnp.asarray(X, dtype=jnp.float32)
    Y_tilde_jax = jnp.asarray(Y_tilde, dtype=jnp.float32)
    S_marg_jax = jnp.asarray(S_marg, dtype=jnp.float32)
    W_jax = jnp.asarray(W, dtype=jnp.float32)
    e_clipped_jax = jnp.asarray(e_clipped, dtype=jnp.float32)
    time_grid_jax = jnp.asarray(time_grid, dtype=jnp.float32)
    delta = float(delta)

    optimizer = keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    # Build the model by calling it once, then build the optimizer slots
    model({"X": X_jax[:1], "time_grid": time_grid_jax})
    optimizer.build(model.trainable_variables)

    # Two separate jitted steps rather than one with a branch inside the traced
    # function: the aux_loss_fn=None branch then keeps the exact closure and body it
    # had before this hook existed, so the unhooked objective and the emitted HLO are
    # unchanged. `has_aux=True` alters the traced output structure, which is why the
    # None path does not simply reuse the penalised body with a zero penalty.
    if aux_loss_fn is None:

        @jax.jit
        def train_step(
            trainable_vars, non_trainable_vars, opt_vars,
            X_batch, Y_tilde_batch, S_marg_batch, W_batch, e_batch,
        ):
            def loss_fn(trainable_vars):
                tau_pred, _ = model.stateless_call(
                    trainable_vars, non_trainable_vars,
                    {"X": X_batch, "time_grid": time_grid_jax},
                )
                return r_loss(tau_pred, Y_tilde_batch, S_marg_batch, W_batch, e_batch, delta)

            loss, grads = jax.value_and_grad(loss_fn)(trainable_vars)
            trainable_vars, opt_vars = optimizer.stateless_apply(opt_vars, grads, trainable_vars)
            # A constant second output keeps the caller's unpacking uniform without
            # entering the traced gradient computation.
            return loss, jnp.zeros((), dtype=loss.dtype), trainable_vars, opt_vars

    else:

        @jax.jit
        def train_step(
            trainable_vars, non_trainable_vars, opt_vars,
            X_batch, Y_tilde_batch, S_marg_batch, W_batch, e_batch,
        ):
            def loss_fn(trainable_vars):
                tau_pred, _ = model.stateless_call(
                    trainable_vars, non_trainable_vars,
                    {"X": X_batch, "time_grid": time_grid_jax},
                )
                r = r_loss(tau_pred, Y_tilde_batch, S_marg_batch, W_batch, e_batch, delta)
                aux = aux_loss_fn(
                    model, trainable_vars, non_trainable_vars,
                    X_batch, time_grid_jax, tau_pred,
                )
                # Gradient descends r + aux; the two are reported separately so the
                # history can keep train_loss pure.
                return r + aux, (r, aux)

            (_, (r, aux)), grads = jax.value_and_grad(loss_fn, has_aux=True)(trainable_vars)
            trainable_vars, opt_vars = optimizer.stateless_apply(opt_vars, grads, trainable_vars)
            return r, aux, trainable_vars, opt_vars

    @jax.jit
    def eval_step(
        trainable_vars, non_trainable_vars,
        X_eval, Y_tilde_eval, S_marg_eval, W_eval, e_eval,
    ):
        tau_pred, _ = model.stateless_call(
            trainable_vars, non_trainable_vars,
            {"X": X_eval, "time_grid": time_grid_jax},
        )
        r = r_loss(tau_pred, Y_tilde_eval, S_marg_eval, W_eval, e_eval, delta)
        # `aux_loss_fn` is a closure constant, so this branch resolves at trace time
        # and emits nothing. No gradient is taken here, so unlike train_step the two
        # cases can share one body.
        if aux_loss_fn is None:
            return r, jnp.zeros((), dtype=r.dtype)
        aux = aux_loss_fn(
            model, trainable_vars, non_trainable_vars,
            X_eval, time_grid_jax, tau_pred,
        )
        return r, aux

    if validation_data is not None:
        val_args = (
            jnp.asarray(validation_data["X"], dtype=jnp.float32),
            jnp.asarray(validation_data["Y_tilde"], dtype=jnp.float32),
            jnp.asarray(validation_data["S_marg"], dtype=jnp.float32),
            jnp.asarray(validation_data["W"], dtype=jnp.float32),
            jnp.asarray(validation_data["e_clipped"], dtype=jnp.float32),
        )

    rng = np.random.default_rng(rng_seed)

    train_history: list[float] = []
    val_history: list[float] = [] if validation_data is not None else None
    train_aux_history: list[float] = []
    val_aux_history: list[float] = [] if validation_data is not None else None
    monitored_history: list[float] = []

    best_loss = float("inf")
    best_epoch = 0
    best_weights = None
    patience_counter = 0
    stopped_epoch = 0

    for epoch in range(epochs):
        stopped_epoch = epoch
        perm = rng.permutation(n)
        epoch_loss_sum = 0.0
        epoch_aux_sum = 0.0

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]

            # Extract pure JAX arrays from Keras variables
            tv = _extract_vars(model.trainable_variables)
            ntv = _extract_vars(model.non_trainable_variables)
            ov = _extract_vars(optimizer.variables)

            loss, aux, new_tv, new_ov = train_step(
                tv, ntv, ov,
                X_jax[idx],
                Y_tilde_jax[idx],
                S_marg_jax[idx],
                W_jax[idx],
                e_clipped_jax[idx],
            )

            _assign_vars(model.trainable_variables, new_tv)
            _assign_vars(optimizer.variables, new_ov)

            # Weight each batch by its size rather than equally. The R-loss is a
            # mean over subjects, so an equal-weight average over unequal batches
            # over-weights the final short batch of n % batch_size rows -- 32 rows
            # carrying 1/4 of the weight at n=800, batch_size=256. The per-subject
            # loss is heavy-tailed (top contributions ~5x the mean), so reshuffling
            # alone then moves the epoch number by sd ~0.39 against a real
            # epoch-to-epoch movement of ~0.05: the training curve becomes almost
            # pure shuffle noise, and early stopping on it fires at random.
            # Size-weighting makes each subject count once per epoch, so the epoch
            # loss is shuffle-invariant.
            epoch_loss_sum += float(loss) * len(idx)
            epoch_aux_sum += float(aux) * len(idx)

        mean_loss = epoch_loss_sum / n
        mean_aux = epoch_aux_sum / n
        train_history.append(mean_loss)
        train_aux_history.append(mean_aux)

        if validation_data is not None:
            val_r, val_aux = eval_step(
                _extract_vars(model.trainable_variables),
                _extract_vars(model.non_trainable_variables),
                *val_args,
            )
            val_history.append(float(val_r))
            val_aux_history.append(float(val_aux))

        # The monitored series is what best_epoch and `patience` read. It picks up the
        # penalty only under aux_in_monitor, so an unhooked run is bit-identical.
        if monitor == "val":
            monitored = val_history[-1] + (val_aux_history[-1] if aux_in_monitor else 0.0)
        else:
            monitored = mean_loss + (mean_aux if aux_in_monitor else 0.0)
        monitored_history.append(monitored)

        if verbose > 0 and (epoch + 1) % 10 == 0:
            msg = f"Epoch {epoch + 1}/{epochs}, loss: {mean_loss:.6f}"
            if validation_data is not None:
                msg += f", val_loss: {val_history[-1]:.6f}"
            if aux_loss_fn is not None:
                msg += f", aux: {mean_aux:.6f}"
                if validation_data is not None:
                    msg += f", val_aux: {val_aux_history[-1]:.6f}"
            print(msg)

        # Snapshot non-trainable variables too: harmless today (no BN in the
        # encoder) but correct if stateful layers are ever added.
        candidate = None
        improved = bool(np.isfinite(monitored)) and monitored < best_loss - 1e-6
        if improved:
            candidate = (
                _extract_vars(model.trainable_variables),
                _extract_vars(model.non_trainable_variables),
            )
            # A finite `monitored` is not enough to trust these weights. The epoch
            # loss averages the *pre-update* loss of each minibatch, so a NaN
            # created by the epoch's final batch does not appear in `monitored` --
            # it first shows up an epoch later. Snapshotting on the loss alone
            # therefore lets a NaN model be recorded as "best" under a perfectly
            # finite loss and restored below, yielding NaN predictions with no
            # warning of any kind. Checking the weights themselves is what closes
            # that window.
            if not all(
                bool(jnp.all(jnp.isfinite(v))) for v in candidate[0] + candidate[1]
            ):
                improved = False

        if improved:
            best_loss = monitored
            best_epoch = epoch
            patience_counter = 0
            best_weights = candidate
        else:
            patience_counter += 1
            if patience is not None and patience_counter >= patience:
                if verbose > 0:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

    # No finite snapshot at all means every epoch was either non-improving from the
    # start or numerically broken, and the model still holds whatever training left
    # in it -- possibly NaN. Returning it would score as iMSE = nan and, in a large
    # simulation, quietly contaminate the averages instead of failing. Fail loudly.
    if best_weights is None:
        first_loss = train_history[0] if train_history else float("nan")
        raise RuntimeError(
            f"CATE training produced no usable weights in {stopped_epoch + 1} epoch(s): "
            f"every epoch was non-improving or non-finite (first epoch loss "
            f"{first_loss:.6g}). Refusing to return a model that may hold NaN."
        )

    if restore_best:
        _assign_vars(model.trainable_variables, best_weights[0])
        _assign_vars(model.non_trainable_variables, best_weights[1])

    return {
        "train_loss": train_history,
        "val_loss": val_history,
        "train_aux": train_aux_history,
        "val_aux": val_aux_history,
        "monitored_loss": monitored_history,
        "aux_in_monitor": bool(aux_in_monitor),
        "best_epoch": best_epoch,
        "best_train_loss": train_history[best_epoch],
        "best_val_loss": val_history[best_epoch] if val_history is not None else None,
        "stopped_epoch": stopped_epoch,
        "monitor": monitor,
    }
