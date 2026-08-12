"""Q-fold cross-fitting orchestrator for nuisance estimation."""

import warnings

import jax.numpy as jnp
import numpy as np
from sklearn.model_selection import KFold

from air_cate._types import NuisanceEstimates, SurvivalData
from air_cate.nuisance.propensity import PropensityMLP
from air_cate.nuisance.weibull_aft import WeibullAFT


def cross_fit_nuisances(
    data: SurvivalData,
    time_grid: jnp.ndarray,
    n_folds: int = 5,
    propensity_config: dict | None = None,
    weibull_config: dict | None = None,
    training_config: dict | None = None,
    rng_seed: int = 42,
    censoring_config: dict | None = None,
    admin_as_censored: bool = False,
    t_end: float | None = None,
) -> NuisanceEstimates:
    """Estimate nuisance functions via Q-fold cross-fitting.

    For each fold q:
      - Train propensity, event survival (per arm), censoring survival (per arm)
        on the complement of fold q.
      - Predict on fold q.

    Args:
        data: SurvivalData with (X, W, Z, Delta).
        time_grid: (M+1,) uniform time grid.
        n_folds: Number of cross-fitting folds.
        propensity_config: kwargs for PropensityMLP constructor.
        weibull_config: kwargs for WeibullAFT constructor.
        training_config: Training kwargs (epochs, batch_size, lr, verbose).
        rng_seed: Random seed for fold splitting.
        censoring_config: kwargs for the two *censoring* WeibullAFT models.
            Defaults to `weibull_config`. Split out because the censoring and
            event processes rarely deserve the same specification -- see
            `WeibullAFT(shared_shape=...)`.
        admin_as_censored: Treat administrative end-of-follow-up (Z at t_end) as
            censored rather than as an observed censoring event when fitting the
            censoring model. Requires `t_end`. Off by default; see below.
        t_end: Maximum follow-up time. Only needed for `admin_as_censored`.

    Why `admin_as_censored` matters. The censoring model is fitted with event
    indicator 1 - Delta, which is 1 both when C < min(D, t_end) -- censoring
    genuinely observed -- and when min(D, C) >= t_end, where Z = t_end exactly
    and all that was learned is that follow-up ended. The observable censoring
    variable is min(C, t_end), so those rows form a point mass at t_end. Two
    consequences, both bad:

      - A Weibull has no atom to offer, so fitting the spike is paid for with a
        distorted S(t) on the interior: the shape is pushed up and late-time
        survival collapses, and 1/S(t) is what enters the AIPCW IPCW term.
      - The AIPCW derivation needs the survival of C itself. At t = t_end the
        IPCW numerator 1{Z > t_end} is 0 for everyone, but the censored
        augmentation term still divides by S*(Z) for every censored row, and Z is
        exactly t_end on the administrative ones. The survival of min(C, t_end)
        is 0 there, so those rows get the largest weight the clip allows, while
        the survival of C -- the quantity the algebra calls for -- is positive.

    Treating Z = t_end as censored for the censoring model targets the law of C
    on all of [0, t_end], which is what both terms require.

    Returns:
        NuisanceEstimates with all cross-fitted predictions in original order.
    """
    prop_cfg = propensity_config or {}
    weib_cfg = weibull_config or {}
    cens_cfg = weib_cfg if censoring_config is None else censoring_config
    if admin_as_censored and t_end is None:
        raise ValueError("admin_as_censored=True requires t_end.")
    # Copy: the setdefault calls below must not mutate the caller's dict, since
    # AIREstimatorCV reuses one config dict across many cross-fitting runs.
    train_cfg = dict(training_config or {})
    train_cfg.setdefault("epochs", 100)
    train_cfg.setdefault("batch_size", 256)
    train_cfg.setdefault("lr", 1e-3)
    train_cfg.setdefault("verbose", 0)

    X = np.asarray(data.X, dtype=np.float32)
    W = np.asarray(data.W, dtype=np.float32)
    Z = np.asarray(data.Z, dtype=np.float32)
    Delta = np.asarray(data.Delta, dtype=np.float32)
    t_grid_np = np.asarray(time_grid, dtype=np.float32)

    # Event indicator for the censoring model. Compare Z against t_end with a
    # tolerance rather than for equality: min(D, C, t_end) returns t_end
    # bit-exactly today, but a t_end that is not representable in float32 would
    # otherwise silently drop the whole correction.
    cens_event_all = 1.0 - Delta
    if admin_as_censored:
        cens_event_all = (cens_event_all * (Z < t_end - 1e-6)).astype(np.float32)

    n = X.shape[0]
    M_plus_1 = len(t_grid_np)

    # Pre-allocate output arrays
    e_hat = np.zeros(n, dtype=np.float32)
    S_event_0_grid = np.zeros((n, M_plus_1), dtype=np.float32)
    S_event_1_grid = np.zeros((n, M_plus_1), dtype=np.float32)
    S_event_0_at_Z = np.zeros(n, dtype=np.float32)
    S_event_1_at_Z = np.zeros(n, dtype=np.float32)
    S_cens_0_grid = np.zeros((n, M_plus_1), dtype=np.float32)
    S_cens_1_grid = np.zeros((n, M_plus_1), dtype=np.float32)
    S_cens_0_at_Z = np.zeros(n, dtype=np.float32)
    S_cens_1_at_Z = np.zeros(n, dtype=np.float32)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=rng_seed)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        W_tr, W_val = W[train_idx], W[val_idx]
        Z_tr = Z[train_idx]
        Z_val = Z[val_idx]
        Delta_tr = Delta[train_idx]
        cens_event_tr = cens_event_all[train_idx]

        # --- Propensity ---
        prop_model = PropensityMLP(**prop_cfg)
        prop_model.fit(X_tr, W_tr, **train_cfg)
        e_hat[val_idx] = prop_model.predict(X_val)

        # --- Per-arm survival models ---
        for w in [0, 1]:
            arm_mask_tr = W_tr == w
            X_arm = X_tr[arm_mask_tr]
            Z_arm = Z_tr[arm_mask_tr]
            Delta_arm = Delta_tr[arm_mask_tr]
            cens_event_arm = cens_event_tr[arm_mask_tr]

            if X_arm.shape[0] < 20:
                warnings.warn(
                    f"Fold {fold_idx}, arm {w}: only {X_arm.shape[0]} subjects "
                    f"(block of n={n} split into {n_folds} folds). "
                    "Nuisance estimates may be unreliable.",
                    stacklevel=2,
                )

            # Event survival: S†_w(t|x), event indicator = Delta
            event_model = WeibullAFT(**weib_cfg)
            event_model.fit(X_arm, Z_arm, Delta_arm, **train_cfg)

            # Censoring survival: S*_w(t|x), event indicator = 1-Delta, with the
            # administrative rows optionally demoted to censored (see docstring).
            cens_model = WeibullAFT(**cens_cfg)
            cens_model.fit(X_arm, Z_arm, cens_event_arm, **train_cfg)

            # Predict on validation fold (all subjects, not just arm w)
            S_ev_grid = np.asarray(event_model.survival(t_grid_np, X_val))
            S_ev_at_Z = np.asarray(event_model.survival(Z_val, X_val))
            S_ce_grid = np.asarray(cens_model.survival(t_grid_np, X_val))
            S_ce_at_Z = np.asarray(cens_model.survival(Z_val, X_val))

            if w == 0:
                S_event_0_grid[val_idx] = S_ev_grid
                S_event_0_at_Z[val_idx] = S_ev_at_Z
                S_cens_0_grid[val_idx] = S_ce_grid
                S_cens_0_at_Z[val_idx] = S_ce_at_Z
            else:
                S_event_1_grid[val_idx] = S_ev_grid
                S_event_1_at_Z[val_idx] = S_ev_at_Z
                S_cens_1_grid[val_idx] = S_ce_grid
                S_cens_1_at_Z[val_idx] = S_ce_at_Z

            # Drop the Keras models as soon as their predictions are stored.
            # A simulation task fits 5 models per fold across many (n, seed)
            # draws in one process; without this the retained graphs push the
            # n=20000 fits past their memory budget.
            del event_model, cens_model
        del prop_model

    return NuisanceEstimates(
        e_hat=jnp.array(e_hat),
        S_event_0_grid=jnp.array(S_event_0_grid),
        S_event_1_grid=jnp.array(S_event_1_grid),
        S_event_0_at_Z=jnp.array(S_event_0_at_Z),
        S_event_1_at_Z=jnp.array(S_event_1_at_Z),
        S_cens_0_grid=jnp.array(S_cens_0_grid),
        S_cens_1_grid=jnp.array(S_cens_1_grid),
        S_cens_0_at_Z=jnp.array(S_cens_0_at_Z),
        S_cens_1_at_Z=jnp.array(S_cens_1_at_Z),
    )
