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

    Returns:
        NuisanceEstimates with all cross-fitted predictions in original order.
    """
    prop_cfg = propensity_config or {}
    weib_cfg = weibull_config or {}
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

            # Censoring survival: S*_w(t|x), event indicator = 1-Delta
            cens_model = WeibullAFT(**weib_cfg)
            cens_model.fit(X_arm, Z_arm, 1.0 - Delta_arm, **train_cfg)

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
