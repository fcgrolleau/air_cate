"""Build NuisanceEstimates from true DGP functions (oracle nuisance)."""

import jax.numpy as jnp
import numpy as np

from air_cate._types import NuisanceEstimates


def _weibull_survival_grid(t_grid, X, w, shape, scale_fn):
    """S_w(t|x) = exp(-(t/scale(X,w))^shape) on a time grid."""
    scales = scale_fn(X, w)  # (n,)
    return np.exp(-((np.asarray(t_grid)[None, :] / scales[:, None]) ** shape))


def _weibull_survival_at_z(Z, X, w, shape, scale_fn):
    """S_w(Z_i|X_i) = exp(-(Z_i/scale(X_i,w))^shape) per subject."""
    scales = scale_fn(X, w)  # (n,)
    return np.exp(-((Z / scales) ** shape))


def _mixture_survival_at_z(Z, X, w, dgp):
    """Per-subject mixture survival S_w(Z_i|X_i) for WeibullMixtureDGP."""
    scales = dgp._scales(X, w)    # (n, 3)
    weights = dgp._weights(X, w)  # (n, 3)
    S = np.zeros(X.shape[0])
    for k in range(3):
        S += weights[:, k] * np.exp(-((Z / scales[:, k]) ** dgp.shapes[k]))
    return S


def build_oracle_nuisances(dgp_class, X, Z, time_grid):
    """Construct NuisanceEstimates from true DGP functions.

    Works for WeibullDGP and WeibullMixtureDGP from tests/conftest.py, which
    share censoring ~ Weibull(shape=1, scale=5) = Exp(rate=1/5), and for
    IST3TextDGP, which exposes its own survival methods and per-subject
    covariate-dependent censoring.

    Args:
        dgp_class: DGP class (WeibullDGP, WeibullMixtureDGP, or IST3TextDGP).
        X: (n, p) covariates (numpy). For IST3TextDGP this must be the raw
            X_full returned by `generate()` -- the survival methods key off the
            pool indices that call stored, so a normalized copy will not do.
        Z: (n,) observed times (numpy).
        time_grid: (M+1,) time grid (jax or numpy array).

    Returns:
        NuisanceEstimates with all fields populated from true functions.
    """
    t_grid_np = np.asarray(time_grid)
    Z_np = np.asarray(Z)
    n = X.shape[0]

    # --- Propensity ---
    e_hat = dgp_class.propensity(X)  # (n,)

    # --- Event and censoring survival ---
    if hasattr(dgp_class, "_control_survival_grid"):
        # IST3TextDGP: pre-computed per-subject survival methods
        S_ev_0_grid = dgp_class._control_survival_grid(t_grid_np, X)
        S_ev_1_grid = dgp_class._treated_survival_grid(t_grid_np, X)
        S_ev_0_at_Z = dgp_class._control_survival_pointwise(Z_np, X)
        S_ev_1_at_Z = dgp_class._treated_survival_pointwise(Z_np, X)

        if hasattr(dgp_class, "_censoring_survival_grid"):
            # Covariate-dependent Weibull censoring
            S_cens_grid_2d = dgp_class._censoring_survival_grid(t_grid_np, X)
            S_cens_at_Z = dgp_class._censoring_survival_pointwise(Z_np, X)
        else:
            # Independent Exp censoring
            cens_scale = dgp_class.cens_scale
            S_cens_grid = np.exp(-t_grid_np / cens_scale)  # (M+1,)
            S_cens_grid_2d = np.broadcast_to(
                S_cens_grid[None, :], (n, len(t_grid_np))
            ).copy()
            S_cens_at_Z = np.exp(-Z_np / cens_scale)  # (n,)

        return NuisanceEstimates(
            e_hat=jnp.array(e_hat, dtype=jnp.float32),
            S_event_0_grid=jnp.array(S_ev_0_grid, dtype=jnp.float32),
            S_event_1_grid=jnp.array(S_ev_1_grid, dtype=jnp.float32),
            S_event_0_at_Z=jnp.array(S_ev_0_at_Z, dtype=jnp.float32),
            S_event_1_at_Z=jnp.array(S_ev_1_at_Z, dtype=jnp.float32),
            S_cens_0_grid=jnp.array(S_cens_grid_2d, dtype=jnp.float32),
            S_cens_1_grid=jnp.array(S_cens_grid_2d, dtype=jnp.float32),
            S_cens_0_at_Z=jnp.array(S_cens_at_Z, dtype=jnp.float32),
            S_cens_1_at_Z=jnp.array(S_cens_at_Z, dtype=jnp.float32),
        )

    if hasattr(dgp_class, "scale"):
        # WeibullDGP: single Weibull per arm
        S_ev_0_grid = _weibull_survival_grid(
            t_grid_np, X, 0, dgp_class.shape, dgp_class.scale
        )
        S_ev_1_grid = _weibull_survival_grid(
            t_grid_np, X, 1, dgp_class.shape, dgp_class.scale
        )
        S_ev_0_at_Z = _weibull_survival_at_z(
            Z_np, X, 0, dgp_class.shape, dgp_class.scale
        )
        S_ev_1_at_Z = _weibull_survival_at_z(
            Z_np, X, 1, dgp_class.shape, dgp_class.scale
        )
    else:
        # WeibullMixtureDGP: 3-component mixture
        S_ev_0_grid = dgp_class._survival(t_grid_np, X, 0)  # (n, M+1)
        S_ev_1_grid = dgp_class._survival(t_grid_np, X, 1)
        S_ev_0_at_Z = _mixture_survival_at_z(Z_np, X, 0, dgp_class)
        S_ev_1_at_Z = _mixture_survival_at_z(Z_np, X, 1, dgp_class)

    # --- Censoring survival (same for both conftest DGPs: Weibull(1, 5) = Exp(1/5)) ---
    cens_scale = dgp_class.cens_scale
    S_cens_grid = np.exp(-t_grid_np / cens_scale)  # (M+1,)
    S_cens_grid_2d = np.broadcast_to(S_cens_grid[None, :], (n, len(t_grid_np))).copy()
    S_cens_at_Z = np.exp(-Z_np / cens_scale)  # (n,)

    return NuisanceEstimates(
        e_hat=jnp.array(e_hat, dtype=jnp.float32),
        S_event_0_grid=jnp.array(S_ev_0_grid, dtype=jnp.float32),
        S_event_1_grid=jnp.array(S_ev_1_grid, dtype=jnp.float32),
        S_event_0_at_Z=jnp.array(S_ev_0_at_Z, dtype=jnp.float32),
        S_event_1_at_Z=jnp.array(S_ev_1_at_Z, dtype=jnp.float32),
        S_cens_0_grid=jnp.array(S_cens_grid_2d, dtype=jnp.float32),
        S_cens_1_grid=jnp.array(S_cens_grid_2d, dtype=jnp.float32),
        S_cens_0_at_Z=jnp.array(S_cens_at_Z, dtype=jnp.float32),
        S_cens_1_at_Z=jnp.array(S_cens_at_Z, dtype=jnp.float32),
    )
