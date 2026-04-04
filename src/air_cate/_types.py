"""Data containers for the AIR estimator pipeline."""

from typing import NamedTuple

import jax.numpy as jnp


class SurvivalData(NamedTuple):
    """Observed right-censored survival data."""

    X: jnp.ndarray  # (n, p) covariates
    W: jnp.ndarray  # (n,) binary treatment
    Z: jnp.ndarray  # (n,) observed time = min(D, C)
    Delta: jnp.ndarray  # (n,) event indicator = 1{D <= C}


class NuisanceEstimates(NamedTuple):
    """Cross-fitted nuisance estimates on time grid and at observed times."""

    e_hat: jnp.ndarray  # (n,) propensity score
    S_event_0_grid: jnp.ndarray  # (n, M+1) event survival, arm 0, on grid
    S_event_1_grid: jnp.ndarray  # (n, M+1) event survival, arm 1, on grid
    S_event_0_at_Z: jnp.ndarray  # (n,) event survival arm 0 at Z_i
    S_event_1_at_Z: jnp.ndarray  # (n,) event survival arm 1 at Z_i
    S_cens_0_grid: jnp.ndarray  # (n, M+1) censoring survival, arm 0, on grid
    S_cens_1_grid: jnp.ndarray  # (n, M+1) censoring survival, arm 1, on grid
    S_cens_0_at_Z: jnp.ndarray  # (n,) censoring survival arm 0 at Z_i
    S_cens_1_at_Z: jnp.ndarray  # (n,) censoring survival arm 1 at Z_i
