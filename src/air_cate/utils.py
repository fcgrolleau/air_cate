"""Utility functions for grid construction and numerical stability."""

import jax.numpy as jnp


def build_time_grid(t_end: float, M: int) -> tuple[jnp.ndarray, float]:
    """Build a uniform time grid on [0, t_end] with M+1 points.

    Returns (grid, delta) where grid has shape (M+1,) and delta = t_end / M.
    """
    delta = t_end / M
    grid = jnp.linspace(0.0, t_end, M + 1)
    return grid, delta


def clip_propensity(e: jnp.ndarray, eps: float) -> jnp.ndarray:
    """Clip propensity scores to [eps, 1-eps]."""
    return jnp.clip(e, eps, 1.0 - eps)


def clip_survival(S: jnp.ndarray, eps: float) -> jnp.ndarray:
    """Clip survival function estimates to [eps, inf)."""
    return jnp.maximum(S, eps)
