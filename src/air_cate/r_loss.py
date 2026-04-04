"""Empirical R-loss for the time-varying CATE."""

import jax.numpy as jnp


def r_loss(
    tau_pred: jnp.ndarray,
    Y_tilde: jnp.ndarray,
    S_marg: jnp.ndarray,
    W: jnp.ndarray,
    e_clipped: jnp.ndarray,
    delta: float,
) -> jnp.ndarray:
    """Compute the discretized empirical R-loss.

    L = (1/n) * sum_i sum_m delta * (Y_tilde_i(t_m) - S_marg(t_m|X_i) - (W_i - e(X_i)) * tau(t_m|X_i))^2

    Args:
        tau_pred: (n, M+1) predicted CATE values on time grid.
        Y_tilde: (n, M+1) pseudo-outcomes.
        S_marg: (n, M+1) marginal survival estimates.
        W: (n,) treatment indicators.
        e_clipped: (n,) clipped propensity scores.
        delta: Grid step size (scalar).

    Returns:
        Scalar loss value.
    """
    residual = Y_tilde - S_marg - (W[:, None] - e_clipped[:, None]) * tau_pred
    return jnp.mean(jnp.sum(delta * residual**2, axis=1))
