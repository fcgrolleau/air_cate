"""Gamma survival function for treatment effect profiles."""

import jax
import jax.numpy as jnp


def gamma_survival(t: jnp.ndarray, alpha: jnp.ndarray, lam: jnp.ndarray) -> jnp.ndarray:
    """Compute Gamma survival function G(t; alpha, lambda) = P(T > t).

    G(t; alpha, lambda) = gammaincc(alpha, t / lambda)

    where gammaincc is the regularized upper incomplete gamma function,
    and T ~ Gamma(shape=alpha, scale=lambda).

    Args:
        t: Time points. Shape is flexible for broadcasting.
        alpha: Shape parameter (must be > 0).
        lam: Scale parameter (must be > 0).

    Returns:
        Survival probabilities with shape determined by broadcasting.
    """
    return jax.scipy.special.gammaincc(alpha, t / lam)
