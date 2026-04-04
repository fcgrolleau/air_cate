"""Tests for gamma survival function."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from air_cate.cate_model.gamma_survival import gamma_survival


def test_gamma_survival_at_zero():
    """G(0; alpha, lambda) = 1 for all positive parameters."""
    for alpha in [0.5, 1.0, 2.0, 5.0]:
        for lam in [0.5, 1.0, 3.0]:
            val = float(gamma_survival(jnp.array(0.0), jnp.array(alpha), jnp.array(lam)))
            np.testing.assert_allclose(val, 1.0, atol=1e-6,
                                        err_msg=f"G(0; {alpha}, {lam}) = {val}, expected 1.0")


def test_gamma_survival_large_t():
    """G(t; alpha, lambda) -> 0 as t -> inf."""
    alpha, lam = 2.0, 1.0
    val = float(gamma_survival(jnp.array(100.0), jnp.array(alpha), jnp.array(lam)))
    assert val < 1e-10, f"G(100; {alpha}, {lam}) = {val}, should be near 0"


def test_gamma_survival_monotone():
    """G is monotonically decreasing in t."""
    alpha, lam = 2.0, 1.5
    t = jnp.linspace(0.01, 20.0, 200)
    S = gamma_survival(t, jnp.array(alpha), jnp.array(lam))
    diffs = jnp.diff(S)
    assert jnp.all(diffs <= 1e-6), "Gamma survival should be non-increasing"


def test_gamma_survival_matches_scipy():
    """Values match scipy.stats.gamma.sf."""
    alphas = [0.5, 1.0, 2.0, 5.0]
    lams = [0.5, 1.0, 3.0]
    t_vals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    for alpha in alphas:
        for lam in lams:
            for t in t_vals:
                jax_val = float(gamma_survival(jnp.array(t), jnp.array(alpha), jnp.array(lam)))
                scipy_val = float(stats.gamma.sf(t, a=alpha, scale=lam))
                np.testing.assert_allclose(
                    jax_val, scipy_val, atol=1e-5, rtol=1e-5,
                    err_msg=f"Mismatch at alpha={alpha}, lam={lam}, t={t}"
                )


def test_gamma_survival_gradients_finite():
    """Gradients w.r.t. alpha and lambda are finite and non-zero."""
    t = jnp.array(2.0)
    alpha = jnp.array(2.0)
    lam = jnp.array(1.5)

    grad_fn = jax.grad(lambda a, l: gamma_survival(t, a, l), argnums=(0, 1))
    g_alpha, g_lam = grad_fn(alpha, lam)

    assert jnp.isfinite(g_alpha), f"Gradient w.r.t. alpha is not finite: {g_alpha}"
    assert jnp.isfinite(g_lam), f"Gradient w.r.t. lambda is not finite: {g_lam}"
    assert g_lam != 0.0, "Gradient w.r.t. lambda should be non-zero"
