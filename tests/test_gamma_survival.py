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


# The real time grid used by the CATE stage: t_end = 5, M = 30. Its first point is
# t = 0 exactly, which is the whole difficulty below.
_T_GRID = jnp.linspace(0.0, 5.0, 31)


@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9, 0.99, 1.0])
def test_gradient_finite_at_t_zero_for_alpha_at_or_below_one(alpha):
    """t = 0 must not poison the gradient when alpha <= 1.

    d/dx gammaincc(a, x) = -x^(a-1) e^(-x) / Gamma(a) diverges at x = 0 for a <= 1,
    and the chain rule to lambda multiplies that infinity by dx/dlam = -t/lam^2 = 0.
    Plain autodiff therefore returns 0 * inf = NaN, which AdamW writes straight into
    every lambda.

    This is the real operating regime, not a corner case: t = 0 is always the first
    grid point, and alpha_bar starts at 1.313 and drifts *down* through 1 during
    training, so on IST-3 every fit crossed this boundary. The older gradient test
    above uses t = 2, alpha = 2 -- the one corner where the bug is invisible.
    """
    def total(a, lam):
        return jnp.sum(gamma_survival(_T_GRID, a, lam))

    g_alpha, g_lam = jax.grad(total, argnums=(0, 1))(
        jnp.array(alpha), jnp.array(1.3)
    )
    assert jnp.isfinite(g_lam), f"grad wrt lambda is {g_lam} at alpha={alpha}"
    assert jnp.isfinite(g_alpha), f"grad wrt alpha is {g_alpha} at alpha={alpha}"


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_t_zero_contributes_no_gradient(alpha):
    """G(0) = 1 identically, so t = 0 must contribute exactly nothing.

    Pins the fix to the *correct* value rather than merely a finite one: dropping the
    t = 0 point must not change the gradient. This is what makes the fix safe for
    alpha > 1, where autodiff was already returning the right answer.
    """
    def total(t):
        return lambda a, lam: jnp.sum(gamma_survival(t, a, lam))

    args = (jnp.array(alpha), jnp.array(1.3))
    with_zero = jax.grad(total(_T_GRID), argnums=(0, 1))(*args)
    without_zero = jax.grad(total(_T_GRID[1:]), argnums=(0, 1))(*args)

    np.testing.assert_allclose(
        np.asarray(with_zero), np.asarray(without_zero), rtol=1e-6,
        err_msg=f"t=0 changed the gradient at alpha={alpha}",
    )


def test_survival_at_zero_is_exactly_one_below_alpha_one():
    """The tau(0|x) = 0 structural constraint must survive the t = 0 special-casing.

    The gradient fix branches on t == 0, so the forward value there is now a
    substituted constant. It has to be exactly 1, not approximately: tau(0|x) = 0
    holds by construction only because G(0) = 1 in both arms and they cancel.
    """
    for alpha in (0.1, 0.5, 0.9, 1.0, 2.0):
        val = float(gamma_survival(jnp.array(0.0), jnp.array(alpha), jnp.array(1.3)))
        assert val == 1.0, f"G(0; {alpha}, 1.3) = {val!r}, expected exactly 1.0"
