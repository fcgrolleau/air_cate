"""Tests for empirical R-loss."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from air_cate.r_loss import r_loss


def test_r_loss_nonnegative():
    """R-loss is always non-negative."""
    rng = np.random.default_rng(0)
    n, M = 50, 20
    tau_pred = jnp.array(rng.standard_normal((n, M + 1)), dtype=jnp.float32)
    Y_tilde = jnp.array(rng.standard_normal((n, M + 1)), dtype=jnp.float32)
    S_marg = jnp.array(rng.uniform(0, 1, (n, M + 1)), dtype=jnp.float32)
    W = jnp.array(rng.binomial(1, 0.5, n), dtype=jnp.float32)
    e_clipped = jnp.array(rng.uniform(0.1, 0.9, n), dtype=jnp.float32)

    loss = r_loss(tau_pred, Y_tilde, S_marg, W, e_clipped, delta=0.25)
    assert float(loss) >= 0


def test_r_loss_zero_at_perfect_fit():
    """R-loss is zero when residuals vanish exactly."""
    n, M = 20, 10
    delta = 0.5
    rng = np.random.default_rng(1)

    W = jnp.array(rng.binomial(1, 0.5, n), dtype=jnp.float32)
    e_clipped = jnp.array(rng.uniform(0.2, 0.8, n), dtype=jnp.float32)
    S_marg = jnp.array(rng.uniform(0.3, 1.0, (n, M + 1)), dtype=jnp.float32)
    tau_pred = jnp.array(rng.standard_normal((n, M + 1)), dtype=jnp.float32)

    # Construct Y_tilde such that residual = 0
    Y_tilde = S_marg + (W[:, None] - e_clipped[:, None]) * tau_pred

    loss = r_loss(tau_pred, Y_tilde, S_marg, W, e_clipped, delta)
    np.testing.assert_allclose(float(loss), 0.0, atol=1e-5)


def test_r_loss_gradient_finite():
    """Gradients of R-loss w.r.t. tau_pred are finite."""
    rng = np.random.default_rng(2)
    n, M = 30, 15
    tau_pred = jnp.array(rng.standard_normal((n, M + 1)), dtype=jnp.float32)
    Y_tilde = jnp.array(rng.standard_normal((n, M + 1)), dtype=jnp.float32)
    S_marg = jnp.array(rng.uniform(0, 1, (n, M + 1)), dtype=jnp.float32)
    W = jnp.array(rng.binomial(1, 0.5, n), dtype=jnp.float32)
    e_clipped = jnp.array(rng.uniform(0.1, 0.9, n), dtype=jnp.float32)

    grad_fn = jax.grad(lambda tp: r_loss(tp, Y_tilde, S_marg, W, e_clipped, delta=0.25))
    grads = grad_fn(tau_pred)
    assert jnp.all(jnp.isfinite(grads)), "Gradient has non-finite values"
