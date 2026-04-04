"""Tests for CATEMixture model."""

import jax.numpy as jnp
import numpy as np
import pytest

from air_cate.cate_model.mixture import CATEMixture


def test_tau_at_zero():
    """tau(0|x) = 0 for all x (structural constraint)."""
    model = CATEMixture(K=4, encoder_hidden_sizes=(16,))
    n, p = 50, 5
    X = jnp.ones((n, p), dtype=jnp.float32)
    time_grid = jnp.linspace(0, 5.0, 21)

    tau = model({"X": X, "time_grid": time_grid})
    np.testing.assert_allclose(np.array(tau[:, 0]), 0.0, atol=1e-5,
                                err_msg="tau(0|x) should be 0")


def test_tau_bounded():
    """tau(t|x) in [-1, 1] for all t and x."""
    model = CATEMixture(K=4, encoder_hidden_sizes=(16,))
    rng = np.random.default_rng(0)
    X = jnp.array(rng.standard_normal((100, 5)), dtype=jnp.float32)
    time_grid = jnp.linspace(0, 10.0, 51)

    tau = model({"X": X, "time_grid": time_grid})
    assert jnp.all(tau >= -1.0 - 1e-6), f"tau min = {float(jnp.min(tau))}"
    assert jnp.all(tau <= 1.0 + 1e-6), f"tau max = {float(jnp.max(tau))}"


def test_output_shape():
    """Output shape is (n, T)."""
    model = CATEMixture(K=3, encoder_hidden_sizes=(16,))
    n, p, T = 30, 5, 41
    X = jnp.ones((n, p), dtype=jnp.float32)
    time_grid = jnp.linspace(0, 5.0, T)

    tau = model({"X": X, "time_grid": time_grid})
    assert tau.shape == (n, T)


def test_gating_sums_to_one():
    """Internal gating weights sum to 1."""
    import keras

    model = CATEMixture(K=4, encoder_hidden_sizes=(16,))
    X = jnp.ones((10, 5), dtype=jnp.float32)

    phi = model.encoder(X)
    rho = keras.activations.softmax(model.gate_dense(phi), axis=-1)
    sums = jnp.sum(rho, axis=-1)
    np.testing.assert_allclose(np.array(sums), 1.0, atol=1e-5)
