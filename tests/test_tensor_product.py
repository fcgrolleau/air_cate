"""Tests for CATETensorProduct model."""

import jax.numpy as jnp
import numpy as np
import pytest


def test_tensor_product_output_shape():
    """Forward pass returns correct shape."""
    from air_cate.cate_model.tensor_product import CATETensorProduct

    model = CATETensorProduct(n_time_basis=6, poly_degree=2)
    X = jnp.ones((8, 5), dtype=jnp.float32)
    time_grid = jnp.linspace(0.0, 5.0, 31)

    tau = model({"X": X, "time_grid": time_grid})
    assert tau.shape == (8, 31)


def test_tensor_product_poly_features_count():
    """Polynomial features have expected dimension for p=5, degree=2."""
    from air_cate.cate_model.tensor_product import CATETensorProduct

    model = CATETensorProduct(n_time_basis=4, poly_degree=2)
    X = jnp.ones((3, 5))
    phi = model._poly_features(X)
    # C(5+2, 2) = 21: 1 intercept + 5 linear + 15 quadratic
    assert phi.shape == (3, 21)


def test_tensor_product_time_basis_shape():
    """Time basis has correct shape and starts with ones row."""
    from air_cate.cate_model.tensor_product import CATETensorProduct

    model = CATETensorProduct(n_time_basis=5, poly_degree=1)
    t = jnp.linspace(0.0, 5.0, 11)
    B = model._time_basis(t)
    assert B.shape == (5, 11)
    # First row should be all ones (t^0)
    np.testing.assert_allclose(B[0], 1.0, atol=1e-6)


def test_tensor_product_no_structural_constraints():
    """Tensor-product model can violate tau(0|x)=0 (unlike CATEMixture)."""
    from air_cate.cate_model.tensor_product import CATETensorProduct

    model = CATETensorProduct(n_time_basis=4, poly_degree=1)
    rng = np.random.default_rng(42)
    X = jnp.array(rng.normal(0, 1, (10, 3)), dtype=jnp.float32)
    time_grid = jnp.linspace(0.0, 5.0, 21)

    # Build model
    _ = model({"X": X, "time_grid": time_grid})

    # Assign random weights to theta so tau(0|x) != 0 in general
    model._theta.assign(jnp.array(rng.normal(0, 1, model._theta.shape), dtype=jnp.float32))
    tau = model({"X": X, "time_grid": time_grid})

    # tau at t=0 should generally NOT be zero (unconstrained)
    assert not np.allclose(np.asarray(tau[:, 0]), 0.0, atol=1e-4)


def test_tensor_product_in_estimator():
    """AIREstimator works with CATETensorProduct via cate_model_class."""
    from air_cate.estimator import AIREstimator
    from air_cate.cate_model.tensor_product import CATETensorProduct

    rng = np.random.default_rng(0)
    n, p = 100, 5
    X = rng.uniform(-1, 1, (n, p)).astype(np.float32)
    W = rng.binomial(1, 0.5, n).astype(np.float32)
    Z = rng.exponential(2.0, n).astype(np.float32)
    Delta = rng.binomial(1, 0.7, n).astype(np.float32)

    est = AIREstimator(
        M=10, n_folds=2, eps=0.01,
        nuisance_training_config={"epochs": 5, "verbose": 0},
        cate_epochs=10,
        cate_model_class=CATETensorProduct,
        cate_model_kwargs=dict(n_time_basis=4, poly_degree=2),
    )
    est.fit(X, W, Z, Delta, t_end=5.0, verbose=0)

    tau = est.predict_tau(X[:10])
    assert tau.shape == (10, 11)
