"""Unconstrained tensor-product basis model for time-varying CATE."""

from itertools import combinations_with_replacement
from math import comb

import jax.numpy as jnp
import keras


class CATETensorProduct(keras.Model):
    """Parametrize tau(t|x) via a tensor-product of polynomial bases.

    tau(t|x) = psi_X(x)^T  Theta  psi_T(t)

    where psi_T is a polynomial basis in (normalized) time and psi_X collects
    all covariate monomials up to a given degree.

    This model is *unconstrained*: it does NOT enforce tau(0|x)=0,
    |tau|<=1, or tau->0 as t->inf.
    """

    def __init__(
        self,
        n_time_basis: int = 8,
        poly_degree: int = 2,
        n_covariates: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_time_basis = n_time_basis
        self.poly_degree = poly_degree
        self._n_covariates = n_covariates
        self._theta = None

    @staticmethod
    def _n_poly_features(p, degree):
        """Number of monomials up to given degree: C(p + degree, degree)."""
        return comb(p + degree, degree)

    def build(self, input_shape):
        p = input_shape["X"][-1] if self._n_covariates is None else self._n_covariates
        d_X = self._n_poly_features(p, self.poly_degree)
        self._theta = self.add_weight(
            name="theta",
            shape=(d_X, self.n_time_basis),
            initializer="glorot_uniform",
        )
        super().build(input_shape)

    def _poly_features(self, X):
        """All monomials of X up to self.poly_degree.

        For p covariates and degree d, produces C(p+d, d) features
        (including the intercept = 1).

        Returns:
            (n, d_X) feature matrix.
        """
        n, p = X.shape
        cols = [jnp.ones((n,))]  # intercept
        for deg in range(1, self.poly_degree + 1):
            for combo in combinations_with_replacement(range(p), deg):
                term = jnp.ones((n,))
                for idx in combo:
                    term = term * X[:, idx]
                cols.append(term)
        return jnp.stack(cols, axis=1)  # (n, d_X)

    def _time_basis(self, time_grid):
        """Polynomial basis [1, t_norm, t_norm^2, ...] on normalized time.

        Returns:
            (d_T, T) basis matrix.
        """
        t_max = jnp.maximum(time_grid[-1], 1e-8)
        t_norm = time_grid / t_max  # [0, 1]
        exponents = jnp.arange(self.n_time_basis, dtype=time_grid.dtype)
        return jnp.power(t_norm[None, :], exponents[:, None])  # (d_T, T)

    def call(self, inputs):
        """Forward pass.

        Args:
            inputs: dict with keys "X" (n, p) and "time_grid" (T,).

        Returns:
            tau_pred: (n, T) predicted CATE curve.
        """
        X = inputs["X"]
        time_grid = inputs["time_grid"]

        phi_X = self._poly_features(X)      # (n, d_X)
        B_time = self._time_basis(time_grid)  # (d_T, T)

        tau_pred = phi_X @ self._theta @ B_time  # (n, T)
        return tau_pred

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, training=True):
        """Empirical R-loss (same interface as CATEMixture)."""
        from air_cate.r_loss import r_loss

        return r_loss(
            tau_pred=y_pred,
            Y_tilde=y["Y_tilde"],
            S_marg=y["S_marg"],
            W=y["W"],
            e_clipped=y["e_clipped"],
            delta=y["delta"],
        )
