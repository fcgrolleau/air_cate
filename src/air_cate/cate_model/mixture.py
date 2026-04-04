"""Covariate-gated mixture of Gamma survival differences (CATEMixture)."""

import jax.numpy as jnp
import keras

from air_cate.cate_model.encoder import FeatureEncoder
from air_cate.cate_model.gamma_survival import gamma_survival


class CATEMixture(keras.Model):
    """Parametrize tau(t|x) as a covariate-gated mixture of Gamma survival differences.

    tau(t|x; theta) = sum_k rho_k(x; beta) * TE_k(t)
    TE_k(t) = G(t; alpha_bar, lambda_{k,1}) - G(t; alpha_bar, lambda_{k,0})

    Structural constraints hold by construction:
    - tau(0|x) = 0  (Gamma survival = 1 at t=0 for both arms)
    - tau(t|x) in [-1, 1]
    - tau(t|x) -> 0 as t -> inf
    """

    def __init__(
        self,
        K: int = 4,
        encoder_hidden_sizes: tuple[int, ...] = (64, 32),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.K = K
        self.encoder = FeatureEncoder(hidden_sizes=encoder_hidden_sizes)
        self.gate_dense = keras.layers.Dense(K)

        # Learnable Gamma parameters (unconstrained, mapped via softplus)
        self.alpha_raw = self.add_weight(
            name="alpha_raw", shape=(), initializer="ones"
        )
        self.lambda_raw = self.add_weight(
            name="lambda_raw",
            shape=(K, 2),
            initializer="ones",
        )

    def call(self, inputs):
        """Forward pass.

        Args:
            inputs: dict with keys "X" (n, p) and "time_grid" (T,).

        Returns:
            tau_pred: (n, T) predicted CATE curve.
        """
        X = inputs["X"]
        time_grid = inputs["time_grid"]

        # Feature encoding + gating
        phi = self.encoder(X)  # (n, d)
        rho = keras.activations.softmax(self.gate_dense(phi), axis=-1)  # (n, K)

        # Gamma parameters (positive)
        alpha_bar = keras.activations.softplus(self.alpha_raw) + 1e-4
        lambdas = keras.activations.softplus(self.lambda_raw) + 1e-4  # (K, 2)

        # Treatment effect profiles: TE_k(t) = G(t; alpha, lambda_{k,1}) - G(t; alpha, lambda_{k,0})
        # time_grid is (T,), lambdas[:, w] is (K,)
        # G(t; alpha, lam) evaluated for each k: shape (K, T)
        G1 = gamma_survival(
            time_grid[None, :], alpha_bar, lambdas[:, 1:2]
        )  # (K, T)
        G0 = gamma_survival(
            time_grid[None, :], alpha_bar, lambdas[:, 0:1]
        )  # (K, T)
        TE = G1 - G0  # (K, T)

        # Mixture: tau = sum_k rho_k * TE_k
        tau_pred = jnp.einsum("nk,kt->nt", rho, TE)  # (n, T)

        return tau_pred

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, training=True):
        """Empirical R-loss using precomputed nuisance quantities.

        y is a dict with keys: Y_tilde, S_marg, W, e_clipped, delta.
        """
        from air_cate.r_loss import r_loss

        return r_loss(
            tau_pred=y_pred,
            Y_tilde=y["Y_tilde"],
            S_marg=y["S_marg"],
            W=y["W"],
            e_clipped=y["e_clipped"],
            delta=y["delta"],
        )
