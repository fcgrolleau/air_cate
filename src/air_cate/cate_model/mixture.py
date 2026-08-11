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

    Args:
        K: Number of mixture components.
        encoder_hidden_sizes: Hidden layer widths of the covariate encoder.
        normalize_time: Rescale the time grid to [0, 1] before evaluating the Gamma
            survival profiles. Off by default, which evaluates G at the raw times.
            Turn it on when t_end is large relative to 1/lambda: lambda is
            initialized near softplus(1.0) ~ 1.31, so with t_end = 5 the product
            lambda * t reaches ~6.5 and G has already decayed to 0.04 by t_end,
            leaving the back half of the grid nearly flat. The model can only spread
            the effect over [0, t_end] by driving lambda_raw from +1.0 down to about
            -1.2 -- roughly four initializer SDs -- using gradient from the very
            region it has flattened, which patience-based early stopping tends to
            cut short. Normalizing keeps lambda * t in [0, 1.31] instead.
            `CATETensorProduct` normalizes unconditionally.
        legacy_lambda_init: Draw lambda_raw from N(0, 1) instead of N(1.0, 0.5).
            Off by default; see the `lambda_raw` comment in `__init__` for why the
            two differ and which one the archived 100-seed run used.
    """

    def __init__(
        self,
        K: int = 4,
        encoder_hidden_sizes: tuple[int, ...] = (64, 32),
        normalize_time: bool = False,
        legacy_lambda_init: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.K = K
        self.normalize_time = normalize_time
        self.legacy_lambda_init = legacy_lambda_init
        self.encoder = FeatureEncoder(hidden_sizes=encoder_hidden_sizes)
        self.gate_dense = keras.layers.Dense(K)

        # Learnable Gamma parameters (unconstrained, mapped via softplus)
        self.alpha_raw = self.add_weight(
            name="alpha_raw", shape=(), initializer="ones"
        )
        # Random init breaks the symmetry between the two arms. With a constant
        # initializer, lambda_{k,1} == lambda_{k,0} makes TE_k(t) identically 0,
        # hence tau == 0 and d tau / d rho == 0: the encoder, the gate and
        # alpha_raw all receive exactly zero gradient, and only lambda_raw can
        # move. Escaping that state takes ~50-100 epochs, so patience-based
        # early stopping usually halts training first and every tv-CATE curve
        # collapses to 0.
        #
        # WHICH random init, though, is a separate question, and the two options
        # here are not interchangeable. lambdas = softplus(lambda_raw), and softplus
        # compresses: pushing the mean to +1.0 to keep lambda away from 0 also cuts
        # the spread of the resulting decay rates.
        #
        #                          lambda mean   lambda sd     cv    p5-p95
        #     N(0, 1.0)  legacy          0.806       0.521  0.647  [0.18, 1.82]
        #     N(1.0, 0.5) default        1.338       0.362  0.271  [0.79, 1.97]
        #
        # The default starts with 2.4x less relative spread and cannot reach the
        # slow-decay half of the range (lambda < 0.79) at all, so the K components
        # begin as near-duplicate temporal shapes and more of the fitting burden
        # falls on the gate and encoder. It is also not better at the job it was
        # introduced for: P(|lambda_{k,1} - lambda_{k,0}| < 0.1) is 0.156 against
        # legacy's 0.128.
        #
        # The default was chosen in fb6c345 while replacing `initializer="ones"`,
        # measured against that -- not against the legacy init, which is what the
        # archived 100-seed run used and which this flag restores.
        lambda_init = (
            keras.initializers.RandomNormal(stddev=1.0)
            if legacy_lambda_init
            else keras.initializers.RandomNormal(mean=1.0, stddev=0.5)
        )
        self.lambda_raw = self.add_weight(
            name="lambda_raw",
            shape=(K, 2),
            initializer=lambda_init,
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

        # Optionally map t onto [0, 1] so the Gamma scales work across DGP time
        # scales. See `normalize_time` in the class docstring.
        if self.normalize_time:
            t_eval = time_grid / jnp.maximum(time_grid[-1], 1e-6)
        else:
            t_eval = time_grid

        # Treatment effect profiles: TE_k(t) = G(t; alpha, lambda_{k,1}) - G(t; alpha, lambda_{k,0})
        # t_eval is (T,), lambdas[:, w] is (K,)
        # G(t; alpha, lam) evaluated for each k: shape (K, T)
        G1 = gamma_survival(
            t_eval[None, :], alpha_bar, lambdas[:, 1:2]
        )  # (K, T)
        G0 = gamma_survival(
            t_eval[None, :], alpha_bar, lambdas[:, 0:1]
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
