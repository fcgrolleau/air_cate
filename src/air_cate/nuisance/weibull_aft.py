"""Weibull Accelerated Failure Time model for survival nuisance estimation."""

import jax.numpy as jnp
import keras
import numpy as np


class WeibullAFT:
    """Weibull AFT survival model parameterized by an MLP.

    S(t|x) = exp(-(t / lambda(x))^{k(x)})

    The MLP outputs raw (unconstrained) shape and scale parameters which are
    mapped to positive values via softplus.
    """

    def __init__(self, hidden_sizes: list[int] = (64, 32)):
        self.hidden_sizes = hidden_sizes
        self.model_ = None

    def _build_model(self, input_dim: int) -> keras.Model:
        inputs = keras.Input(shape=(input_dim,))
        x = inputs
        for units in self.hidden_sizes:
            x = keras.layers.Dense(units, activation="relu")(x)
        # Two raw outputs: log-shape and log-scale (mapped via softplus)
        raw_params = keras.layers.Dense(2)(x)
        return keras.Model(inputs, raw_params)

    @staticmethod
    def _get_params(raw_params):
        """Map raw MLP outputs to positive shape k and scale lambda."""
        k = keras.activations.softplus(raw_params[:, 0]) + 1e-4
        lam = keras.activations.softplus(raw_params[:, 1]) + 1e-4
        return k, lam

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        event: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        verbose: int = 0,
    ) -> "WeibullAFT":
        """Fit Weibull AFT by maximizing the censored likelihood.

        Args:
            X: Covariates (n, p).
            T: Observed times (n,). Must be > 0.
            event: Event indicator (n,). 1 = event observed, 0 = censored.
        """
        self.model_ = self._build_model(X.shape[1])

        # Custom training step via a wrapper model
        weibull_model = _WeibullTrainModel(self.model_)
        weibull_model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))

        # Pack targets as (T, event) matrix
        y = np.stack([T, event], axis=-1).astype(np.float32)

        weibull_model.fit(
            X.astype(np.float32),
            y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="loss")],
        )
        return self

    def survival(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Evaluate S(t|X) = exp(-(t/lambda(X))^k(X)).

        Args:
            t: Time points. Either scalar, (T,), or broadcastable with X.
            X: Covariates (n, p).

        Returns:
            S: Survival probabilities. Shape depends on inputs:
               - If t is (T,) and X is (n, p): returns (n, T).
               - If t is scalar or (n,): returns (n,).
        """
        raw_params = self.model_(np.asarray(X, dtype=np.float32), training=False)
        k, lam = self._get_params(raw_params)

        t = jnp.asarray(t, dtype=jnp.float32)
        k = jnp.asarray(k)
        lam = jnp.asarray(lam)

        if t.ndim == 1 and t.shape[0] != X.shape[0]:
            # Grid evaluation: t is (T,), result is (n, T)
            # log S = -(t / lambda)^k  computed in log-space for stability
            # (n, 1) and (1, T) -> (n, T)
            log_ratio = jnp.log(jnp.maximum(t[None, :], 1e-30)) - jnp.log(lam[:, None])
            log_S = -jnp.exp(k[:, None] * log_ratio)
            return jnp.exp(log_S)
        else:
            # Point evaluation: t broadcastable with (n,)
            log_ratio = jnp.log(jnp.maximum(t, 1e-30)) - jnp.log(lam)
            log_S = -jnp.exp(k * log_ratio)
            return jnp.exp(log_S)


class _WeibullTrainModel(keras.Model):
    """Wrapper that computes Weibull NLL as the training loss."""

    def __init__(self, backbone: keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone

    def call(self, inputs, training=False):
        return self.backbone(inputs, training=training)

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, training=True):
        """Weibull negative log-likelihood for right-censored data.

        y[:, 0] = observed time T
        y[:, 1] = event indicator (1=event, 0=censored)
        y_pred = raw MLP output (n, 2)
        """
        T = y[:, 0]
        event = y[:, 1]

        k = keras.activations.softplus(y_pred[:, 0]) + 1e-4
        lam = keras.activations.softplus(y_pred[:, 1]) + 1e-4

        # log S(t|x) = -(t/lambda)^k
        log_ratio = keras.ops.log(keras.ops.maximum(T, 1e-30)) - keras.ops.log(lam)
        log_S = -keras.ops.exp(k * log_ratio)

        # log f(t|x) = log k + (k-1)*log(t) - k*log(lambda) - (t/lambda)^k
        log_f = (
            keras.ops.log(k)
            + (k - 1) * keras.ops.log(keras.ops.maximum(T, 1e-30))
            - k * keras.ops.log(lam)
            + log_S
        )

        # NLL = -sum[ event * log_f + (1-event) * log_S ]
        nll = -keras.ops.mean(event * log_f + (1.0 - event) * log_S)
        return nll
