"""Weibull Accelerated Failure Time model for survival nuisance estimation."""

import jax.numpy as jnp
import keras
import numpy as np


class _GlobalScalar(keras.layers.Layer):
    """A single learnable scalar, broadcast over the batch, ignoring its input.

    Used to turn one output head of the Weibull MLP into a parameter shared by
    every subject. The input is read only for its batch dimension.
    """

    def build(self, input_shape):
        self.b = self.add_weight(name="b", shape=(1,), initializer="zeros")

    def call(self, inputs):
        return keras.ops.zeros_like(inputs[:, :1]) + self.b


class WeibullAFT:
    """Weibull AFT survival model parameterized by an MLP.

    S(t|x) = exp(-(t / lambda(x))^{k(x)})

    The MLP outputs raw (unconstrained) shape and scale parameters which are
    mapped to positive values via softplus.

    Args:
        hidden_sizes: Hidden layer widths of the MLP.
        shared_shape: Fit one shape parameter for the whole sample instead of a
            per-subject k(x). Off by default, which reproduces the original
            behaviour. Turning it on is what "accelerated failure time" actually
            means: an AFT model is log T = mu(x) + sigma * W with sigma
            *constant*, so covariates shift the location of log T and nothing
            else. Letting k vary with x instead gives a heteroscedastic Weibull
            regression, whose extra freedom lands exactly where the AIPCW
            pseudo-outcomes are most fragile -- a subject handed a small lambda
            and a large k gets S(t) ~ 0 at late t, and 1/S(t) enters the
            pseudo-outcome directly. Worth turning on for the censoring model
            whenever the censoring shape is not believed to vary with x; leave it
            off for an event process with covariate-dependent shape.
        normalize_time: Divide T by a scale (default max(T)) before fitting, and
            divide every `survival` query by the same scale, so the fitted parameters
            live on a unit time axis. ON by default, because the un-normalized
            behaviour is a silent trap rather than a neutral alternative.

            k and lambda are two heads of the SAME MLP. On a time axis in days with
            t_end = 1825, lambda must reach ~1e3 while k needs to be ~1.4 -- three
            orders of magnitude apart from one linear layer. The lambda gradient
            dominates and the shape head collapses onto its 0.3 floor. Measured on a
            Stanford CAD cohort (n = 2909, 10.4% events), fitting on raw days returned
            k = 0.300 exactly -- the floor -- with lambda = 1081, giving mean S(t) of
            0.557 / 0.487 / 0.368 / 0.312 at 6 months / 1 / 3 / 5 years against
            Kaplan-Meier 0.922 / 0.907 / 0.859 / 0.812, i.e. 38-60% of the truth.
            The same data on t/1825 returned k = 5.22, lambda = 2.25 and mean S(t) of
            0.906 / 0.876 / 0.793 / 0.714 -- within 2-12% of KM.

            The downstream cost is not a modest bias. Too small an S_cens inflates the
            IPCW weight 1/S_cens, which pushed the AIPCW pseudo-outcome's mean above 1
            -- impossible for a survival probability -- and broke the R-learner's
            orthogonalization: the residual Y_tilde - S_marg had mean 1.20 at 5 years
            where it must be ~0. The tv-R-loss's implied homogeneous effect then came
            out anywhere from -0.67 to +0.62 depending only on the clipping floor
            `eps`, against a Kaplan-Meier difference of +0.014.

            This is the same failure mode `CATEMixture.normalize_time` exists for.
    """

    def __init__(self, hidden_sizes: list[int] = (64, 32), shared_shape: bool = False,
                 normalize_time: bool = True):
        self.hidden_sizes = hidden_sizes
        self.shared_shape = shared_shape
        self.normalize_time = normalize_time
        self.model_ = None
        # Set by `fit`; 1.0 means "no rescaling", so `survival` is safe either way.
        self.t_scale_ = 1.0

    def _build_model(self, input_dim: int) -> keras.Model:
        inputs = keras.Input(shape=(input_dim,))
        x = inputs
        for units in self.hidden_sizes:
            x = keras.layers.Dense(units, activation="relu")(x)
        if self.shared_shape:
            # Scale still depends on x; shape is one parameter for everyone.
            # Column order must stay (shape, scale) to match `_get_params`.
            lam_raw = keras.layers.Dense(1)(x)
            k_raw = _GlobalScalar()(lam_raw)
            raw_params = keras.layers.Concatenate()([k_raw, lam_raw])
        else:
            # Two raw outputs: log-shape and log-scale (mapped via softplus)
            raw_params = keras.layers.Dense(2)(x)
        return keras.Model(inputs, raw_params)

    @staticmethod
    def _get_params(raw_params):
        """Map raw MLP outputs to positive shape k and scale lambda.

        The shape floor is 0.3 rather than 1e-4: as k -> 0 the Weibull hazard
        diverges at t = 0 and the log-likelihood gradient blows up, which
        destabilizes the IST-3 nuisance fits. Keep this in sync with the
        identical mapping in `_WeibullTrainModel.compute_loss`.
        """
        k = keras.activations.softplus(raw_params[:, 0]) + 0.3
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
        t_scale: float | None = None,
    ) -> "WeibullAFT":
        """Fit Weibull AFT by maximizing the censored likelihood.

        Args:
            X: Covariates (n, p).
            T: Observed times (n,). Must be > 0.
            event: Event indicator (n,). 1 = event observed, 0 = censored.
            t_scale: Time scale to divide T by when `normalize_time`; defaults to
                max(T). Pass it explicitly to pin ONE scale across cross-fitting
                folds -- otherwise each fold normalizes by its own maximum, which is
                harmless when the follow-up is administratively capped (every fold
                then sees the same maximum) and a source of fold-to-fold drift when
                it is not.
        """
        T = np.asarray(T, dtype=np.float64)
        if self.normalize_time:
            scale = float(np.max(T)) if t_scale is None else float(t_scale)
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    f"t_scale must be finite and positive, got {scale!r}"
                )
            self.t_scale_ = scale
        else:
            self.t_scale_ = 1.0
        T = T / self.t_scale_

        self.model_ = self._build_model(X.shape[1])

        # Custom training step via a wrapper model
        weibull_model = _WeibullTrainModel(self.model_)
        weibull_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        )

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

        # Queries arrive on the ORIGINAL time scale, so rescale them exactly as `fit`
        # did. Callers therefore never see the normalization, and t_scale_ = 1.0
        # reproduces the un-normalized behaviour.
        t = jnp.asarray(t, dtype=jnp.float32) / jnp.asarray(
            self.t_scale_, dtype=jnp.float32
        )
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

        # Must match WeibullAFT._get_params: shape floor 0.3, scale floor 1e-4.
        k = keras.activations.softplus(y_pred[:, 0]) + 0.3
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
