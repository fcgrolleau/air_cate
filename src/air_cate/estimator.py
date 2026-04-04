"""Top-level AIREstimator: orchestrates nuisance estimation, pseudo-outcomes, and CATE model."""

import jax.numpy as jnp
import keras
import numpy as np

from air_cate._types import SurvivalData
from air_cate.cate_model.mixture import CATEMixture
from air_cate.nuisance.cross_fit import cross_fit_nuisances
from air_cate.pseudo_outcome import compute_pseudo_outcomes
from air_cate.utils import build_time_grid


class AIREstimator:
    """Augmented and Integrated R-loss estimator for time-varying CATE.

    Implements Algorithm 1 from the paper:
    1. Cross-fit nuisance functions (propensity, event/censoring survival).
    2. Compute AIPCW pseudo-outcomes on a time grid.
    3. Minimize the empirical R-loss over a CATEMixture model.

    Args:
        K: Number of mixture components for the CATE model.
        M: Number of grid intervals (M+1 grid points).
        n_folds: Number of cross-fitting folds.
        eps: Clipping bound for nuisance estimates.
        encoder_hidden_sizes: Hidden layer sizes for the feature encoder.
        propensity_config: kwargs for PropensityMLP constructor.
        weibull_config: kwargs for WeibullAFT constructor.
        nuisance_training_config: Training kwargs for nuisance models.
        cate_epochs: Training epochs for the CATE model.
        cate_lr: Learning rate for the CATE model.
        cate_batch_size: Batch size for the CATE model.
        weight_decay: Weight decay for AdamW optimizer.
    """

    def __init__(
        self,
        K: int = 4,
        M: int = 100,
        n_folds: int = 5,
        eps: float = 0.01,
        encoder_hidden_sizes: tuple[int, ...] = (64, 32),
        propensity_config: dict | None = None,
        weibull_config: dict | None = None,
        nuisance_training_config: dict | None = None,
        cate_epochs: int = 200,
        cate_lr: float = 1e-3,
        cate_batch_size: int = 256,
        weight_decay: float = 1e-4,
    ):
        self.K = K
        self.M = M
        self.n_folds = n_folds
        self.eps = eps
        self.encoder_hidden_sizes = encoder_hidden_sizes
        self.propensity_config = propensity_config
        self.weibull_config = weibull_config
        self.nuisance_training_config = nuisance_training_config
        self.cate_epochs = cate_epochs
        self.cate_lr = cate_lr
        self.cate_batch_size = cate_batch_size
        self.weight_decay = weight_decay

        self.cate_model_ = None
        self.time_grid_ = None
        self.delta_ = None

    def fit(
        self,
        X: np.ndarray,
        W: np.ndarray,
        Z: np.ndarray,
        Delta: np.ndarray,
        t_end: float,
        verbose: int = 0,
    ) -> "AIREstimator":
        """Fit the AIR estimator.

        Args:
            X: Covariates (n, p).
            W: Binary treatment (n,).
            Z: Observed times min(D, C) (n,).
            Delta: Event indicator 1{D <= C} (n,).
            t_end: Maximum follow-up time.
            verbose: Verbosity level (0=silent).

        Returns:
            self
        """
        # Step 1: Build time grid
        self.time_grid_, self.delta_ = build_time_grid(t_end, self.M)

        data = SurvivalData(
            X=jnp.array(X, dtype=jnp.float32),
            W=jnp.array(W, dtype=jnp.float32),
            Z=jnp.array(Z, dtype=jnp.float32),
            Delta=jnp.array(Delta, dtype=jnp.float32),
        )

        # Step 2: Cross-fit nuisance functions
        nuisances = cross_fit_nuisances(
            data,
            self.time_grid_,
            n_folds=self.n_folds,
            propensity_config=self.propensity_config,
            weibull_config=self.weibull_config,
            training_config=self.nuisance_training_config,
        )

        # Step 3: Compute pseudo-outcomes
        Y_tilde, S_marg, e_clipped = compute_pseudo_outcomes(
            data, nuisances, self.time_grid_, self.delta_, eps=self.eps
        )

        # Step 4: Build and train CATEMixture
        self.cate_model_ = CATEMixture(
            K=self.K,
            encoder_hidden_sizes=self.encoder_hidden_sizes,
        )

        self.cate_model_.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=self.cate_lr,
                weight_decay=self.weight_decay,
            ),
        )

        # Prepare training data: x is dict with X and time_grid, y is dict of nuisance quantities
        x_train = {
            "X": np.asarray(data.X),
            "time_grid": np.asarray(self.time_grid_),
        }
        y_train = {
            "Y_tilde": np.asarray(Y_tilde),
            "S_marg": np.asarray(S_marg),
            "W": np.asarray(data.W),
            "e_clipped": np.asarray(e_clipped),
            "delta": self.delta_,
        }

        # Use a custom training loop since Keras model.fit() with dict y
        # and dict x requires a Dataset that preserves the structure
        self._train_cate(x_train, y_train, verbose=verbose)

        return self

    def _train_cate(self, x_train: dict, y_train: dict, verbose: int = 0):
        """Custom training loop for CATEMixture."""
        import jax

        n = x_train["X"].shape[0]
        X_jax = jnp.array(x_train["X"])
        time_grid_jax = jnp.array(x_train["time_grid"])
        Y_tilde_jax = jnp.array(y_train["Y_tilde"])
        S_marg_jax = jnp.array(y_train["S_marg"])
        W_jax = jnp.array(y_train["W"])
        e_clipped_jax = jnp.array(y_train["e_clipped"])
        delta = float(y_train["delta"])

        optimizer = keras.optimizers.AdamW(
            learning_rate=self.cate_lr,
            weight_decay=self.weight_decay,
        )
        # Build the model by calling it once
        self.cate_model_({"X": X_jax[:1], "time_grid": time_grid_jax})
        optimizer.build(self.cate_model_.trainable_variables)

        from air_cate.r_loss import r_loss as r_loss_fn

        def _extract_vars(keras_vars):
            """Extract JAX arrays from Keras variables."""
            return [jnp.array(v.value) for v in keras_vars]

        def _assign_vars(keras_vars, values):
            """Assign JAX arrays back to Keras variables."""
            for var, val in zip(keras_vars, values):
                var.assign(val)

        @jax.jit
        def train_step(trainable_vars, non_trainable_vars, opt_vars, X_batch, Y_tilde_batch, S_marg_batch, W_batch, e_batch):
            def loss_fn(trainable_vars):
                tau_pred, _ = self.cate_model_.stateless_call(
                    trainable_vars, non_trainable_vars,
                    {"X": X_batch, "time_grid": time_grid_jax},
                )
                return r_loss_fn(tau_pred, Y_tilde_batch, S_marg_batch, W_batch, e_batch, delta)

            loss, grads = jax.value_and_grad(loss_fn)(trainable_vars)
            trainable_vars, opt_vars = optimizer.stateless_apply(opt_vars, grads, trainable_vars)
            return loss, trainable_vars, opt_vars

        best_loss = float("inf")
        patience_counter = 0
        patience = 10
        best_weights = None

        for epoch in range(self.cate_epochs):
            perm = np.random.permutation(n)
            epoch_losses = []

            for start in range(0, n, self.cate_batch_size):
                idx = perm[start : start + self.cate_batch_size]

                # Extract pure JAX arrays from Keras variables
                tv = _extract_vars(self.cate_model_.trainable_variables)
                ntv = _extract_vars(self.cate_model_.non_trainable_variables)
                ov = _extract_vars(optimizer.variables)

                loss, new_tv, new_ov = train_step(
                    tv, ntv, ov,
                    X_jax[idx],
                    Y_tilde_jax[idx],
                    S_marg_jax[idx],
                    W_jax[idx],
                    e_clipped_jax[idx],
                )

                _assign_vars(self.cate_model_.trainable_variables, new_tv)
                _assign_vars(optimizer.variables, new_ov)

                epoch_losses.append(float(loss))

            mean_loss = np.mean(epoch_losses)

            if verbose > 0 and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.cate_epochs}, loss: {mean_loss:.6f}")

            if mean_loss < best_loss - 1e-6:
                best_loss = mean_loss
                patience_counter = 0
                best_weights = _extract_vars(self.cate_model_.trainable_variables)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose > 0:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

        if best_weights is not None:
            _assign_vars(self.cate_model_.trainable_variables, best_weights)

    def predict_tau(self, X: np.ndarray) -> np.ndarray:
        """Predict tau(t|x) on the training time grid.

        Args:
            X: Covariates (n, p).

        Returns:
            tau_hat: (n, M+1) predicted CATE curves.
        """
        X_jax = jnp.array(X, dtype=jnp.float32)
        tau = self.cate_model_({"X": X_jax, "time_grid": self.time_grid_})
        return np.asarray(tau)

    def predict_gamma(self, X: np.ndarray) -> np.ndarray:
        """Predict CRMST difference Gamma(t|x) = cumulative sum of delta * tau.

        Args:
            X: Covariates (n, p).

        Returns:
            gamma_hat: (n, M+1) CRMST differences.
        """
        tau = self.predict_tau(X)
        gamma = np.cumsum(self.delta_ * tau, axis=1)
        return gamma
