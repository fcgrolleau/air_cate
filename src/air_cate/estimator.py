"""Top-level AIREstimator: orchestrates nuisance estimation, pseudo-outcomes, and CATE model."""

import jax.numpy as jnp
import keras
import numpy as np

from air_cate._types import SurvivalData
from air_cate.cate_model.mixture import CATEMixture
from air_cate.cate_model.train import train_cate
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
        patience: Epochs without training-loss improvement before early stopping.
            None trains the full `cate_epochs`.
        rng_seed: Seed for minibatch shuffling and, with `n_restarts > 1`, for the
            per-restart weight initialization. None uses fresh entropy.
        n_restarts: Number of independent CATE fits from different initializations,
            keeping the one with the lowest training R-loss. **Defaults to 1, and
            raising it is not recommended** -- it is retained for diagnostics.

            The R-loss is non-convex in the mixture parameters, so best-of-N looks
            like cheap insurance against a bad local optimum. Measured on this
            implementation, it is not:

            - There is nothing to escape. Across 5 restarts the training losses
              differ by ~0.7% (e.g. 2.4833 to 2.4973); every initialization reaches
              an equally good optimum. Check `restart_losses_` before assuming
              otherwise.
            - Selection has no upside and plausibly a downside. `fit` holds nothing
              out, so restarts are ranked by *training* loss -- which ranks partly by
              how well each fit the noise. On a null DGP (true tau == 0, so every
              nonzero value is artefact) mean max|tau_hat| was 0.187 at 1 restart
              (15 draws, sd 0.077) against 0.261 at 5 (5 draws): directionally worse,
              but ~1.8 standard errors, so suggestive rather than established. The
              case for 1 rests on the point above, not on this one.

            The apparent evidence for restarts, a right tail in iMSE with occasional
            seeds near 2x the median, is unrelated: that tail is equally present in
            runs that used 5 restarts, so it is the sampling distribution of iMSE
            across data draws, not optimizer failure.

            Costs about `n_restarts` times the CATE stage. The nuisance stage is
            unaffected, so the multiplier is smaller for cross-fitted fits than for
            ones given oracle nuisances.

    Attributes (after fit):
        cate_model_: The fitted CATE model (the winning restart).
        time_grid_, delta_: The time grid and its step size.
        history_: dict returned by `train_cate` for the winning restart (per-epoch
            loss trajectory). `plot_convergence()` draws it.
        restart_losses_: best training R-loss of each restart, in order. A wide
            spread here is the diagnostic that restarts are earning their cost.
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
        patience: int | None = 10,
        rng_seed: int | None = None,
        n_restarts: int = 1,
        cate_model_class=None,
        cate_model_kwargs: dict | None = None,
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
        self.patience = patience
        self.rng_seed = rng_seed
        if n_restarts < 1:
            raise ValueError(f"n_restarts must be >= 1, got {n_restarts}")
        self.n_restarts = n_restarts
        self.cate_model_class = cate_model_class
        self.cate_model_kwargs = cate_model_kwargs or {}

        self.cate_model_ = None
        self.time_grid_ = None
        self.delta_ = None
        self.history_ = None
        self.restart_losses_ = None

    def fit(
        self,
        X: np.ndarray,
        W: np.ndarray,
        Z: np.ndarray,
        Delta: np.ndarray,
        t_end: float,
        nuisances=None,
        verbose: int = 0,
    ) -> "AIREstimator":
        """Fit the AIR estimator.

        Args:
            X: Covariates (n, p).
            W: Binary treatment (n,).
            Z: Observed times min(D, C) (n,).
            Delta: Event indicator 1{D <= C} (n,).
            t_end: Maximum follow-up time.
            nuisances: Pre-computed NuisanceEstimates (skip cross-fitting if provided).
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

        # Step 2: Cross-fit nuisance functions (or use provided oracle)
        if nuisances is None:
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

        # Step 4: Build and train the CATE model. With the default n_restarts=1 this
        # is a single fit and the loop is a no-op.
        #
        # Selection is on TRAINING R-loss because `fit` holds nothing out, so there
        # is no honest held-out score to rank by -- and scoring restarts on a block
        # that then also reports the result would leak. That is precisely why
        # n_restarts > 1 is not recommended: ranking by training loss ranks partly by
        # noise-fit. See the `n_restarts` entry in the class docstring.
        best_loss = float("inf")
        self.cate_model_ = None
        self.history_ = None
        self.restart_losses_ = []

        for restart in range(self.n_restarts):
            # Distinct initializations are the entire point of restarting. With
            # rng_seed set, seed Keras per restart so the *set* of restarts is
            # reproducible; with rng_seed None, leave the global RNG untouched and
            # let it advance on its own, which also gives distinct draws.
            restart_seed = None if self.rng_seed is None else self.rng_seed + restart
            if restart_seed is not None:
                keras.utils.set_random_seed(restart_seed)

            if self.cate_model_class is not None:
                model = self.cate_model_class(**self.cate_model_kwargs)
            else:
                model = CATEMixture(
                    K=self.K,
                    encoder_hidden_sizes=self.encoder_hidden_sizes,
                )

            # A custom training loop is needed because Keras model.fit() with dict x
            # and dict y requires a Dataset that preserves the structure.
            history = train_cate(
                model,
                X=data.X,
                Y_tilde=Y_tilde,
                S_marg=S_marg,
                W=data.W,
                e_clipped=e_clipped,
                time_grid=self.time_grid_,
                delta=self.delta_,
                epochs=self.cate_epochs,
                lr=self.cate_lr,
                batch_size=self.cate_batch_size,
                weight_decay=self.weight_decay,
                monitor="train",
                patience=self.patience,
                rng_seed=restart_seed,
                verbose=verbose,
            )

            loss = history["best_train_loss"]
            self.restart_losses_.append(loss)

            if loss < best_loss:
                best_loss = loss
                # Retain the model object rather than copying its weights out:
                # every restart builds a fresh model, so at most two are ever live.
                self.cate_model_ = model
                self.history_ = history
            else:
                del model

            if verbose > 0 and self.n_restarts > 1:
                print(
                    f"  Restart {restart}: train loss={loss:.6f} (best={best_loss:.6f})"
                )

        return self

    def plot_convergence(self, **kwargs):
        """Plot the training AIR CATE loss against epoch. See `air_cate.plotting`.

        A convergence check on the CATE stage: if the curve is still descending at
        the last epoch, `cate_epochs` was too small. Because `fit` trains with no
        validation split, this is a training curve only -- it says whether the
        optimizer converged, not whether the chosen K generalizes. Use
        `AIREstimatorCV.plot_convergence_grid` for held-out curves.

        Requires matplotlib, which ships in the optional [plots] extra:
        `uv sync --extra plots`.

        Args:
            **kwargs: Forwarded to `air_cate.plotting.plot_convergence`
                (`ax`, `log_y`, `mark_best`, `title`, `figsize`).

        Returns:
            (fig, ax)
        """
        if self.history_ is None:
            raise RuntimeError("Call fit() before plot_convergence().")

        from air_cate.plotting import plot_convergence

        return plot_convergence(self.history_, **kwargs)

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
