"""Top-level AIREstimator: orchestrates nuisance estimation, pseudo-outcomes, and CATE model."""

import math

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
        censoring_config: kwargs for the censoring WeibullAFT models; defaults to
            `weibull_config`. Pass `{"shared_shape": True}` to fit one censoring
            shape for the whole sample.
        admin_as_censored: Treat administrative end-of-follow-up (Z at t_end) as
            censored, not as an observed censoring event, when fitting the
            censoring model. Off by default. See `cross_fit_nuisances`.
        nuisance_training_config: Training kwargs for nuisance models.
        cate_epochs: Training epochs for the CATE model.
        cate_lr: Learning rate for the CATE model.
        cate_batch_size: Batch size for the CATE model.
        weight_decay: Weight decay for AdamW optimizer.
        patience: Epochs without improvement in the monitored loss before early
            stopping. None trains the full `cate_epochs`. Note that at
            `cate_val_frac=0` the monitored loss is the *training* R-loss, which falls
            monotonically, so patience effectively never fires.
        cate_val_frac: Fraction of rows held out of the CATE fit to score early
            stopping honestly. 0 (the default) holds nothing out and trains for the
            full `cate_epochs`; a value like 0.2 makes `patience` a real stopping rule
            at the cost of training the CATE model on fewer rows. The nuisances and
            pseudo-outcomes always use every row. Held-out R-loss is the same
            criterion `AIREstimatorCV` uses to select K.
        cate_refit_full: After the held-out split has selected a stopping point,
            discard that model and refit from scratch on **all** n rows for the same
            amount of training. Requires `cate_val_frac > 0`. This is the
            refit-after-selection step `GridSearchCV(refit=True)` performs: the split
            earns an honest epoch count, and the final model is not left paying for it
            with `cate_val_frac` of the data.

            The budget is matched in **optimizer steps, not epochs**. The stopping
            point was measured at `ceil((1 - cate_val_frac) * n / batch)` updates per
            epoch; all n rows give `ceil(n / batch)`, a factor `1/(1 - cate_val_frac)`
            more. Reusing the epoch count verbatim would train ~25% longer than the
            held-out block ever endorsed -- reintroducing exactly the over-training
            the split exists to prevent. See `refit_epochs_`.

            Costs one extra CATE fit. Measured on IST-3 at n=500, 20 paired seeds:
            +9% wall clock cross-fitted (the five nuisance models across three folds
            dominate) and +33% with oracle nuisances, where the CATE fit is the whole
            cost. Cheaper than the epoch count suggests because the refit runs only
            ~(1 - cate_val_frac) of the selected epochs.

            Left off by default because it does not measurably help. IST-3, 20 paired
            seeds, against the same fits without it:

                n=500   AIR 0.3300 -> 0.3298 (t=-0.01)  oracle 0.1879 -> 0.1891 (t=+0.13)
                n=2000  AIR 0.4106 -> 0.3994 (t=-0.30)  oracle 0.1486 -> 0.1521 (t=+0.62)

            n=500 is the strongest test -- the split spends 100 of 500 rows there --
            and it is the flattest result. So the split's benefit is entirely in *when*
            training stops, not in the rows it costs: at these sample sizes the CATE
            model does not miss the held-out fifth. Kept as an opt-in because that is a
            claim worth being able to re-check, not because enabling it should pay.
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
        cate_model_: The fitted CATE model. The winning restart, or -- with
            `cate_refit_full` -- the full-data refit of it.
        time_grid_, delta_: The time grid and its step size.
        history_: dict returned by `train_cate` for the winning restart's
            **selection** fit (per-epoch loss trajectory, and the val curve when
            there is one). `plot_convergence()` draws it. Note that under
            `cate_refit_full` this describes the fit that chose the epoch count, not
            the one that produced `cate_model_` -- see `refit_history_`.
        refit_history_: dict returned by `train_cate` for the full-data refit, or
            None when `cate_refit_full` is False.
        refit_epochs_: number of epochs the refit ran, or None. Lower than the
            selection fit's `best_epoch + 1` because the budget is matched in
            optimizer steps and the full data has more steps per epoch.
        restart_losses_: best monitored R-loss of each restart, in order -- held-out
            when `cate_val_frac > 0`, training loss otherwise. A wide spread here is
            the diagnostic that restarts are earning their cost.
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
        censoring_config: dict | None = None,
        admin_as_censored: bool = False,
        nuisance_training_config: dict | None = None,
        cate_epochs: int = 200,
        cate_lr: float = 1e-3,
        cate_batch_size: int = 256,
        weight_decay: float = 1e-4,
        patience: int | None = 10,
        cate_val_frac: float = 0.0,
        cate_refit_full: bool = False,
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
        self.censoring_config = censoring_config
        self.admin_as_censored = admin_as_censored
        self.nuisance_training_config = nuisance_training_config
        self.cate_epochs = cate_epochs
        self.cate_lr = cate_lr
        self.cate_batch_size = cate_batch_size
        self.weight_decay = weight_decay
        self.patience = patience
        if not 0.0 <= cate_val_frac < 1.0:
            raise ValueError(
                f"cate_val_frac must be in [0, 1), got {cate_val_frac}"
            )
        self.cate_val_frac = cate_val_frac
        if cate_refit_full and cate_val_frac == 0.0:
            raise ValueError(
                "cate_refit_full=True requires cate_val_frac > 0: with nothing held "
                "out there is no honest stopping point to refit at."
            )
        self.cate_refit_full = cate_refit_full
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
        self.refit_history_ = None
        self.refit_epochs_ = None
        self.restart_losses_ = None

    def _build_cate_model(self):
        """Construct a fresh, unfitted CATE model from the configured options."""
        if self.cate_model_class is not None:
            return self.cate_model_class(**self.cate_model_kwargs)
        # cate_model_kwargs also applies to the default model, so options like
        # normalize_time can be set without naming the class and restating K /
        # encoder_hidden_sizes. Explicit keys win.
        return CATEMixture(
            K=self.K,
            encoder_hidden_sizes=self.encoder_hidden_sizes,
            **self.cate_model_kwargs,
        )

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
                censoring_config=self.censoring_config,
                admin_as_censored=self.admin_as_censored,
                t_end=t_end,
                training_config=self.nuisance_training_config,
            )

        # Step 3: Compute pseudo-outcomes
        Y_tilde, S_marg, e_clipped = compute_pseudo_outcomes(
            data, nuisances, self.time_grid_, self.delta_, eps=self.eps
        )

        # Step 3b: optional honest validation split for the CATE stage.
        #
        # Without one, `patience` cannot do anything. The monitored quantity is the
        # *training* R-loss, which keeps falling for as long as the model has capacity
        # to fit noise, so early stopping never fires and every fit runs the full
        # `cate_epochs`. That is not a hypothetical cost: on the IST-3 DGP with oracle
        # nuisances, 20 paired seeds at n=2000 gave iMSE 0.192 at 200 epochs against
        # 0.147 at 50 -- a 27% penalty paid purely to over-training.
        #
        # Held-out R-loss is a legitimate stopping criterion for the same reason it is
        # a legitimate model-selection criterion in AIREstimatorCV's K search: the
        # R-loss *is* the estimation objective, so its value on unseen rows measures
        # fit honestly. The price is that the CATE model trains on (1 - cate_val_frac)
        # of the rows; the nuisances and pseudo-outcomes are still built from all of
        # them, so no information is discarded upstream.
        cate_fit_idx = None
        validation_data = None
        if self.cate_val_frac > 0.0:
            n = data.X.shape[0]
            n_val = int(round(n * self.cate_val_frac))
            if n_val < 1 or n - n_val < 1:
                raise ValueError(
                    f"cate_val_frac={self.cate_val_frac} leaves {n_val} validation and "
                    f"{n - n_val} training rows out of {n}; both must be >= 1."
                )
            # Seeded from rng_seed but drawn with a private generator, so the split is
            # identical across restarts. Restarts must differ only in initialization;
            # if they were also scored on different rows, the comparison between them
            # would be partly a comparison of validation sets.
            perm = np.random.default_rng(
                0 if self.rng_seed is None else self.rng_seed
            ).permutation(n)
            val_idx, cate_fit_idx = perm[:n_val], perm[n_val:]
            validation_data = {
                "X": data.X[val_idx],
                "Y_tilde": Y_tilde[val_idx],
                "S_marg": S_marg[val_idx],
                "W": data.W[val_idx],
                "e_clipped": e_clipped[val_idx],
            }

        if cate_fit_idx is None:
            X_fit, Y_fit, S_fit, W_fit, e_fit = (
                data.X, Y_tilde, S_marg, data.W, e_clipped,
            )
        else:
            X_fit, Y_fit, S_fit, W_fit, e_fit = (
                data.X[cate_fit_idx], Y_tilde[cate_fit_idx], S_marg[cate_fit_idx],
                data.W[cate_fit_idx], e_clipped[cate_fit_idx],
            )

        # Step 4: Build and train the CATE model. With the default n_restarts=1 this
        # is a single fit and the loop is a no-op.
        #
        # Without a validation split, selection falls back to TRAINING R-loss, since
        # nothing is held out and scoring restarts on a block that then also reports
        # the result would leak. That is precisely why n_restarts > 1 is not
        # recommended at cate_val_frac=0: ranking by training loss ranks partly by
        # noise-fit. With cate_val_frac > 0 the ranking is honest. See the
        # `n_restarts` entry in the class docstring.
        best_loss = float("inf")
        best_restart = 0
        self.cate_model_ = None
        self.history_ = None
        self.refit_history_ = None
        self.refit_epochs_ = None
        self.restart_losses_ = []

        for restart in range(self.n_restarts):
            # Distinct initializations are the entire point of restarting. With
            # rng_seed set, seed Keras per restart so the *set* of restarts is
            # reproducible; with rng_seed None, leave the global RNG untouched and
            # let it advance on its own, which also gives distinct draws.
            restart_seed = None if self.rng_seed is None else self.rng_seed + restart
            if restart_seed is not None:
                keras.utils.set_random_seed(restart_seed)

            model = self._build_cate_model()

            # A custom training loop is needed because Keras model.fit() with dict x
            # and dict y requires a Dataset that preserves the structure.
            history = train_cate(
                model,
                X=X_fit,
                Y_tilde=Y_fit,
                S_marg=S_fit,
                W=W_fit,
                e_clipped=e_fit,
                time_grid=self.time_grid_,
                delta=self.delta_,
                epochs=self.cate_epochs,
                lr=self.cate_lr,
                batch_size=self.cate_batch_size,
                weight_decay=self.weight_decay,
                validation_data=validation_data,
                # "auto" resolves to "val" when a split exists and "train" otherwise,
                # so this is the one switch that turns `patience` from inert into a
                # real stopping rule.
                monitor="auto",
                patience=self.patience,
                rng_seed=restart_seed,
                verbose=verbose,
            )

            # Rank restarts by whatever early stopping itself selected the epoch on:
            # the held-out loss when there is one, the training loss otherwise. Mixing
            # the two would rank a val-stopped fit against a train-stopped one on
            # incomparable scales.
            loss = (
                history["best_val_loss"]
                if validation_data is not None
                else history["best_train_loss"]
            )
            self.restart_losses_.append(loss)

            if loss < best_loss:
                best_loss = loss
                best_restart = restart
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

        # Step 5: optional refit on all n rows at the selected stopping point.
        #
        # The selection fit earned an honest epoch count on held-out R-loss but paid
        # for it by training on only (1 - cate_val_frac) of the rows. Refitting keeps
        # the stopping point and gives those rows back -- the refit-after-selection
        # step GridSearchCV(refit=True) performs once its search ends.
        #
        # The budget is matched in optimizer STEPS, not epochs. `best_epoch` was
        # measured at ceil(n_fit / B) updates per epoch; all n rows give ceil(n / B),
        # a factor 1/(1 - cate_val_frac) more. Reusing the epoch count verbatim would
        # train ~25% longer at cate_val_frac=0.2 than the held-out block ever
        # endorsed, quietly reintroducing the over-training that this whole mechanism
        # exists to prevent (IST-3 oracle: 0.192 at 200 epochs vs 0.142 at 100).
        if self.cate_refit_full:
            n_full = data.X.shape[0]
            steps_per_epoch_sel = math.ceil(X_fit.shape[0] / self.cate_batch_size)
            steps_per_epoch_full = math.ceil(n_full / self.cate_batch_size)
            # best_epoch is 0-indexed, so the selected budget is best_epoch + 1 epochs.
            selected_steps = (self.history_["best_epoch"] + 1) * steps_per_epoch_sel
            self.refit_epochs_ = max(
                1, round(selected_steps / steps_per_epoch_full)
            )

            # Reuse the winning restart's seed: the refit is meant to reproduce that
            # fit on more data, not to draw a fresh initialization whose luck is
            # unrelated to the one the held-out block actually endorsed.
            if self.rng_seed is not None:
                keras.utils.set_random_seed(self.rng_seed + best_restart)

            model = self._build_cate_model()
            self.refit_history_ = train_cate(
                model,
                X=data.X,
                Y_tilde=Y_tilde,
                S_marg=S_marg,
                W=data.W,
                e_clipped=e_clipped,
                time_grid=self.time_grid_,
                delta=self.delta_,
                epochs=self.refit_epochs_,
                lr=self.cate_lr,
                batch_size=self.cate_batch_size,
                weight_decay=self.weight_decay,
                # Nothing is held out any more -- every row is in the fit. patience
                # None spends the whole selected budget rather than stopping on the
                # training loss, which is the quantity we deliberately do not trust.
                # restore_best stays on purely as the NaN guard: it picks the best
                # *training* epoch, which on a monotone curve is the last one anyway,
                # and it is what raises rather than returning NaN weights.
                validation_data=None,
                monitor="train",
                patience=None,
                rng_seed=self.rng_seed,
                verbose=verbose,
            )
            self.cate_model_ = model

            if verbose > 0:
                print(
                    f"  Refit on all {n_full} rows for {self.refit_epochs_} epochs "
                    f"(selected {self.history_['best_epoch'] + 1} on "
                    f"{X_fit.shape[0]} rows)"
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
