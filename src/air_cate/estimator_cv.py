"""Cross-validated screening of the number of mixture components K.

AIREstimatorCV is a *diagnostic* class: it scores a grid of candidate K values by
cross-validating the AIR R-loss and records the full per-epoch loss trajectory of
every training run. It does not refit a final model and has no predict methods --
once you have picked K from the plots, construct AIREstimator(K=...) and fit it on
the full training set.
"""

import numpy as np
from sklearn.model_selection import KFold

from air_cate._types import SurvivalData
from air_cate.cate_model.mixture import CATEMixture
from air_cate.cate_model.train import train_cate
from air_cate.nuisance.cross_fit import cross_fit_nuisances
from air_cate.pseudo_outcome import compute_pseudo_outcomes
from air_cate.utils import build_time_grid


class AIREstimatorCV:
    """Screen candidate values of K by cross-validating the AIR R-loss.

    For each outer CV fold, the data is split into a training block A and a
    held-out block B. Nuisances are estimated **independently within each block**:
    cross-fit within A for the training pseudo-outcomes, and re-cross-fit within B
    for the validation pseudo-outcomes. No model fitted on A ever touches B, so the
    CATE model (fit on A) and the validation targets (built from B) are independent,
    and both blocks' pseudo-outcomes are constructed by the same procedure.

    Within a fold the pseudo-outcomes are computed once and reused for every K. This
    reuse is load-bearing: it makes the comparison across K paired, so the *ranking*
    of K is unaffected by fold-level nuisance noise.

    The score for a given (K, fold) is the validation R-loss at the epoch minimizing
    validation loss; the CV score for K is the mean over folds.

    Cost: `n_cv_folds * (n_folds + val_n_folds) * 5` Keras nuisance models -- 250 at
    the defaults -- plus `n_cv_folds * len(K_grid)` CATE fits. The val-block models
    are individually cheap (they see ~n/n_cv_folds rows). For a first screen, use
    `n_folds=2, val_n_folds=2`.

    Sample-size rule of thumb: each val-block nuisance model trains on roughly
    `n / n_cv_folds * (1 - 1/val_n_folds)` rows, then splits per arm. You want
    `n / n_cv_folds * (1 - 1/val_n_folds) / 2` comfortably above 20.

    Args:
        K_grid: Candidate numbers of mixture components to screen.
        n_cv_folds: Number of OUTER folds (model selection over K).
        n_folds: Number of INNER nuisance cross-fitting folds for the training block.
        val_n_folds: Inner nuisance folds for the held-out block. None uses n_folds.
        cv_rng_seed: Seed for the outer split, nuisance splits, and weight init.
        M: Number of grid intervals (M+1 grid points).
        eps: Clipping bound for nuisance estimates.
        encoder_hidden_sizes: Hidden layer sizes for the feature encoder.
        propensity_config: kwargs for PropensityMLP constructor.
        weibull_config: kwargs for WeibullAFT constructor.
        nuisance_training_config: Training kwargs for nuisance models.
        cate_epochs: Training epochs for each CATE model.
        cate_lr: Learning rate for the CATE model.
        cate_batch_size: Batch size for the CATE model.
        weight_decay: Weight decay for AdamW optimizer.
        cv_patience: Early-stopping patience on validation loss. None (the default)
            trains the full `cate_epochs` so the convergence grid shows complete,
            equal-length trajectories. Scoring at the best-val epoch needs only the
            minimum, not early stopping.
        cate_model_class: Optional alternative CATE model class.
        cate_model_kwargs: Extra kwargs for `cate_model_class`.
        k_param_name: Name of the constructor argument that sets K on the model
            class, so a custom `cate_model_class` can also be screened.

    Attributes (after fit):
        cv_results_: dict of K_grid, fold_val_loss, mean_val_loss, se_val_loss,
            fold_best_epoch.
        histories_: dict mapping (K, fold) -> the history dict from `train_cate`.
        best_K_: K minimizing mean_val_loss. A suggestion only -- read the plots.
        time_grid_, delta_: The time grid and its step size.
    """

    def __init__(
        self,
        K_grid=(1, 2, 3, 4, 6, 8),
        n_cv_folds: int = 5,
        n_folds: int = 5,
        val_n_folds: int | None = None,
        cv_rng_seed: int = 42,
        M: int = 100,
        eps: float = 0.01,
        encoder_hidden_sizes: tuple[int, ...] = (64, 32),
        propensity_config: dict | None = None,
        weibull_config: dict | None = None,
        nuisance_training_config: dict | None = None,
        cate_epochs: int = 200,
        cate_lr: float = 1e-3,
        cate_batch_size: int = 256,
        weight_decay: float = 1e-4,
        cv_patience: int | None = None,
        cate_model_class=None,
        cate_model_kwargs: dict | None = None,
        k_param_name: str = "K",
    ):
        self.K_grid = tuple(int(k) for k in K_grid)
        self.n_cv_folds = n_cv_folds
        self.n_folds = n_folds
        self.val_n_folds = n_folds if val_n_folds is None else val_n_folds
        self.cv_rng_seed = cv_rng_seed
        self.M = M
        self.eps = eps
        self.encoder_hidden_sizes = encoder_hidden_sizes
        self.propensity_config = propensity_config
        self.weibull_config = weibull_config
        self.nuisance_training_config = nuisance_training_config
        self.cate_epochs = cate_epochs
        self.cate_lr = cate_lr
        self.cate_batch_size = cate_batch_size
        self.weight_decay = weight_decay
        self.cv_patience = cv_patience
        self.cate_model_class = cate_model_class
        self.cate_model_kwargs = cate_model_kwargs or {}
        self.k_param_name = k_param_name

        self.cv_results_ = None
        self.histories_ = None
        self.best_K_ = None
        self.time_grid_ = None
        self.delta_ = None

    def _build_model(self, K: int):
        """Instantiate a fresh CATE model with K components."""
        if self.cate_model_class is not None:
            kwargs = dict(self.cate_model_kwargs)
            kwargs[self.k_param_name] = K
            return self.cate_model_class(**kwargs)
        return CATEMixture(K=K, encoder_hidden_sizes=self.encoder_hidden_sizes)

    def _cross_fit_block(self, data: SurvivalData, idx, n_folds: int, fold_seed: int):
        """Cross-fit nuisances and compute pseudo-outcomes within one row block.

        `cross_fit_nuisances` uses its `rng_seed` for the KFold split only -- the
        five Keras nuisance models it fits per fold draw their weight init from
        Keras's *global* RNG. Seeding that here is what makes the whole screen
        reproducible: without it, the first fold's nuisances inherit whatever
        global RNG state the process happened to be in, so two identically seeded
        fits give different pseudo-outcomes (and hence different CV scores) for
        fold 0, while later folds are accidentally pinned by the `set_random_seed`
        call in the K loop.
        """
        import keras

        keras.utils.set_random_seed(fold_seed)

        block = SurvivalData(
            X=data.X[idx], W=data.W[idx], Z=data.Z[idx], Delta=data.Delta[idx]
        )
        nuisances = cross_fit_nuisances(
            block,
            self.time_grid_,
            n_folds=n_folds,
            propensity_config=self.propensity_config,
            weibull_config=self.weibull_config,
            training_config=self.nuisance_training_config,
            rng_seed=fold_seed,
        )
        Y_tilde, S_marg, e_clipped = compute_pseudo_outcomes(
            block, nuisances, self.time_grid_, self.delta_, eps=self.eps
        )
        return block, Y_tilde, S_marg, e_clipped

    def fit(
        self,
        X: np.ndarray,
        W: np.ndarray,
        Z: np.ndarray,
        Delta: np.ndarray,
        t_end: float,
        verbose: int = 0,
    ) -> "AIREstimatorCV":
        """Run the K screen.

        Args:
            X: Covariates (n, p).
            W: Binary treatment (n,).
            Z: Observed times min(D, C) (n,).
            Delta: Event indicator 1{D <= C} (n,).
            t_end: Maximum follow-up time.
            verbose: Verbosity level (0=silent, >0 prints per-fold progress).

        Returns:
            self
        """
        import jax.numpy as jnp
        import keras

        self.time_grid_, self.delta_ = build_time_grid(t_end, self.M)

        data = SurvivalData(
            X=jnp.array(X, dtype=jnp.float32),
            W=jnp.array(W, dtype=jnp.float32),
            Z=jnp.array(Z, dtype=jnp.float32),
            Delta=jnp.array(Delta, dtype=jnp.float32),
        )

        n = data.X.shape[0]
        if self.n_cv_folds < 2:
            raise ValueError(f"n_cv_folds must be >= 2, got {self.n_cv_folds}")

        # The held-out block is the sample-size-limited one: it holds ~n/n_cv_folds
        # rows and is cross-fit again inside itself.
        val_block_size = n // self.n_cv_folds
        if val_block_size < self.val_n_folds:
            raise ValueError(
                f"Held-out block of ~{val_block_size} rows (n={n}, "
                f"n_cv_folds={self.n_cv_folds}) cannot be cross-fit into "
                f"val_n_folds={self.val_n_folds} folds. Lower n_cv_folds or "
                f"val_n_folds, or use more data."
            )

        n_K = len(self.K_grid)
        fold_val_loss = np.full((n_K, self.n_cv_folds), np.nan)
        fold_best_epoch = np.full((n_K, self.n_cv_folds), -1, dtype=int)
        histories: dict[tuple[int, int], dict] = {}

        kf = KFold(
            n_splits=self.n_cv_folds, shuffle=True, random_state=self.cv_rng_seed
        )

        for j, (train_idx, val_idx) in enumerate(kf.split(np.asarray(data.X))):
            if verbose > 0:
                print(
                    f"[fold {j + 1}/{self.n_cv_folds}] "
                    f"n_train={len(train_idx)} n_val={len(val_idx)} — nuisances A",
                    end=" ",
                    flush=True,
                )

            # Each (fold, role) gets its own seed out of a 1000-wide block, so the
            # two nuisance runs and every CATE run are independently seeded but
            # fully reproducible. Sharing one seed between A and B would hand both
            # blocks identical nuisance weight init.
            seed_A = self.cv_rng_seed + 1000 * j + 1
            seed_B = self.cv_rng_seed + 1000 * j + 2

            # Nuisances for the training block, cross-fit using only its own rows.
            block_A, Y_A, S_A, e_A = self._cross_fit_block(
                data, train_idx, self.n_folds, seed_A
            )

            if verbose > 0:
                print("B", end=" ", flush=True)

            # Nuisances for the held-out block, RE-CROSS-FIT USING ONLY ITS OWN ROWS.
            # No model fitted on the training block ever touches these rows.
            block_B, Y_B, S_B, e_B = self._cross_fit_block(
                data, val_idx, self.val_n_folds, seed_B
            )

            # Computed once per fold and reused for every K below: this is what
            # makes the comparison across K paired.
            validation_data = {
                "X": block_B.X,
                "Y_tilde": Y_B,
                "S_marg": S_B,
                "W": block_B.W,
                "e_clipped": e_B,
            }

            if verbose > 0:
                print("| ", end="", flush=True)

            for i, K in enumerate(self.K_grid):
                run_seed = self.cv_rng_seed + 1000 * j + 100 + K
                keras.utils.set_random_seed(run_seed)
                model = self._build_model(K)

                history = train_cate(
                    model,
                    X=block_A.X,
                    Y_tilde=Y_A,
                    S_marg=S_A,
                    W=block_A.W,
                    e_clipped=e_A,
                    time_grid=self.time_grid_,
                    delta=self.delta_,
                    epochs=self.cate_epochs,
                    lr=self.cate_lr,
                    batch_size=self.cate_batch_size,
                    weight_decay=self.weight_decay,
                    validation_data=validation_data,
                    monitor="val",
                    patience=self.cv_patience,
                    rng_seed=run_seed,
                    verbose=0,
                )

                histories[(K, j)] = history
                fold_val_loss[i, j] = history["best_val_loss"]
                fold_best_epoch[i, j] = history["best_epoch"]

                if verbose > 0:
                    print(
                        f"K={K}: {history['best_val_loss']:.5f} ",
                        end="",
                        flush=True,
                    )

            if verbose > 0:
                print()

        mean_val_loss = fold_val_loss.mean(axis=1)
        # SE of the mean across folds (ddof=1); NaN if only one fold.
        se_val_loss = fold_val_loss.std(axis=1, ddof=1) / np.sqrt(self.n_cv_folds)

        self.cv_results_ = {
            "K_grid": list(self.K_grid),
            "fold_val_loss": fold_val_loss,
            "mean_val_loss": mean_val_loss,
            "se_val_loss": se_val_loss,
            "fold_best_epoch": fold_best_epoch,
        }
        self.histories_ = histories
        self.best_K_ = self.K_grid[int(np.argmin(mean_val_loss))]

        return self

    # ------------------------------------------------------------------
    # Serialization (JSON-safe), so plots can be regenerated without refitting
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serializable snapshot of the screen results."""
        if self.cv_results_ is None:
            raise ValueError("Nothing to serialize: call fit() first.")

        # Tuple keys are not valid JSON keys, so histories become a record list.
        history_records = []
        for (K, fold), h in self.histories_.items():
            record = {"K": int(K), "fold": int(fold)}
            record.update(h)
            history_records.append(record)

        return {
            "K_grid": list(self.K_grid),
            "n_cv_folds": self.n_cv_folds,
            "n_folds": self.n_folds,
            "val_n_folds": self.val_n_folds,
            "cate_epochs": self.cate_epochs,
            "cv_results": {
                "K_grid": list(self.K_grid),
                "fold_val_loss": self.cv_results_["fold_val_loss"].tolist(),
                "mean_val_loss": self.cv_results_["mean_val_loss"].tolist(),
                "se_val_loss": self.cv_results_["se_val_loss"].tolist(),
                "fold_best_epoch": self.cv_results_["fold_best_epoch"].tolist(),
            },
            "histories": history_records,
            "best_K": int(self.best_K_),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AIREstimatorCV":
        """Rebuild a (results-only) AIREstimatorCV from `to_dict` output."""
        obj = cls(
            K_grid=d["K_grid"],
            n_cv_folds=d["n_cv_folds"],
            n_folds=d["n_folds"],
            val_n_folds=d["val_n_folds"],
            cate_epochs=d["cate_epochs"],
        )
        cv = d["cv_results"]
        obj.cv_results_ = {
            "K_grid": list(cv["K_grid"]),
            "fold_val_loss": np.asarray(cv["fold_val_loss"], dtype=float),
            "mean_val_loss": np.asarray(cv["mean_val_loss"], dtype=float),
            "se_val_loss": np.asarray(cv["se_val_loss"], dtype=float),
            "fold_best_epoch": np.asarray(cv["fold_best_epoch"], dtype=int),
        }
        obj.histories_ = {}
        for record in d["histories"]:
            record = dict(record)
            K = record.pop("K")
            fold = record.pop("fold")
            obj.histories_[(K, fold)] = record
        obj.best_K_ = d["best_K"]
        return obj

    # ------------------------------------------------------------------
    # Plot delegation
    # ------------------------------------------------------------------

    def plot_cv_curve(self, **kwargs):
        """Plot CV AIR loss against K on the absolute scale. See air_cate.plotting.

        Usually not the panel to pick K from -- fold-level nuisance offsets dominate
        this scale. Use `plot_k_selection` (both panels) or `plot_cv_paired`.
        """
        from air_cate.plotting import plot_cv_curve

        return plot_cv_curve(self.cv_results_, **kwargs)

    def plot_cv_paired(self, **kwargs):
        """Plot the paired change in CV AIR loss against K. Decide K from this."""
        from air_cate.plotting import plot_cv_paired

        return plot_cv_paired(self.cv_results_, **kwargs)

    def plot_k_selection(self, **kwargs):
        """Two-panel K-selection figure (absolute + paired). See air_cate.plotting."""
        from air_cate.plotting import plot_k_selection

        return plot_k_selection(self.cv_results_, **kwargs)

    def plot_convergence_grid(self, **kwargs):
        """Plot the grid of per-run convergence curves. See air_cate.plotting."""
        from air_cate.plotting import plot_convergence_grid

        kwargs.setdefault("K_grid", list(self.K_grid))
        kwargs.setdefault("n_cv_folds", self.n_cv_folds)
        return plot_convergence_grid(self.histories_, **kwargs)

    def summary(self) -> str:
        """Return a console table of K vs cross-validated AIR loss.

        Reports both scales. The absolute loss and its marginal SE describe the
        level; the paired difference against the smallest K in the grid and its
        paired SE are what the K comparison actually rests on, since differencing
        within folds removes the fold-level nuisance offset. Expect the paired SE
        to be far smaller than the marginal one.
        """
        if self.cv_results_ is None:
            raise ValueError("Call fit() first.")

        from air_cate.plotting import _paired_stats

        cv = self.cv_results_
        K_list = cv["K_grid"]
        _, mean_diff, se_diff = _paired_stats(
            np.asarray(cv["fold_val_loss"], dtype=float), 0
        )
        ref_K = K_list[0]

        lines = [
            f"{'K':>4}  {'CV AIR loss':>12}  {'SE':>8}  "
            f"{'vs K=' + str(ref_K):>10}  {'paired SE':>10}  {'best epoch':>10}",
            "-" * 64,
        ]
        for i, K in enumerate(K_list):
            marker = "  <- min" if K == self.best_K_ else ""
            lines.append(
                f"{K:>4}  {cv['mean_val_loss'][i]:>12.6f}  "
                f"{cv['se_val_loss'][i]:>8.4f}  "
                f"{mean_diff[i]:>+10.5f}  {se_diff[i]:>10.5f}  "
                f"{cv['fold_best_epoch'][i].mean():>10.1f}{marker}"
            )

        n_at_zero = int((np.asarray(cv["fold_best_epoch"]) == 0).sum())
        if n_at_zero:
            lines.append(
                f"\nNote: {n_at_zero} run(s) scored at epoch 0 -- validation loss "
                "never improved on the untrained model, so those scores reflect "
                "initialization rather than a fitted model."
            )
        return "\n".join(lines)
