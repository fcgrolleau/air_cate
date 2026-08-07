"""Quickstart example: choose K by cross-validating the AIR CATE loss.

AIREstimator takes the number of mixture components K as a fixed argument.
AIREstimatorCV screens a grid of candidate K values by cross-validating the AIR
R-loss, so you can pick one from a plot instead of guessing.

It is a *diagnostic only*: no predict methods, no final refit. The workflow is

    1. cv = AIREstimatorCV(K_grid=...).fit(...)   # score every K
    2. read K off cv.summary() and the two figures
    3. AIREstimator(K=<your choice>).fit(...)     # refit on the full data

Data here is WeibullMixtureDGP, a 3-component Weibull mixture in each arm, so the
right answer is K = 3.

Usage:
  uv run python examples/quickstart.py                     # the single-fit version
  uv run --extra plots python examples/quickstart_cv.py    # this one, with figures

Runtime is a few minutes: the screen fits n_cv_folds * (n_folds + val_n_folds) * 5
nuisance models plus n_cv_folds * len(K_grid) CATE models. See
examples/plot_k_selection.py for a larger configuration and a --plot-only cache.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from air_cate import AIREstimator, AIREstimatorCV
from conftest import WeibullMixtureDGP

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")


def screen_K(X, W, Z, Delta, t_end):
    """Cross-validate the AIR loss over a grid of K."""
    cv = AIREstimatorCV(
        K_grid=(1, 2, 3, 4),
        n_cv_folds=3,      # OUTER folds: the K comparison
        n_folds=2,         # INNER nuisance folds, training block
        val_n_folds=2,     # INNER nuisance folds, held-out block
        cv_rng_seed=42,
        M=50,
        encoder_hidden_sizes=(32, 16),
        nuisance_training_config={"epochs": 40, "verbose": 0},
        cate_epochs=100,
        cate_lr=1e-3,
        # Each epoch's *training* loss is a mean over minibatches, while the
        # held-out loss is one clean full-block pass. The R-loss weights subjects
        # by 1/e_clipped, so it is heavy-tailed: with the default batch_size=256
        # an 800-row training block gives only ~3 minibatches per epoch, and a
        # single extreme subject swings that mean far more than the held-out curve
        # ever moves -- which makes the convergence grid unreadable. Smaller
        # batches average more of that noise away.
        cate_batch_size=64,
    )
    cv.fit(X, W, Z, Delta, t_end, verbose=1)
    return cv


def read_off_K(cv):
    """Print the screen result on the scale the K comparison actually rests on.

    `best_K_` is a bare argmin over the mean CV loss, and that mean is dominated by
    fold-level noise: each held-out block gets its own nuisances, so each fold sits
    at its own loss level. Those offsets are common to every K within a fold, so
    differencing against a reference K inside each fold cancels them -- which is why
    the paired SE below is much smaller than the marginal one.
    """
    print("\n" + cv.summary())

    fold_loss = np.asarray(cv.cv_results_["fold_val_loss"], dtype=float)
    K_list = list(cv.cv_results_["K_grid"])

    # Difference against the first K in the grid, fold by fold.
    diff = fold_loss - fold_loss[0][None, :]
    mean_diff = diff.mean(axis=1)
    se_diff = diff.std(axis=1, ddof=1) / np.sqrt(diff.shape[1])

    # One-SE rule on the paired scale: smallest K whose mean sits inside one
    # paired SE of the best mean.
    best = int(np.argmin(mean_diff))
    threshold = mean_diff[best] + se_diff[best]
    selected = K_list[int(np.argmax(mean_diff <= threshold))]

    print(f"\nArgmin of the CV curve:        K = {cv.best_K_}")
    print(f"One-SE rule on paired scale:   K = {selected}")
    print("True K for WeibullMixtureDGP:  K = 3")
    return selected


def make_figures(cv):
    """Save the two diagnostic figures, if matplotlib is installed."""
    try:
        import matplotlib
    except ImportError:
        print("\nmatplotlib not installed; skipping figures.")
        print("Install with: uv sync --extra plots")
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGURES_DIR, exist_ok=True)

    def save(fig, stem):
        path = os.path.join(FIGURES_DIR, f"{stem}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # Left panel: the absolute loss level. Right panel: the paired differences,
    # which is the panel to decide K from.
    fig, _ = cv.plot_k_selection()
    save(fig, "quickstart_cv_k")

    # Columns = K, rows = CV run. Each panel: training vs held-out loss by epoch.
    # A held-out curve turning upward means that K overfits.
    fig, _ = cv.plot_convergence_grid()
    save(fig, "quickstart_cv_convergence")


def main():
    print("Generating synthetic data (WeibullMixtureDGP, true K = 3)...")
    X, W, Z, Delta, t_end = WeibullMixtureDGP.generate(n=1200, rng_seed=0)

    print("Screening K with AIREstimatorCV...")
    cv = screen_K(X, W, Z, Delta, t_end)

    K = read_off_K(cv)
    make_figures(cv)

    # Step 3: refit on the full data at the chosen K. The screen hands over nothing
    # but the integer -- the nuisance stage is re-run here on all n rows.
    print(f"\nRefitting AIREstimator(K={K}) on the full sample...")
    est = AIREstimator(
        K=K,
        M=50,
        n_folds=3,
        encoder_hidden_sizes=(32, 16),
        nuisance_training_config={"epochs": 40, "verbose": 0},
        cate_epochs=150,
        cate_lr=1e-3,
        cate_batch_size=64,  # match the screen, so K carries over meaningfully
    )
    est.fit(X, W, Z, Delta, t_end, verbose=1)

    t_grid = np.array(est.time_grid_)
    tau_hat = est.predict_tau(X)
    tau_true = WeibullMixtureDGP.true_tau(t_grid, X)
    imse = np.mean(np.sum(est.delta_ * (tau_hat - tau_true) ** 2, axis=1))
    print(f"\nIntegrated MSE of the refit model: {imse:.4f}")


if __name__ == "__main__":
    main()
