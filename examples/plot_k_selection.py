"""Screen the number of clusters K by cross-validating the AIR CATE loss.

Runs AIREstimatorCV on the Weibull mixture DGP -- which is genuinely K=3 in both
arms -- and produces two figures:

  k_selection_cv           CV AIR CATE loss against K, on two scales: absolute
                           (shows the level and the fold offsets) and paired
                           against the reference K (decide K from this one).
  k_selection_convergence  Grid of convergence curves: columns = K, rows = CV run.
                           Each panel plots training vs held-out AIR loss by epoch,
                           each as the change from its own epoch-1 value.

Usage:
  uv run --extra plots python examples/plot_k_selection.py
  uv run --extra plots python examples/plot_k_selection.py --large
  uv run --extra plots python examples/plot_k_selection.py --plot-only
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from air_cate import AIREstimatorCV
from air_cate.plotting import plot_convergence_grid, plot_k_selection
from conftest import WeibullMixtureDGP

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

RESULTS_PATH = os.path.join(FIGURES_DIR, "k_selection_results.json")

# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

SEED = 0

# Quick screen: small inner-fold counts keep the nuisance stage affordable.
#
# On n: the held-out block holds n/n_cv_folds rows and is cross-fit again inside
# itself, so each of its nuisance models sees only n/n_cv_folds * (1 - 1/val_n_folds)
# rows, halved again per arm -- ~93 subjects per arm here. That sounds too thin, and
# it does put the held-out loss at roughly 3x the training level, but the level gap
# is handled by the plots (both figures work on within-curve and within-fold
# differences, never on the absolute level), and raising n does not help the part
# that matters: at n=3000 the CATE models overfit the training pseudo-outcomes from
# about epoch 15, so the best-validation epoch lands almost at initialization and
# every K scores within noise of every other. At n=1500 the best epochs land at
# 40-107 and every K >= 2 sits clearly below K=1 on the paired scale (differences of
# -0.022 to -0.026 against paired SEs of 0.007-0.010). Smaller is the better demo.
#
# The tradeoff it buys: with 1125 training rows and batch_size=256 each epoch's
# training loss averages only ~4 minibatches, so the training curve in the
# convergence grid is visibly jittery (the held-out curve, evaluated in one pass
# over the whole block, stays smooth). Raise n to smooth it, at the cost of the K
# signal above.
QUICK_CONFIG = dict(
    n=1500,
    K_grid=(1, 2, 3, 4, 6),
    n_cv_folds=4,
    n_folds=2,
    val_n_folds=2,
    M=50,
    encoder_hidden_sizes=(32, 16),
    nuisance_training_config={"epochs": 50, "verbose": 0},
    cate_epochs=150,
    cate_lr=1e-3,
)

LARGE_CONFIG = dict(
    n=4000,
    K_grid=(1, 2, 3, 4, 5, 6, 8),
    n_cv_folds=5,
    n_folds=5,
    val_n_folds=2,
    M=100,
    encoder_hidden_sizes=(64, 32),
    nuisance_training_config={"epochs": 100, "verbose": 0},
    cate_epochs=200,
    cate_lr=1e-3,
)


# ---------------------------------------------------------------------------
# Run the screen
# ---------------------------------------------------------------------------

def run_screen(config: dict) -> AIREstimatorCV:
    """Fit AIREstimatorCV on one draw from the Weibull mixture DGP."""
    cfg = dict(config)
    n = cfg.pop("n")

    print(f"Generating WeibullMixtureDGP draw (n={n}, seed={SEED}) ...")
    X, W, Z, Delta, t_end = WeibullMixtureDGP.generate(n=n, rng_seed=SEED)

    cv = AIREstimatorCV(cv_rng_seed=42, **cfg)

    n_nuisance = cv.n_cv_folds * (cv.n_folds + cv.val_n_folds) * 5
    n_cate = cv.n_cv_folds * len(cv.K_grid)
    print(
        f"Screening K in {cv.K_grid} with {cv.n_cv_folds} CV folds\n"
        f"  ~{n_nuisance} nuisance models + {n_cate} CATE fits\n"
    )

    cv.fit(X, W, Z, Delta, t_end, verbose=1)
    return cv


# ---------------------------------------------------------------------------
# Persistence, so figures can be regenerated without refitting
# ---------------------------------------------------------------------------

def save_results(cv: AIREstimatorCV) -> None:
    with open(RESULTS_PATH, "w") as f:
        json.dump(cv.to_dict(), f, indent=2)
    print(f"Saved: {RESULTS_PATH}")


def load_results() -> AIREstimatorCV:
    with open(RESULTS_PATH) as f:
        return AIREstimatorCV.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _save(fig, stem: str) -> None:
    for ext in ("png", "pdf"):
        path = os.path.join(FIGURES_DIR, f"{stem}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def make_plots(cv: AIREstimatorCV) -> None:
    # Two panels: the absolute scale shows the loss level and the fold offsets that
    # dominate it; the paired scale is what K is actually chosen from.
    fig, _ = plot_k_selection(cv.cv_results_)
    _save(fig, "k_selection_cv")

    fig, _ = plot_convergence_grid(
        cv.histories_, K_grid=list(cv.K_grid), n_cv_folds=cv.n_cv_folds
    )
    _save(fig, "k_selection_convergence")


def _verdict(cv: AIREstimatorCV) -> str:
    """Report the argmin K alongside whether the screen can actually resolve it.

    `best_K_` is a bare argmin over the mean CV loss, so it always names some K
    even when the K values are indistinguishable. The paired difference against
    the reference K and its paired SE say whether that name means anything, so
    print both rather than the argmin alone.
    """
    import numpy as np

    from air_cate.plotting import _paired_stats

    K_list = list(cv.cv_results_["K_grid"])
    fold_loss = np.asarray(cv.cv_results_["fold_val_loss"], dtype=float)
    _, mean_diff, se_diff = _paired_stats(fold_loss, 0)

    best_i = K_list.index(cv.best_K_)
    diff, se = mean_diff[best_i], se_diff[best_i]
    ref_K = K_list[0]

    lines = [f"\nMinimum of the CV curve: K = {cv.best_K_} "
             f"(true K = 3 for WeibullMixtureDGP)."]

    if best_i == 0:
        lines.append(
            f"The screen ranks the no-heterogeneity model K={ref_K} best, so it "
            "found no evidence of covariate heterogeneity at this sample size."
        )
    elif np.isfinite(se) and diff + se < 0:
        lines.append(
            f"It beats K={ref_K} by {-diff:.5f} against a paired SE of {se:.5f}, "
            "so the difference clears one paired SE -- the screen does separate "
            "these K values."
        )
    else:
        lines.append(
            f"But it beats K={ref_K} by only {-diff:.5f} against a paired SE of "
            f"{se:.5f}, so it does NOT clear one paired SE. Treat this K as "
            "unresolved by the screen rather than as a selection, and prefer the "
            "smallest K in the shaded band of the paired panel."
        )

    lines.append(
        "\nDecide from the right-hand (paired) panel of k_selection_cv, check "
        "k_selection_convergence for held-out curves turning upward, then fit the "
        "final model yourself:\n"
        "  AIREstimator(K=<your choice>, ...).fit(X, W, Z, Delta, t_end)"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--large", action="store_true",
        help="Use the larger configuration (slower, more K values).",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Re-plot from cached JSON results without refitting.",
    )
    args = parser.parse_args()

    if args.plot_only:
        if not os.path.exists(RESULTS_PATH):
            raise SystemExit(
                f"No cached results at {RESULTS_PATH}. Run without --plot-only first."
            )
        print(f"Loading cached results from {RESULTS_PATH}")
        cv = load_results()
    else:
        config = LARGE_CONFIG if args.large else QUICK_CONFIG
        cv = run_screen(config)
        save_results(cv)

    print("\n" + cv.summary())
    print(_verdict(cv))

    make_plots(cv)


if __name__ == "__main__":
    main()
