"""Plot predicted time-varying tv-CATE curves for a single test observation.

Fits 4 non-oracle methods on one training draw (Weibull DGP) and overlays
their predicted tau(t|x) against the true tv-CATE for the test subject
where Cross-fitted AIR w/ MGD tracks the truth most closely.

Usage:
  uv run python examples/plot_cate_curves.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from air_cate import AIREstimator
from air_cate.baselines import SurvivalForestTLearner, SurvivalForestSLearner
from air_cate.cate_model.tensor_product import CATETensorProduct
from conftest import WeibullDGP

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Style (consistent with plot_simulation.py)
# ---------------------------------------------------------------------------

_JAMA = {
    "teal":   "#374e55",
    "orange": "#df8f44",
    "blue":   "#00a1d5",
    "red":    "#b24745",
    "green":  "#79af97",
}

METHODS = [
    ("Cross-fitted AIR w/ MGD",    _JAMA["teal"],   "-",  2.2),
    ("Cross-fitted AIR w/ TP",     _JAMA["orange"], "-",  2.2),
    ("Survival Forest T-learner",  _JAMA["red"],    "-",  2.2),
    ("Survival Forest S-learner",  _JAMA["green"],  "-",  2.2),
]

# ---------------------------------------------------------------------------
# Fit all methods, return predicted curves for ALL test subjects
# ---------------------------------------------------------------------------

N_TRAIN = 2000
SEED = 0
M = 50
N_TEST = 200

COMMON_AIR = dict(
    M=M, n_folds=2, eps=0.01,
    nuisance_training_config={"epochs": 50, "verbose": 0},
    cate_epochs=200, cate_lr=1e-3,
)


def fit_all_methods():
    """Fit 4 methods on Weibull DGP, return predictions for all test subjects."""
    np.random.seed(SEED)
    X_train, W, Z, Delta, t_end = WeibullDGP.generate(n=N_TRAIN, rng_seed=SEED)

    rng = np.random.default_rng(9999)
    X_test = rng.uniform(-1, 1, (N_TEST, 5)).astype(np.float32)

    estimators = [
        AIREstimator(**COMMON_AIR, K=3, encoder_hidden_sizes=(32, 16)),
        AIREstimator(**COMMON_AIR, cate_model_class=CATETensorProduct,
                     cate_model_kwargs=dict(n_time_basis=8, poly_degree=2)),
        SurvivalForestTLearner(M=M, n_estimators=200, min_samples_leaf=10),
        SurvivalForestSLearner(M=M, n_estimators=200, min_samples_leaf=10),
    ]

    tau_hats_all = []  # list of (N_TEST, M+1) arrays
    for est in estimators:
        print(f"  Fitting {est.__class__.__name__} ...", end=" ", flush=True)
        est.fit(X_train, W, Z, Delta, t_end, verbose=0)
        tau_hat = est.predict_tau(X_test)  # (N_TEST, M+1)
        tau_hats_all.append(tau_hat)
        print("done")

    t_grid = np.array(estimators[0].time_grid_)
    delta = estimators[0].delta_
    tau_true = WeibullDGP.true_tau(t_grid, X_test)  # (N_TEST, M+1)

    return t_grid, delta, tau_true, tau_hats_all


def find_best_observation(delta, tau_true, tau_hats_all):
    """Find the test subject where:
      1. The true tv-CATE has a large dynamic range (interesting shape).
      2. AIR w/ MGD has the lowest iMSE among all methods.
    """
    n_methods = len(tau_hats_all)

    # iMSE per subject per method: (n_methods, N_TEST)
    imse = np.zeros((n_methods, N_TEST))
    for j, tau_hat in enumerate(tau_hats_all):
        imse[j] = np.sum(delta * (tau_hat - tau_true) ** 2, axis=1)

    # Dynamic range of true tv-CATE per subject
    tau_range = tau_true.max(axis=1) - tau_true.min(axis=1)

    # Filter: AIR w/ MGD must be the best method (lowest iMSE)
    air_best = imse[0] == imse.min(axis=0)

    # Among those, pick the one with the largest true tv-CATE range
    score = np.where(air_best, tau_range, -np.inf)
    best_idx = int(np.argmax(score))

    print(f"\nBest observation: idx={best_idx}  "
          f"(tv-CATE range: {tau_range[best_idx]:.4f})")
    for j, (label, _, _, _) in enumerate(METHODS):
        print(f"  {label}: iMSE = {imse[j, best_idx]:.6f}")

    return best_idx


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_curves(t_grid, tau_true_i, tau_hats_i, obs_idx):
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    # True tv-CATE
    ax.plot(t_grid, tau_true_i, color="#888888", ls="--", lw=1.4,
            label="True tv-CATE", zorder=6)

    # Method curves
    for (label, color, ls, lw), tau_hat_i in zip(METHODS, tau_hats_i):
        ax.plot(t_grid, tau_hat_i, color=color, ls=ls, lw=lw,
                label=label, zorder=4)

    ax.axhline(0, color="black", lw=0.4, zorder=0)
    ax.set_xlabel("Time $t$", fontsize=10.5)
    ax.set_ylabel(r"$\hat{\tau}(t \mid x)$", fontsize=11)
    ax.set_title(f"Weibull DGP — test observation {obs_idx}",
                 fontsize=11.5, fontweight="bold", pad=8)
    ax.tick_params(labelsize=8.5)
    ax.grid(True, alpha=0.12, linewidth=0.4)

    # Legend below
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06),
               columnspacing=1.5, handletextpad=0.6)

    fig.tight_layout(rect=[0, 0.10, 1, 1])
    for ext in ("png", "pdf"):
        path = os.path.join(FIGURES_DIR, f"cate_curves.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def main():
    print("--- Weibull DGP ---")
    t_grid, delta, tau_true, tau_hats_all = fit_all_methods()

    best_idx = find_best_observation(delta, tau_true, tau_hats_all)

    tau_true_i = tau_true[best_idx]
    tau_hats_i = [th[best_idx] for th in tau_hats_all]

    plot_curves(t_grid, tau_true_i, tau_hats_i, best_idx)


if __name__ == "__main__":
    main()
