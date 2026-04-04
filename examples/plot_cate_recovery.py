"""Visualize CATE recovery: true vs estimated time-varying treatment effects."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Ensure air_cate is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from air_cate import AIREstimator
from conftest import WeibullDGP

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def fit_estimator():
    """Fit AIREstimator on Weibull DGP and return predictions on test data."""
    print("Generating training data (n=2000)...")
    X_train, W, Z, Delta, t_end = WeibullDGP.generate(n=2000, rng_seed=42)

    print("Fitting AIREstimator (K=4, M=50, 3-fold cross-fitting)...")
    est = AIREstimator(
        K=4,
        M=50,
        n_folds=3,
        eps=0.01,
        encoder_hidden_sizes=(32, 16),
        nuisance_training_config={"epochs": 60, "verbose": 0},
        cate_epochs=200,
        cate_lr=1e-3,
    )
    est.fit(X_train, W, Z, Delta, t_end, verbose=1)

    # Test data
    rng = np.random.default_rng(99)
    X_test = rng.uniform(-1, 1, (500, X_train.shape[1])).astype(np.float32)

    t_grid = np.array(est.time_grid_)
    tau_hat = est.predict_tau(X_test)
    tau_true = WeibullDGP.true_tau(t_grid, X_test)
    delta = est.delta_

    imse = np.mean(np.sum(delta * (tau_hat - tau_true) ** 2, axis=1))
    print(f"Integrated MSE: {imse:.4f}\n")

    return t_grid, X_test, tau_hat, tau_true, delta


def plot_individual_curves(t_grid, X_test, tau_hat, tau_true):
    """Figure 1: Individual CATE curves — true vs estimated for 4 subjects."""
    # Select subjects with x_1 near -0.8, -0.3, +0.3, +0.8
    target_x1 = [-0.8, -0.3, 0.3, 0.8]
    x1_vals = X_test[:, 1]
    selected = [int(np.argmin(np.abs(x1_vals - t))) for t in target_x1]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    fig.suptitle("Individual CATE Curves: True vs Estimated", fontsize=14, fontweight="bold")

    for ax, idx, target in zip(axes.flat, selected, target_x1):
        ax.plot(t_grid, tau_true[idx], "k-", linewidth=2, label="True")
        ax.plot(t_grid, tau_hat[idx], "C0--", linewidth=2, label="Estimated")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(f"$x_1 = {X_test[idx, 1]:.2f}$", fontsize=12)
        ax.legend(fontsize=9)
        ax.set_ylim(-0.5, 0.5)

    for ax in axes[1, :]:
        ax.set_xlabel("Time $t$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\tau(t \mid x)$")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(FIGURES_DIR, "fig1_individual_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_strata_averages(t_grid, X_test, tau_hat, tau_true):
    """Figure 2: Average CATE across x_1 quartile strata."""
    x1 = X_test[:, 1]
    quartiles = np.percentile(x1, [25, 50, 75])
    bins = np.digitize(x1, quartiles)  # 0, 1, 2, 3

    labels = [
        f"$x_1$ < {quartiles[0]:.2f}",
        f"{quartiles[0]:.2f} < $x_1$ < {quartiles[1]:.2f}",
        f"{quartiles[1]:.2f} < $x_1$ < {quartiles[2]:.2f}",
        f"$x_1$ > {quartiles[2]:.2f}",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    fig.suptitle("Average CATE by $x_1$ Quartile: True vs Estimated", fontsize=14, fontweight="bold")

    for ax, b, label in zip(axes.flat, range(4), labels):
        mask = bins == b
        mean_true = tau_true[mask].mean(axis=0)
        std_true = tau_true[mask].std(axis=0)
        mean_hat = tau_hat[mask].mean(axis=0)
        std_hat = tau_hat[mask].std(axis=0)

        ax.plot(t_grid, mean_true, "k-", linewidth=2, label="True mean")
        ax.fill_between(t_grid, mean_true - std_true, mean_true + std_true, color="gray", alpha=0.2, label="True $\\pm$ 1 SD")
        ax.plot(t_grid, mean_hat, "C0--", linewidth=2, label="Est. mean")
        ax.fill_between(t_grid, mean_hat - std_hat, mean_hat + std_hat, color="C0", alpha=0.15, label="Est. $\\pm$ 1 SD")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8, loc="lower right")

    for ax in axes[1, :]:
        ax.set_xlabel("Time $t$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\tau(t \mid x)$")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(FIGURES_DIR, "fig2_strata_averages.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_pointwise_scatter(t_grid, tau_hat, tau_true):
    """Figure 3: Scatterplots of true vs estimated CATE at 3 time horizons."""
    target_times = [1.0, 2.5, 4.0]
    t_indices = [int(np.argmin(np.abs(t_grid - t))) for t in target_times]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.suptitle("Pointwise CATE: True vs Estimated at Fixed Horizons", fontsize=14, fontweight="bold")

    for ax, t_idx, t_val in zip(axes, t_indices, target_times):
        true_vals = tau_true[:, t_idx]
        hat_vals = tau_hat[:, t_idx]

        ax.scatter(true_vals, hat_vals, s=8, alpha=0.5, color="C0", edgecolors="none")
        lims = [min(true_vals.min(), hat_vals.min()) - 0.05, max(true_vals.max(), hat_vals.max()) + 0.05]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.7, label="$y = x$")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(r"True $\tau(t \mid x)$")
        ax.set_title(f"$t = {t_val:.1f}$", fontsize=12)
        ax.set_aspect("equal")
        ax.legend(fontsize=9)

        # Correlation
        corr = np.corrcoef(true_vals, hat_vals)[0, 1]
        ax.text(0.05, 0.92, f"$r = {corr:.3f}$", transform=ax.transAxes, fontsize=10,
                verticalalignment="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    axes[0].set_ylabel(r"Estimated $\hat{\tau}(t \mid x)$")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(FIGURES_DIR, "fig3_pointwise_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_imse_by_x1(X_test, tau_hat, tau_true, delta):
    """Figure 4: Per-subject integrated MSE as a function of x_1."""
    per_subject_imse = np.sum(delta * (tau_hat - tau_true) ** 2, axis=1)
    x1 = X_test[:, 1]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x1, per_subject_imse, s=12, alpha=0.4, color="C0", edgecolors="none")

    # Bin-averaged line
    n_bins = 20
    bin_edges = np.linspace(x1.min(), x1.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.digitize(x1, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    bin_means = np.array([per_subject_imse[bin_idx == i].mean() if (bin_idx == i).sum() > 0 else np.nan for i in range(n_bins)])
    ax.plot(bin_centers, bin_means, "C3-o", linewidth=2, markersize=5, label="Bin average")

    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("Per-subject integrated MSE", fontsize=12)
    ax.set_title("Estimation Error vs Treatment Effect Modifier $x_1$", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig4_imse_by_x1.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def main():
    # Use non-interactive backend for saving figures
    import matplotlib
    matplotlib.use("Agg")

    t_grid, X_test, tau_hat, tau_true, delta = fit_estimator()

    print("Generating plots...")
    plot_individual_curves(t_grid, X_test, tau_hat, tau_true)
    plot_strata_averages(t_grid, X_test, tau_hat, tau_true)
    plot_pointwise_scatter(t_grid, tau_hat, tau_true)
    plot_imse_by_x1(X_test, tau_hat, tau_true, delta)
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
