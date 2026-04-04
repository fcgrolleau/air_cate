"""Visualize CATE recovery over 2D covariate space for a 3-component Weibull mixture DGP."""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from air_cate import AIREstimator
from conftest import WeibullMixtureDGP

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Time slices to visualize
TARGET_TIMES = [1.0, 2.5, 4.0]
# Grid resolution
GRID_N = 40


def fit_and_evaluate():
    """Fit estimator with K=3, build (x_0, x_1) meshgrid, evaluate true and estimated CATE."""
    print("Generating training data (n=5000, WeibullMixtureDGP)...")
    X_train, W, Z, Delta, t_end = WeibullMixtureDGP.generate(n=5000, rng_seed=42)

    print("Fitting AIREstimator (K=3)...")
    np.random.seed(0)
    est = AIREstimator(
        K=3, M=50, n_folds=3, eps=0.01,
        encoder_hidden_sizes=(64, 32),
        nuisance_training_config={"epochs": 80, "verbose": 0},
        cate_epochs=300, cate_lr=5e-4,
    )
    est.fit(X_train, W, Z, Delta, t_end, verbose=1)

    # Build (x_0, x_1) meshgrid, hold x_2=x_3=x_4=0
    x0_vals = np.linspace(-0.9, 0.9, GRID_N)
    x1_vals = np.linspace(-0.9, 0.9, GRID_N)
    X0, X1 = np.meshgrid(x0_vals, x1_vals, indexing="ij")

    n_grid = GRID_N * GRID_N
    X_grid = np.zeros((n_grid, 5), dtype=np.float32)
    X_grid[:, 0] = X0.ravel()
    X_grid[:, 1] = X1.ravel()

    t_grid = np.array(est.time_grid_)
    tau_hat_flat = est.predict_tau(X_grid)
    tau_true_flat = WeibullMixtureDGP.true_tau(t_grid, X_grid)

    tau_hat = tau_hat_flat.reshape(GRID_N, GRID_N, -1)
    tau_true = tau_true_flat.reshape(GRID_N, GRID_N, -1)

    t_indices = [int(np.argmin(np.abs(t_grid - t))) for t in TARGET_TIMES]

    delta = est.delta_
    imse_flat = np.sum(delta * (tau_hat_flat - tau_true_flat) ** 2, axis=1)
    print(f"Mean integrated MSE on grid: {imse_flat.mean():.4f}\n")

    return x0_vals, x1_vals, X0, X1, tau_hat, tau_true, t_grid, t_indices


def plot_heatmaps(x0_vals, x1_vals, tau_hat, tau_true, t_grid, t_indices):
    """Figure 1: True (top) vs Estimated (bottom) CATE heatmaps at 3 time slices."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharex=True, sharey=True)
    fig.suptitle(
        r"True vs Estimated $\tau(t \mid x_0, x_1)$ — Weibull Mixture DGP (K=3)",
        fontsize=14, fontweight="bold",
    )

    vmax = 0
    for ti in t_indices:
        vmax = max(vmax, np.abs(tau_true[:, :, ti]).max(), np.abs(tau_hat[:, :, ti]).max())
    vmax = float(np.ceil(vmax * 20) / 20)

    for col, ti in enumerate(t_indices):
        t_val = float(t_grid[ti])

        im0 = axes[0, col].pcolormesh(
            x0_vals, x1_vals, tau_true[:, :, ti].T,
            cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto",
        )
        axes[0, col].set_title(f"True, $t = {t_val:.1f}$", fontsize=11)

        im1 = axes[1, col].pcolormesh(
            x0_vals, x1_vals, tau_hat[:, :, ti].T,
            cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto",
        )
        axes[1, col].set_title(f"Estimated, $t = {t_val:.1f}$", fontsize=11)

    for ax in axes[:, 0]:
        ax.set_ylabel("$x_1$")
    for ax in axes[1, :]:
        ax.set_xlabel("$x_0$")

    fig.colorbar(im0, ax=axes, shrink=0.8, label=r"$\tau(t \mid x)$")
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    path = os.path.join(FIGURES_DIR, "fig8_heatmaps_mixture.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_mse_heatmaps(x0_vals, x1_vals, tau_hat, tau_true, t_grid, t_indices):
    """Figure 2: Pointwise MSE heatmaps at 3 time slices."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.suptitle(
        r"Pointwise MSE $\left(\hat{\tau} - \tau\right)^2$ — Weibull Mixture DGP (K=3)",
        fontsize=14, fontweight="bold",
    )

    mse = (tau_hat - tau_true) ** 2
    vmax = 0
    for ti in t_indices:
        vmax = max(vmax, mse[:, :, ti].max())
    vmax = float(np.ceil(vmax * 100) / 100)

    for col, ti in enumerate(t_indices):
        t_val = float(t_grid[ti])
        im = axes[col].pcolormesh(
            x0_vals, x1_vals, mse[:, :, ti].T,
            cmap="hot_r", vmin=0, vmax=vmax, shading="auto",
        )
        axes[col].set_title(f"$t = {t_val:.1f}$", fontsize=11)
        axes[col].set_xlabel("$x_0$")

    axes[0].set_ylabel("$x_1$")
    fig.colorbar(im, ax=axes, shrink=0.8, label=r"$(\hat{\tau} - \tau)^2$")
    fig.tight_layout(rect=[0, 0, 0.92, 0.93])
    path = os.path.join(FIGURES_DIR, "fig9_mse_mixture.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_contour_overlay(x0_vals, x1_vals, X0, X1, tau_hat, tau_true, t_grid, t_indices):
    """Figure 3: Contour overlays — true (solid) vs estimated (dashed)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.suptitle(
        r"Contour Overlay: True (solid) vs Estimated (dashed) — Weibull Mixture DGP (K=3)",
        fontsize=14, fontweight="bold",
    )

    all_vals = []
    for ti in t_indices:
        all_vals.extend([tau_true[:, :, ti].ravel(), tau_hat[:, :, ti].ravel()])
    vmin = min(v.min() for v in all_vals)
    vmax = max(v.max() for v in all_vals)
    levels = np.linspace(vmin, vmax, 12)
    levels = np.sort(np.unique(np.concatenate([levels, [0.0]])))

    for col, ti in enumerate(t_indices):
        t_val = float(t_grid[ti])
        ax = axes[col]

        cs_true = ax.contour(
            x0_vals, x1_vals, tau_true[:, :, ti].T,
            levels=levels, colors="black", linewidths=1.5, linestyles="solid",
        )
        cs_hat = ax.contour(
            x0_vals, x1_vals, tau_hat[:, :, ti].T,
            levels=levels, colors="C0", linewidths=1.5, linestyles="dashed",
        )

        ax.clabel(cs_true, inline=True, fontsize=7, fmt="%.2f")
        ax.set_title(f"$t = {t_val:.1f}$", fontsize=11)
        ax.set_xlabel("$x_0$")

    axes[0].set_ylabel("$x_1$")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="black", linewidth=1.5, linestyle="solid", label="True"),
        Line2D([0], [0], color="C0", linewidth=1.5, linestyle="dashed", label="Estimated"),
    ]
    axes[-1].legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(FIGURES_DIR, "fig10_contour_mixture.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def main():
    results = fit_and_evaluate()
    x0_vals, x1_vals, X0, X1, tau_hat, tau_true, t_grid, t_indices = results

    print("Generating 2D mixture plots...")
    plot_heatmaps(x0_vals, x1_vals, tau_hat, tau_true, t_grid, t_indices)
    plot_mse_heatmaps(x0_vals, x1_vals, tau_hat, tau_true, t_grid, t_indices)
    plot_contour_overlay(x0_vals, x1_vals, X0, X1, tau_hat, tau_true, t_grid, t_indices)
    print(f"\nAll mixture figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
