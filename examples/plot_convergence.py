"""Convergence plot: iMSE vs training sample size for both DGPs."""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from air_cate import AIREstimator
from conftest import WeibullDGP, WeibullMixtureDGP

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# --- Experiment constants ---
N_VALUES = [500, 1000, 2000, 3000, 5000]
N_SEEDS = 3
N_TEST = 1000
TEST_SEED = 9999

ESTIMATOR_KWARGS = dict(
    K=3, M=50, n_folds=3, eps=0.01,
    encoder_hidden_sizes=(64, 32),
    nuisance_training_config={"epochs": 80, "verbose": 0},
    cate_epochs=300, cate_lr=5e-4,
)


def generate_test_set(p=5):
    """Fixed test covariates for iMSE evaluation (shared across all fits)."""
    rng = np.random.default_rng(TEST_SEED)
    return rng.uniform(-1, 1, (N_TEST, p)).astype(np.float32)


def run_single_experiment(dgp_class, n, seed, X_test):
    """Fit AIR on one (n, seed) draw and return mean iMSE on the fixed test set."""
    np.random.seed(seed)
    X_train, W, Z, Delta, t_end = dgp_class.generate(n=n, rng_seed=seed)

    est = AIREstimator(**ESTIMATOR_KWARGS)
    est.fit(X_train, W, Z, Delta, t_end, verbose=0)

    tau_hat = est.predict_tau(X_test)
    t_grid = np.array(est.time_grid_)
    tau_true = dgp_class.true_tau(t_grid, X_test)
    delta = est.delta_

    imse_per_subject = np.sum(delta * (tau_hat - tau_true) ** 2, axis=1)
    return float(np.mean(imse_per_subject))


def run_all_experiments(dgp_class, dgp_name, n_values, seeds, X_test):
    """Run all (n, seed) combinations for one DGP."""
    results = {n: [] for n in n_values}
    total = len(n_values) * len(seeds)
    counter = 0

    for n in n_values:
        for seed in seeds:
            counter += 1
            print(f"  [{counter}/{total}] {dgp_name} n={n}, seed={seed} ...",
                  end=" ", flush=True)
            imse = run_single_experiment(dgp_class, n, seed, X_test)
            results[n].append(imse)
            print(f"iMSE = {imse:.6f}")

    return results


def plot_convergence_curve(results_weibull, results_mixture, n_values):
    """Fig 12: Two-panel convergence curve with error bars."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, results, title in [
        (ax1, results_weibull, "Weibull DGP"),
        (ax2, results_mixture, "Weibull Mixture DGP (K=3)"),
    ]:
        means = [np.mean(results[n]) for n in n_values]
        ses = [np.std(results[n]) / np.sqrt(len(results[n])) for n in n_values]

        ax.errorbar(n_values, means, yerr=ses, fmt="o-", capsize=4,
                     linewidth=2, markersize=6, color="C0")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training sample size $n$", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(n_values)
        ax.set_xticklabels([str(n) for n in n_values])
        ax.grid(True, alpha=0.3)

    ax1.set_ylabel("Integrated MSE", fontsize=12)
    fig.suptitle(
        "Convergence: iMSE vs Training Sample Size\n"
        f"({N_SEEDS} seeds per $n$, $n_{{\\mathrm{{test}}}}$={N_TEST})",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    path = os.path.join(FIGURES_DIR, "fig12_convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_loglog_convergence(results_weibull, results_mixture, n_values):
    """Fig 13: Log-log with fitted power-law reference line."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, results, title in [
        (ax1, results_weibull, "Weibull DGP"),
        (ax2, results_mixture, "Weibull Mixture DGP (K=3)"),
    ]:
        means = np.array([np.mean(results[n]) for n in n_values])
        ses = np.array([np.std(results[n]) / np.sqrt(len(results[n]))
                        for n in n_values])
        log_n = np.log(np.array(n_values, dtype=float))
        log_imse = np.log(means)

        # Fit linear regression in log-log space: log(iMSE) = slope*log(n) + intercept
        slope, intercept = np.polyfit(log_n, log_imse, 1)

        ax.errorbar(n_values, means, yerr=ses, fmt="o", capsize=4,
                     markersize=6, color="C0", label="Mean iMSE", zorder=3)

        # Reference power-law line
        n_fine = np.logspace(
            np.log10(min(n_values) * 0.8),
            np.log10(max(n_values) * 1.2), 100,
        )
        ax.plot(n_fine, np.exp(intercept) * n_fine ** slope,
                "k--", linewidth=1.5, alpha=0.7,
                label=f"Fit: $n^{{{slope:.2f}}}$")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training sample size $n$", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    ax1.set_ylabel("Integrated MSE", fontsize=12)
    fig.suptitle(
        "Log-Log Convergence with Rate Estimate",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(FIGURES_DIR, "fig13_convergence_loglog.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def main():
    seeds = list(range(N_SEEDS))
    X_test = generate_test_set()

    print("=" * 60)
    print("Convergence Experiment: iMSE vs Sample Size")
    print(f"  n_values = {N_VALUES}")
    print(f"  seeds    = {seeds}")
    print(f"  n_test   = {N_TEST}")
    print(f"  Total fits per DGP: {len(N_VALUES) * len(seeds)}")
    print("=" * 60)

    print("\n--- WeibullDGP ---")
    results_weibull = run_all_experiments(
        WeibullDGP, "WeibullDGP", N_VALUES, seeds, X_test)

    print("\n--- WeibullMixtureDGP ---")
    results_mixture = run_all_experiments(
        WeibullMixtureDGP, "WeibullMixtureDGP", N_VALUES, seeds, X_test)

    # Summary table
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"{'DGP':<20} {'n':>6} {'mean_iMSE':>12} {'SE':>10}")
    print("-" * 50)
    for dgp_name, results in [("Weibull", results_weibull),
                               ("Mixture", results_mixture)]:
        for n in N_VALUES:
            vals = results[n]
            m = np.mean(vals)
            se = np.std(vals) / np.sqrt(len(vals))
            print(f"{dgp_name:<20} {n:>6d} {m:>12.6f} {se:>10.6f}")
    print("=" * 60)

    print("\nGenerating plots...")
    plot_convergence_curve(results_weibull, results_mixture, N_VALUES)
    plot_loglog_convergence(results_weibull, results_mixture, N_VALUES)
    print(f"\nAll convergence figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
