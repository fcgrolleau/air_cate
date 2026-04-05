"""Convergence simulation: iMSE vs sample size for 6 methods × 2 DGPs.

Methods:
  1. AIR          — constrained Gamma mixture + estimated nuisance
  2. Tensor-prod  — unconstrained polynomial basis + estimated nuisance
  3. AIR-oracle   — mixture + oracle (true) nuisance
  4. TP-oracle    — polynomial basis + oracle nuisance
  5. RSF-T        — Random Survival Forest T-learner
  6. RSF-S        — Random Survival Forest S-learner

Usage:
  uv run python examples/plot_simulation.py           # quick (~5 min)
  uv run python examples/plot_simulation.py --large   # full scale
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from air_cate import AIREstimator
from air_cate.baselines import SurvivalForestTLearner, SurvivalForestSLearner
from air_cate.cate_model.tensor_product import CATETensorProduct
from air_cate.nuisance.oracle import build_oracle_nuisances
from air_cate.utils import build_time_grid
from conftest import WeibullDGP, WeibullMixtureDGP

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

QUICK_CONFIG = dict(
    n_values=[200, 500, 1000, 2000, 5000],
    n_seeds=2,
    n_test=500,
    M=30,
    n_folds=2,
    nuisance_epochs=30,
    cate_epochs=100,
    cate_lr=1e-3,
    K=3,
    encoder_hidden=(32, 16),
    n_time_basis=8,
    poly_degree=2,
    rsf_n_estimators=100,
    rsf_min_samples_leaf=15,
)

LARGE_CONFIG = dict(
    n_values=[500, 1000, 2000, 3000, 5000],
    n_seeds=5,
    n_test=1000,
    M=50,
    n_folds=3,
    nuisance_epochs=80,
    cate_epochs=300,
    cate_lr=5e-4,
    K=3,
    encoder_hidden=(64, 32),
    n_time_basis=8,
    poly_degree=2,
    rsf_n_estimators=200,
    rsf_min_samples_leaf=10,
)

TEST_SEED = 9999

METHODS = [
    ("AIR",        dict(oracle=False, tensor=False, rsf=None)),
    ("Tensor-prod", dict(oracle=False, tensor=True,  rsf=None)),
    ("AIR-oracle", dict(oracle=True,  tensor=False, rsf=None)),
    ("TP-oracle",  dict(oracle=True,  tensor=True,  rsf=None)),
    ("RSF-T",      dict(oracle=False, tensor=False, rsf="T")),
    ("RSF-S",      dict(oracle=False, tensor=False, rsf="S")),
]

DGPS = [
    ("Weibull", WeibullDGP),
    ("Mixture", WeibullMixtureDGP),
]

# JAMA palette
_JAMA = {
    "teal":   "#374e55",
    "orange": "#df8f44",
    "blue":   "#00a1d5",
    "red":    "#b24745",
    "green":  "#79af97",
}

METHOD_LABELS = {
    "AIR":         "Cross-fitted AIR w/ MDG",
    "Tensor-prod": "Cross-fitted AIR w/ TP",
    "AIR-oracle":  "AIR w/ MDG (oracle nuis.)",
    "TP-oracle":   "AIR w/ TP (oracle nuis.)",
    "RSF-T":       "Survival Forest T-learner",
    "RSF-S":       "Survival Forest S-learner",
}

METHOD_STYLES = {
    "AIR":         dict(color=_JAMA["teal"],   marker="o", ls="-",  lw=2.2, ms=6, zorder=5),
    "Tensor-prod": dict(color=_JAMA["orange"], marker="s", ls="-",  lw=2.2, ms=6, zorder=4),
    "AIR-oracle":  dict(color=_JAMA["teal"],   marker="o", ls="--", lw=1.4, ms=4.5, zorder=3),
    "TP-oracle":   dict(color=_JAMA["orange"], marker="s", ls="--", lw=1.4, ms=4.5, zorder=3),
    "RSF-T":       dict(color=_JAMA["red"],    marker="^", ls="-",  lw=2.2, ms=6, zorder=4),
    "RSF-S":       dict(color=_JAMA["green"],  marker="D", ls="-",  lw=2.2, ms=6, zorder=4),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_estimator(cfg, oracle, tensor, rsf):
    """Build an estimator for a given method configuration."""
    if rsf == "T":
        return SurvivalForestTLearner(
            M=cfg["M"],
            n_estimators=cfg["rsf_n_estimators"],
            min_samples_leaf=cfg["rsf_min_samples_leaf"],
        )
    if rsf == "S":
        return SurvivalForestSLearner(
            M=cfg["M"],
            n_estimators=cfg["rsf_n_estimators"],
            min_samples_leaf=cfg["rsf_min_samples_leaf"],
        )

    common = dict(
        M=cfg["M"],
        n_folds=cfg["n_folds"],
        eps=0.01,
        nuisance_training_config={"epochs": cfg["nuisance_epochs"], "verbose": 0},
        cate_epochs=cfg["cate_epochs"],
        cate_lr=cfg["cate_lr"],
    )
    if tensor:
        common["cate_model_class"] = CATETensorProduct
        common["cate_model_kwargs"] = dict(
            n_time_basis=cfg["n_time_basis"],
            poly_degree=cfg["poly_degree"],
        )
    else:
        common["K"] = cfg["K"]
        common["encoder_hidden_sizes"] = cfg["encoder_hidden"]
    return AIREstimator(**common)


def run_single(dgp_class, cfg, n, seed, X_test, oracle, tensor, rsf):
    """Fit one method on one (n, seed) draw and return iMSE."""
    np.random.seed(seed)
    X_train, W, Z, Delta, t_end = dgp_class.generate(n=n, rng_seed=seed)

    est = make_estimator(cfg, oracle, tensor, rsf)

    nuisances = None
    if oracle:
        time_grid, _ = build_time_grid(t_end, cfg["M"])
        nuisances = build_oracle_nuisances(dgp_class, X_train, Z, time_grid)

    est.fit(X_train, W, Z, Delta, t_end, nuisances=nuisances, verbose=0)

    tau_hat = est.predict_tau(X_test)
    t_grid = np.array(est.time_grid_)
    tau_true = dgp_class.true_tau(t_grid, X_test)
    delta = est.delta_

    imse_per_subject = np.sum(delta * (tau_hat - tau_true) ** 2, axis=1)
    return float(np.mean(imse_per_subject))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_all(cfg):
    """Run all experiments and return nested dict: results[dgp][method][n] = [imse, ...]."""
    seeds = list(range(cfg["n_seeds"]))
    n_values = cfg["n_values"]
    rng = np.random.default_rng(TEST_SEED)
    X_test = rng.uniform(-1, 1, (cfg["n_test"], 5)).astype(np.float32)

    total = len(DGPS) * len(METHODS) * len(n_values) * len(seeds)
    results = {}
    counter = 0
    t0 = time.time()

    for dgp_name, dgp_class in DGPS:
        results[dgp_name] = {}
        for method_name, method_opts in METHODS:
            results[dgp_name][method_name] = {n: [] for n in n_values}
            for n in n_values:
                for seed in seeds:
                    counter += 1
                    tag = f"[{counter}/{total}] {dgp_name}/{method_name} n={n} seed={seed}"
                    print(f"  {tag} ...", end=" ", flush=True)
                    imse = run_single(
                        dgp_class, cfg, n, seed, X_test,
                        oracle=method_opts["oracle"],
                        tensor=method_opts["tensor"],
                        rsf=method_opts["rsf"],
                    )
                    elapsed = time.time() - t0
                    print(f"iMSE={imse:.6f}  ({elapsed:.0f}s)")
                    results[dgp_name][method_name][n].append(imse)

    return results, n_values, X_test


RESULTS_PATH = os.path.join(FIGURES_DIR, "simulation_results.json")


def save_results(results, n_values, cfg):
    """Persist results to JSON (int keys become strings; reconverted on load)."""
    payload = {
        "n_values": n_values,
        "cfg": {k: v for k, v in cfg.items()
                if not callable(v) and k != "encoder_hidden"},
        "results": {
            dgp: {method: {str(n): vals for n, vals in by_n.items()}
                   for method, by_n in by_method.items()}
            for dgp, by_method in results.items()
        },
    }
    # Store encoder_hidden as list (tuples aren't JSON serializable)
    payload["cfg"]["encoder_hidden"] = list(cfg.get("encoder_hidden", ()))
    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to {RESULTS_PATH}")


def load_results():
    """Load results from JSON, restoring int n-keys."""
    with open(RESULTS_PATH) as f:
        payload = json.load(f)
    n_values = payload["n_values"]
    cfg = payload["cfg"]
    cfg["encoder_hidden"] = tuple(cfg["encoder_hidden"])
    results = {
        dgp: {method: {int(n): vals for n, vals in by_n.items()}
               for method, by_n in by_method.items()}
        for dgp, by_method in payload["results"].items()
    }
    return results, n_values, cfg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _pop_avg_imse(dgp_class, X_test, M):
    """iMSE of the population-average CATE (no heterogeneity baseline).

    tau_bar(t) = mean_i tau_true(t, X_i),  then
    iMSE = mean_i sum_m delta * (tau_bar(t_m) - tau_true(t_m, X_i))^2.
    """
    t_end = 5.0
    time_grid, delta = build_time_grid(t_end, M)
    t_grid = np.asarray(time_grid)
    tau_true = dgp_class.true_tau(t_grid, X_test)        # (n_test, M+1)
    tau_bar = tau_true.mean(axis=0, keepdims=True)        # (1, M+1)
    imse = np.mean(np.sum(delta * (tau_bar - tau_true) ** 2, axis=1))
    return float(imse)


def plot_convergence(results, n_values, cfg, X_test):
    """Two-panel convergence plot (one per DGP), one curve per method."""
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, (dgp_name, dgp_class) in zip(axes, DGPS):
        dgp_results = results[dgp_name]

        # --- Population-average reference line ---
        pa_imse = _pop_avg_imse(dgp_class, X_test, cfg["M"])
        ax.axhline(pa_imse, color="#888888", ls="--", linewidth=1.2,
                    zorder=1, label="True tv-ATE")

        # --- Method curves with shaded SE bands ---
        for method_name, _ in METHODS:
            vals = dgp_results[method_name]
            means = np.array([np.mean(vals[n]) for n in n_values])
            ses = np.array([np.std(vals[n]) / max(np.sqrt(len(vals[n])), 1)
                            for n in n_values])
            st = METHOD_STYLES[method_name]
            label = METHOD_LABELS[method_name]

            ax.plot(
                n_values, means,
                marker=st["marker"], ls=st["ls"],
                linewidth=st["lw"], markersize=st["ms"],
                color=st["color"],
                markeredgecolor="white", markeredgewidth=0.5,
                label=label, zorder=st["zorder"],
            )
            ax.fill_between(
                n_values, means - ses, means + ses,
                color=st["color"], alpha=0.12, zorder=st["zorder"] - 1,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Sample size", fontsize=10.5)
        ax.set_title(f"{dgp_name} DGP", fontsize=11.5, fontweight="bold", pad=8)
        ax.set_xticks(n_values)
        ax.set_xticklabels([str(n) for n in n_values], fontsize=8.5)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.grid(True, alpha=0.12, linewidth=0.4)

    axes[0].set_ylabel("Mean Squared Error", fontsize=10.5)

    # Shared legend below panels — order: reference, then solid, then dashed
    handles, labels = axes[0].get_legend_handles_labels()
    # Reorder: True tv-ATE first, then solid methods, then oracle (dashed)
    order = [0, 1, 2, 5, 6, 3, 4]  # tv-ATE, AIR, TP, RSF-T, RSF-S, AIR-orc, TP-orc
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.04),
               columnspacing=1.5, handletextpad=0.6)

    fig.tight_layout(rect=[0, 0.10, 1, 1], w_pad=2.5)
    path = os.path.join(FIGURES_DIR, "simulation_convergence.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def print_summary(results, n_values):
    """Console summary table."""
    print("\n" + "=" * 70)
    print(f"{'DGP':<10} {'Method':<14} {'n':>6} {'mean_iMSE':>12} {'SE':>10}")
    print("-" * 70)
    for dgp_name, _ in DGPS:
        for method_name, _ in METHODS:
            for n in n_values:
                vals = results[dgp_name][method_name][n]
                m = np.mean(vals)
                se = np.std(vals) / max(np.sqrt(len(vals)), 1)
                print(f"{dgp_name:<10} {method_name:<14} {n:>6d} {m:>12.6f} {se:>10.6f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convergence simulation")
    parser.add_argument("--large", action="store_true",
                        help="Use large-scale settings (slower, more seeds/sizes)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Re-plot from saved results (no simulation)")
    args = parser.parse_args()

    if args.plot_only:
        print("Loading saved results...")
        results, n_values, cfg = load_results()
        # Recreate the same test set used during the run
        rng = np.random.default_rng(TEST_SEED)
        X_test = rng.uniform(-1, 1, (cfg["n_test"], 5)).astype(np.float32)
        print_summary(results, n_values)
        print("\nGenerating plots...")
        plot_convergence(results, n_values, cfg, X_test)
        print(f"All figures saved to {FIGURES_DIR}/")
        return

    cfg = LARGE_CONFIG if args.large else QUICK_CONFIG

    print("=" * 70)
    print("Convergence Simulation: iMSE vs Sample Size")
    print(f"  n_values = {cfg['n_values']}")
    print(f"  n_seeds  = {cfg['n_seeds']}")
    print(f"  n_test   = {cfg['n_test']}")
    print(f"  methods  = {[m for m, _ in METHODS]}")
    print(f"  DGPs     = {[d for d, _ in DGPS]}")
    n_fits = len(DGPS) * len(METHODS) * len(cfg['n_values']) * cfg['n_seeds']
    print(f"  Total fits: {n_fits}")
    print("=" * 70)

    results, n_values, X_test = run_all(cfg)
    save_results(results, n_values, cfg)
    print_summary(results, n_values)

    print("\nGenerating plots...")
    plot_convergence(results, n_values, cfg, X_test)
    print(f"All figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
