"""Convergence simulation: iMSE vs sample size for 6 methods × 3 DGPs.

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

Smoke test (all DGPs and methods, two seeds, two small sample sizes). Keep
--figures-dir off examples/figures/, which holds the tracked publication figures:
  ... --n-values 500,1000 --extra-seeds 2 --output /tmp/smoke.json
  ... --plot-only --output /tmp/smoke.json --figures-dir examples/figures/smoke

One (DGP, method, seed-block) shard -- how the SLURM array in sherlock/ drives it:
  ... --large --dgp Weibull --methods AIR --seed-start 40 --extra-seeds 5 \
      --output partial_Weibull_AIR_b8.json

Then collect:
  ... --merge partial_*.json --output merged.json
  ... --plot-only --output merged.json
"""

import argparse
import gc
import json
import os
import sys
import time

# Limit CPU parallelism to avoid starving the system (watchdog resets).
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from scipy.stats import chi2 as chi2_dist
from scipy.stats import t as t_dist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import jax

# air_cate sets KERAS_BACKEND=jax on import, so it must come before keras.
from air_cate import AIREstimator

import keras
from air_cate.baselines import SurvivalForestTLearner, SurvivalForestSLearner
from air_cate.cate_model.tensor_product import CATETensorProduct
from air_cate.nuisance.oracle import build_oracle_nuisances
from air_cate.utils import build_time_grid
from conftest import WeibullDGP, WeibullMixtureDGP
from air_cate.dgp import IST3TextDGP

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

QUICK_CONFIG = dict(
    n_values=[500, 1000, 2000, 5000, 10000],
    n_seeds=2,
    n_test=500,
    M=30,
    n_folds=3,
    nuisance_epochs=200,
    nuisance_hidden=(64, 32),
    cate_epochs=200,
    cate_lr=1e-3,
    cate_patience=30,
    # 1, not the archived run's 5. Best-of-5 buys nothing here at 5x the CATE-stage
    # cost: the restarts land within ~0.7% of each other in training loss, so there is
    # no local optimum for the search to escape. Ranking them by *training* loss also
    # selects for noise-fit, which on a null DGP looked worse rather than better,
    # though not decisively. See the n_restarts entry on AIREstimator.
    n_restarts=1,
    K=2,
    encoder_hidden=(16,),
    n_time_basis=8,
    poly_degree=2,
    rsf_n_estimators=100,
    rsf_min_samples_leaf=10,
    # Text DGP overrides
    text_encoder_hidden=(16,),
    text_K=4,
)

LARGE_CONFIG = dict(
    n_values=[500, 1000, 2000, 5000, 10000, 20000],
    n_seeds=5,
    n_test=1000,
    M=30,
    n_folds=3,
    nuisance_epochs=200,
    nuisance_hidden=(64, 32),
    cate_epochs=200,
    cate_lr=1e-3,
    cate_patience=30,
    # See QUICK_CONFIG: 1 rather than the archived run's 5, deliberately.
    n_restarts=1,
    K=2,
    encoder_hidden=(16,),
    n_time_basis=8,
    poly_degree=2,
    rsf_n_estimators=50,
    rsf_min_samples_leaf=15,
    rsf_max_depth=15,
    # Text DGP overrides
    text_encoder_hidden=(16,),
    text_K=4,
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
    ("IST-3",   IST3TextDGP),
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
    "AIR":         "Cross-fitted AIR w/ MGD",
    "Tensor-prod": "Cross-fitted AIR w/ TP",
    "AIR-oracle":  "AIR w/ MGD (oracle nuis.)",
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

def make_estimator(cfg, oracle, tensor, rsf, is_text_dgp=False):
    """Build an estimator for a given method configuration."""
    if rsf == "T" or rsf == "S":
        cls = SurvivalForestTLearner if rsf == "T" else SurvivalForestSLearner
        return cls(
            M=cfg["M"],
            n_estimators=cfg["rsf_n_estimators"],
            max_depth=cfg.get("rsf_max_depth"),
            min_samples_leaf=cfg["rsf_min_samples_leaf"],
            n_jobs=cfg.get("rsf_n_jobs", 2),
        )

    nuis_hidden = cfg.get("nuisance_hidden")
    # Text DGP features are PCA-reduced before the estimator (in run_single),
    # so nuisance and encoder networks use standard sizes.
    if is_text_dgp:
        encoder_hidden = cfg.get("text_encoder_hidden", (16,))
    else:
        encoder_hidden = cfg["encoder_hidden"]

    common = dict(
        M=cfg["M"],
        n_folds=cfg["n_folds"],
        eps=0.01,
        nuisance_training_config={"epochs": cfg["nuisance_epochs"], "verbose": 0},
        cate_epochs=cfg["cate_epochs"],
        cate_lr=cfg["cate_lr"],
        # `patience` is main's name for what sherlock-sims called `cate_patience`.
        patience=cfg["cate_patience"],
        # 1 at both configs' defaults -- see the comment in QUICK_CONFIG for why the
        # archived run's best-of-5 was dropped rather than carried over.
        n_restarts=cfg["n_restarts"],
    )
    if nuis_hidden:
        common["weibull_config"] = {"hidden_sizes": nuis_hidden}
        common["propensity_config"] = {"hidden_sizes": nuis_hidden}
    if tensor:
        common["cate_model_class"] = CATETensorProduct
        common["cate_model_kwargs"] = dict(
            n_time_basis=cfg["n_time_basis"],
            poly_degree=cfg["poly_degree"],
        )
    else:
        common["K"] = cfg.get("text_K", cfg["K"]) if is_text_dgp else cfg["K"]
        common["encoder_hidden_sizes"] = encoder_hidden
    return AIREstimator(**common)


def run_single(dgp_class, cfg, n, seed, X_test, oracle, tensor, rsf):
    """Fit one method on one (n, seed) draw and return detailed results dict."""
    # Seed python, numpy, AND the Keras global generator. Seeding numpy alone
    # leaves network initialization keyed to Keras's own counter, which advances
    # with every fit already done in this process -- so a cell's init depended on
    # how many cells preceded it, and a requeued SLURM task started from a
    # different point than a fresh one.
    #
    # This makes the draw and the init order-independent, but NOT bit-reproducible:
    # two identical invocations still differ, because XLA's CPU reductions are not
    # run-to-run deterministic and early stopping amplifies a tiny numeric difference
    # into a different stopping epoch. Measured on one cell (Weibull/AIR-oracle/
    # n=5000/seed 0) the spread over four runs was 0.0152-0.0163, ~7%. That is well
    # under the ~20-25% between-seed spread, so it does not bias the averages.
    keras.utils.set_random_seed(seed)
    t_start = time.time()
    X_train_raw, W, Z, Delta, t_end = dgp_class.generate(n=n, rng_seed=seed)

    # Normalize IST-3 family covariates (feature scales vary wildly)
    normalize = dgp_class is IST3TextDGP

    # For text DGP: RSF baselines get only structured covariates
    is_text_dgp = hasattr(dgp_class, "p_struct")
    if is_text_dgp and rsf:
        p_s = dgp_class.p_struct
        X_train_features = X_train_raw[:, :p_s]
        X_test_features = X_test[:, :p_s]
    else:
        X_train_features = X_train_raw
        X_test_features = X_test

    if normalize:
        x_mean = X_train_features.mean(axis=0)
        x_std = X_train_features.std(axis=0) + 1e-8
        X_train = ((X_train_features - x_mean) / x_std).astype(np.float32)
        X_test_est = ((X_test_features - x_mean) / x_std).astype(np.float32)
    else:
        X_train = X_train_features
        X_test_est = X_test_features

    est = make_estimator(cfg, oracle, tensor, rsf, is_text_dgp=is_text_dgp)

    nuisances = None
    if oracle:
        time_grid, _ = build_time_grid(t_end, cfg["M"])
        # Oracle needs original X (DGP formulas use stored pool indices)
        nuisances = build_oracle_nuisances(dgp_class, X_train_raw, Z, time_grid)

    est.fit(X_train, W, Z, Delta, t_end, nuisances=nuisances, verbose=0)

    tau_hat = est.predict_tau(X_test_est)
    t_grid = np.array(est.time_grid_)
    tau_true = dgp_class.true_tau(t_grid, X_test)  # uses stored pool indices
    delta = est.delta_

    mse_per_timepoint = np.mean((tau_hat - tau_true) ** 2, axis=0)
    imse = float(np.sum(delta * mse_per_timepoint))
    runtime = time.time() - t_start

    out = {
        "seed": seed,
        "imse": imse,
        "mse_per_timepoint": [float(v) for v in mse_per_timepoint],
        "time_grid": [float(v) for v in t_grid],
        "runtime_seconds": round(runtime, 1),
    }

    # Record the per-restart training losses so the run itself answers whether
    # best-of-n_restarts is earning its ~5x CATE-stage cost. If the spread within a
    # cell is negligible, every restart found the same optimum and the search is
    # waste; if the winner is well below the others, it is not. The RSF baselines
    # have no restart loop, hence the getattr guard.
    restart_losses = getattr(est, "restart_losses_", None)
    if restart_losses:
        out["restart_losses"] = [float(v) for v in restart_losses]

    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _make_test_set(dgp_class, n_test, rng_seed=TEST_SEED):
    """Create DGP-appropriate test covariates."""
    if hasattr(dgp_class, "generate_test_X"):
        return dgp_class.generate_test_X(n_test, rng_seed)
    rng = np.random.default_rng(rng_seed)
    return rng.uniform(-1, 1, (n_test, 5)).astype(np.float32)


def run_all(cfg, resume=False, results_path=None, dgps=None, methods=None):
    """Run all experiments and return nested dict: results[dgp][method][n] = [imse, ...]."""
    if results_path is None:
        results_path = RESULTS_PATH
    if dgps is None:
        dgps = DGPS
    if methods is None:
        methods = METHODS
    seed_start = cfg.get("seed_start", 0)
    seeds = list(range(seed_start, seed_start + cfg["n_seeds"]))
    n_values = cfg["n_values"]

    # Per-DGP test sets
    X_tests = {}
    for dgp_name, dgp_class in dgps:
        X_tests[dgp_name] = _make_test_set(dgp_class, cfg["n_test"])

    # Load prior results for resume
    prior = {}
    if resume and os.path.exists(results_path):
        try:
            prior, _, _ = load_results(results_path)
            n_prior = sum(
                len(v) for by_m in prior.values()
                for by_n in by_m.values() for v in by_n.values()
            )
            print(f"  Resuming: loaded {n_prior} prior results from {results_path}")
        except Exception:
            prior = {}

    total = len(dgps) * len(methods) * len(n_values) * len(seeds)
    results = {}
    counter = 0
    skipped = 0
    t0 = time.time()

    for dgp_name, dgp_class in dgps:
        results[dgp_name] = {}
        X_test = X_tests[dgp_name]
        for method_name, method_opts in methods:
            results[dgp_name][method_name] = {n: [] for n in n_values}
            for n in n_values:
                # Check for complete prior results (all seeds present)
                prior_vals = (prior.get(dgp_name, {})
                              .get(method_name, {}).get(n, []))
                if resume and len(prior_vals) >= len(seeds):
                    results[dgp_name][method_name][n] = prior_vals[:len(seeds)]
                    skipped += len(seeds)
                    counter += len(seeds)
                    print(f"  [{counter}/{total}] {dgp_name}/{method_name} "
                          f"n={n} -- cached ({len(seeds)} seeds)")
                    continue

                for seed in seeds:
                    counter += 1
                    tag = f"[{counter}/{total}] {dgp_name}/{method_name} n={n} seed={seed}"
                    print(f"  {tag} ...", end=" ", flush=True)
                    try:
                        result = run_single(
                            dgp_class, cfg, n, seed, X_test,
                            oracle=method_opts["oracle"],
                            tensor=method_opts["tensor"],
                            rsf=method_opts["rsf"],
                        )
                    except Exception as exc:
                        result = {
                            "seed": seed, "imse": float("nan"),
                            "mse_per_timepoint": [], "time_grid": [],
                            "runtime_seconds": 0.0,
                        }
                        print(f"FAILED ({exc})", end=" ")
                    elapsed = time.time() - t0
                    print(f"iMSE={result['imse']:.6f}  ({elapsed:.0f}s)")
                    results[dgp_name][method_name][n].append(result)
                    gc.collect()
                    jax.clear_caches()

                # Save incrementally after each (dgp, method, n) block
                save_results(results, n_values, cfg, results_path=results_path)

    if skipped:
        print(f"  Skipped {skipped}/{total} fits from prior run")

    return results, n_values, X_tests


RESULTS_PATH = os.path.join(FIGURES_DIR, "simulation_results.json")


def save_results(results, n_values, cfg, results_path=None):
    """Persist results to JSON (int keys become strings; reconverted on load)."""
    if results_path is None:
        results_path = RESULTS_PATH
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
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to {results_path}")


def load_results(results_path=None):
    """Load results from JSON, restoring int n-keys."""
    if results_path is None:
        results_path = RESULTS_PATH
    with open(results_path) as f:
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


def merge_partial_results(main_path, partial_paths):
    """Merge partial JSON result files into the main results file."""
    if os.path.exists(main_path):
        main_results, n_values, cfg = load_results(main_path)
    else:
        main_results = {}
        first = load_results(partial_paths[0])
        n_values, cfg = first[1], first[2]
    for path in partial_paths:
        partial, _, _ = load_results(path)
        for dgp in partial:
            if dgp not in main_results:
                main_results[dgp] = {}
            for method in partial[dgp]:
                if method not in main_results[dgp]:
                    main_results[dgp][method] = {}
                for n, vals in partial[dgp][method].items():
                    if n not in main_results[dgp][method]:
                        main_results[dgp][method][n] = []
                    main_results[dgp][method][n].extend(vals)
    save_results(main_results, n_values, cfg, results_path=main_path)
    print(f"Merged {len(partial_paths)} files into {main_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _pop_avg_imse(dgp_class, X_test, M):
    """iMSE of the population-average CATE (no heterogeneity baseline).

    tau_bar(t) = mean_i tau_true(t, X_i),  then
    iMSE = mean_i sum_m delta * (tau_bar(t_m) - tau_true(t_m, X_i))^2.
    """
    t_end = getattr(dgp_class, "t_end", 5.0)
    time_grid, delta = build_time_grid(t_end, M)
    t_grid = np.asarray(time_grid)
    tau_true = dgp_class.true_tau(t_grid, X_test)        # (n_test, M+1)
    tau_bar = tau_true.mean(axis=0, keepdims=True)        # (1, M+1)
    imse = np.mean(np.sum(delta * (tau_bar - tau_true) ** 2, axis=1))
    return float(imse)


def plot_convergence(results, n_values, cfg, X_tests, figures_dir=None):
    """Multi-panel convergence plot: mean iMSE (top) and variance (bottom) per DGP.

    figures_dir defaults to FIGURES_DIR, which holds the tracked publication
    figures -- pass somewhere else for smoke runs so they are not overwritten.
    """
    out_dir = figures_dir or FIGURES_DIR
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    n_dgps = len(DGPS)
    fig, axes = plt.subplots(2, n_dgps, figsize=(6.5 * n_dgps, 9),
                             sharex="col")
    if n_dgps == 1:
        axes = axes[:, None]  # ensure 2D
    ax_mean = axes[0]   # top row: mean iMSE
    ax_var = axes[1]     # bottom row: variance of iMSE

    for col, (dgp_name, dgp_class) in enumerate(DGPS):
        if dgp_name not in results:
            continue
        dgp_results = results[dgp_name]
        X_test = X_tests[dgp_name] if isinstance(X_tests, dict) else X_tests

        # --- Population-average reference line (top row only) ---
        pa_imse = _pop_avg_imse(dgp_class, X_test, cfg["M"])
        ax_mean[col].axhline(pa_imse, color="#888888", ls="--", linewidth=1.2,
                              zorder=1, label="True tv-ATE")

        # --- Method curves ---
        for method_name, _ in METHODS:
            if method_name not in dgp_results:
                continue
            vals = dgp_results[method_name]
            st = METHOD_STYLES[method_name]
            label = METHOD_LABELS[method_name]

            # Get n_values that exist in results for this method (skip all-NaN)
            def _imses(n):
                return [v["imse"] for v in vals[n]]

            ns = [n for n in n_values if n in vals
                  and len(vals[n]) > 0
                  and np.any(np.isfinite(_imses(n)))]
            if not ns:
                continue

            # --- Top row: mean iMSE with 95% CI (t-distribution) ---
            def _finite(arr):
                a = np.asarray(arr, dtype=float)
                return a[np.isfinite(a)]

            means = np.array([np.mean(_finite(_imses(n))) for n in ns])
            ses = np.array([np.std(_finite(_imses(n)), ddof=1)
                            / np.sqrt(len(_finite(_imses(n))))
                            for n in ns])
            t_crit = np.array([t_dist.ppf(0.975, max(len(_finite(_imses(n))) - 1, 1))
                               for n in ns])
            hw = t_crit * ses

            ax_mean[col].plot(
                ns, means,
                marker=st["marker"], ls=st["ls"],
                linewidth=st["lw"], markersize=st["ms"],
                color=st["color"],
                markeredgecolor="white", markeredgewidth=0.5,
                label=label, zorder=st["zorder"],
            )
            ax_mean[col].fill_between(
                ns, means - hw, means + hw,
                color=st["color"], alpha=0.12, zorder=st["zorder"] - 1,
            )

            # --- Bottom row: variance of iMSE with 95% CI (chi-squared) ---
            variances = np.array([np.var(_finite(_imses(n)), ddof=1) for n in ns])
            var_lo = np.array([
                (len(_finite(_imses(n))) - 1) * np.var(_finite(_imses(n)), ddof=1)
                / chi2_dist.ppf(0.975, max(len(_finite(_imses(n))) - 1, 1))
                for n in ns
            ])
            var_hi = np.array([
                (len(_finite(_imses(n))) - 1) * np.var(_finite(_imses(n)), ddof=1)
                / chi2_dist.ppf(0.025, max(len(_finite(_imses(n))) - 1, 1))
                for n in ns
            ])

            ax_var[col].plot(
                ns, variances,
                marker=st["marker"], ls=st["ls"],
                linewidth=st["lw"], markersize=st["ms"],
                color=st["color"],
                markeredgecolor="white", markeredgewidth=0.5,
                label=label, zorder=st["zorder"],
            )
            ax_var[col].fill_between(
                ns, var_lo, var_hi,
                color=st["color"], alpha=0.12, zorder=st["zorder"] - 1,
            )

        # --- Axes formatting ---
        for row_axes in (ax_mean, ax_var):
            row_axes[col].set_xscale("log")
            row_axes[col].set_yscale("log")
            row_axes[col].set_xticks(n_values)
            row_axes[col].set_xticklabels([str(n) for n in n_values], fontsize=8)
            row_axes[col].tick_params(axis="y", labelsize=8.5)
            row_axes[col].grid(True, alpha=0.12, linewidth=0.4)

        ax_mean[col].set_title(f"{dgp_name} DGP", fontsize=11.5,
                                fontweight="bold", pad=8)
        ax_var[col].set_xlabel("Sample size", fontsize=10.5)

    ax_mean[0].set_ylabel("Average MSE", fontsize=10.5)
    ax_var[0].set_ylabel("Variance of MSE", fontsize=10.5)

    # Shared legend below all panels
    handles, labels = ax_mean[0].get_legend_handles_labels()
    # Reorder: True tv-ATE first, then solid methods, then oracle (dashed)
    order = [0, 1, 2, 5, 6, 3, 4]  # tv-ATE, AIR, TP, RSF-T, RSF-S, AIR-orc, TP-orc
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.02),
               columnspacing=1.5, handletextpad=0.6)

    fig.tight_layout(rect=[0, 0.06, 1, 1], w_pad=2.5, h_pad=1.5)
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"simulation_convergence.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def print_summary(results, n_values):
    """Console summary table."""
    print("\n" + "=" * 70)
    print(f"{'DGP':<12} {'Method':<14} {'n':>6} {'mean_iMSE':>12} {'SE':>10}")
    print("-" * 70)
    for dgp_name in results:
        for method_name in results[dgp_name]:
            for n in n_values:
                entries = results[dgp_name].get(method_name, {}).get(n, [])
                if not entries:
                    continue
                vals = [v["imse"] for v in entries]
                m = np.mean(vals)
                se = np.std(vals) / max(np.sqrt(len(vals)), 1)
                print(f"{dgp_name:<12} {method_name:<14} {n:>6d} {m:>12.6f} {se:>10.6f}")
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
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, ignoring any saved partial results")
    parser.add_argument("--seed-start", type=int, default=0,
                        help="First seed index (default 0)")
    parser.add_argument("--extra-seeds", type=int, default=None,
                        help="Run N extra seeds starting from --seed-start, "
                             "saving to a separate file (does not overwrite main results)")
    parser.add_argument("--dgp", type=str, default=None,
                        help="Filter to a single DGP by name (e.g. 'IST-3 Text')")
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated method names to run (e.g. 'AIR,RSF-T')")
    parser.add_argument("--n-values", type=str, default=None,
                        help="Comma-separated sample sizes, overriding the config "
                             "(e.g. '500,1000' for a smoke test)")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output JSON path")
    parser.add_argument("--figures-dir", type=str, default=None,
                        help="Where to write figures (default examples/figures/, which "
                             "holds the tracked publication figures)")
    parser.add_argument("--merge", nargs="+", metavar="FILE",
                        help="Merge partial result files into simulation_results.json")
    args = parser.parse_args()

    # --- Merge mode ---
    if args.merge:
        merge_path = args.output or RESULTS_PATH
        merge_partial_results(merge_path, args.merge)
        return

    # --- Plot-only mode ---
    if args.plot_only:
        print("Loading saved results...")
        results, n_values, cfg = load_results(args.output or RESULTS_PATH)
        # Recreate per-DGP test sets
        X_tests = {}
        for dgp_name, dgp_class in DGPS:
            X_tests[dgp_name] = _make_test_set(dgp_class, cfg["n_test"])
        print_summary(results, n_values)
        print("\nGenerating plots...")
        plot_convergence(results, n_values, cfg, X_tests, figures_dir=args.figures_dir)
        print(f"All figures saved to {args.figures_dir or FIGURES_DIR}/")
        return

    # --- Config selection ---
    if args.large:
        cfg = LARGE_CONFIG
    else:
        cfg = QUICK_CONFIG

    if args.n_values:
        try:
            n_values = [int(v) for v in args.n_values.split(",")]
        except ValueError:
            parser.error(f"--n-values must be comma-separated integers, got {args.n_values!r}")
        if not n_values:
            parser.error("--n-values must list at least one sample size")
        cfg = dict(cfg)  # shallow copy so we don't mutate the constant
        cfg["n_values"] = n_values

    # --- DGP / method filters ---
    dgps = DGPS
    if args.dgp:
        dgps = [(n, c) for n, c in DGPS if n == args.dgp]
        if not dgps:
            valid = [n for n, _ in DGPS]
            parser.error(f"Unknown DGP '{args.dgp}'. Valid: {valid}")

    methods = METHODS
    if args.methods:
        method_names = [m.strip() for m in args.methods.split(",")]
        methods = [(n, o) for n, o in METHODS if n in method_names]
        if not methods:
            valid = [n for n, _ in METHODS]
            parser.error(f"No matching methods. Valid: {valid}")

    # --- Extra-seeds mode ---
    if args.extra_seeds is not None:
        cfg = dict(cfg)  # shallow copy so we don't mutate the constant
        cfg["n_seeds"] = args.extra_seeds
        cfg["seed_start"] = args.seed_start
        extra_tag = f"seeds{args.seed_start}-{args.seed_start + args.extra_seeds - 1}"
        extra_path = os.path.join(FIGURES_DIR, f"simulation_results_{extra_tag}.json")
    else:
        cfg.setdefault("seed_start", 0)
        extra_path = None

    results_path = args.output or extra_path or RESULTS_PATH

    print("=" * 70)
    print("Convergence Simulation: iMSE vs Sample Size")
    print(f"  n_values = {cfg['n_values']}")
    seed_start = cfg.get("seed_start", 0)
    print(f"  seeds    = {seed_start}..{seed_start + cfg['n_seeds'] - 1} ({cfg['n_seeds']} seeds)")
    print(f"  n_test   = {cfg['n_test']}")
    print(f"  methods  = {[m for m, _ in methods]}")
    print(f"  DGPs     = {[d for d, _ in dgps]}")
    n_fits = len(dgps) * len(methods) * len(cfg['n_values']) * cfg['n_seeds']
    print(f"  Total fits: {n_fits}")
    print(f"  Output:  {results_path}")
    print("=" * 70)

    results, n_values, X_tests = run_all(
        cfg, resume=not args.no_resume, results_path=results_path,
        dgps=dgps, methods=methods,
    )
    save_results(results, n_values, cfg, results_path=results_path)
    print_summary(results, n_values)

    if not extra_path and not args.output:
        print("\nGenerating plots...")
        plot_convergence(results, n_values, cfg, X_tests, figures_dir=args.figures_dir)
        print(f"All figures saved to {args.figures_dir or FIGURES_DIR}/")


if __name__ == "__main__":
    main()
