"""Convergence simulation: four trajectory metrics vs sample size, 7 methods × 3 DGPs.

Methods:
  1. AIR          — constrained Gamma mixture + estimated nuisance
  2. Tensor-prod  — unconstrained polynomial basis + estimated nuisance
  3. AIR-oracle   — mixture + oracle (true) nuisance
  4. TP-oracle    — polynomial basis + oracle nuisance
  5. RSF-T        — Random Survival Forest T-learner
  6. RSF-S        — Random Survival Forest S-learner
  7. CSF          — grf causal survival forest, one refit per horizon

Metrics (see air_cate.metrics; every method is evaluated on the SAME uniform time
grid, and CSF's per-horizon refits land on exactly that grid):
  iMSE    — integrated pointwise squared error, as reported since the first run
  D       — derivative MSE, the first metric here that sees roughness
  H1      — mean H^1 Sobolev norm of the error
  Excess  — excess count of direction reversals against the truth

One figure per metric, each with the mean-vs-n and variance-vs-n panel pair. iMSE
keeps the filename `simulation_convergence.*`; the rest are suffixed.

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

Excess at both recorded reversal thresholds, one row per DGP and one column per
threshold, in a single figure (needs no test set, so no IST-3 embedding artifact):
  ... --plot-only --output merged.json --figures-dir <dir> --excess-thresholds
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
from matplotlib.ticker import MultipleLocator
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import jax

# air_cate sets KERAS_BACKEND=jax on import, so it must come before keras.
from air_cate import AIREstimator

import keras
from air_cate.baselines import SurvivalForestTLearner, SurvivalForestSLearner
from air_cate.baselines_csf import CausalSurvivalForest, grf_available
from air_cate.cate_model.tensor_product import CATETensorProduct
from air_cate.metrics import DEFAULT_THRESHOLD_FRAC, METRIC_KEYS, curve_metrics
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
    # Without a held-out block, `cate_patience` is inert: it would monitor the TRAINING
    # R-loss, which falls monotonically, so early stopping never fires and every fit
    # runs the full `cate_epochs` and over-fits. Measured on IST-3 at n=2000 over 20
    # paired seeds, that cost 0.192 vs 0.149 iMSE with oracle nuisances and 0.563 vs
    # 0.411 cross-fitted -- i.e. the whole of the apparent IST-3 regression against the
    # archived 100-seed run, which had been early-stopping by accident (a NaN killed
    # its fits at epoch 60-120; see gamma_survival.py).
    #
    # 0.2 rather than a tuned epoch cap: a hard cap at 100 epochs scored marginally
    # better here (0.374 cross-fitted, paired t=0.79 against the archive vs t=1.33 for
    # the split) but the number was chosen by looking at this DGP's answer and has no
    # defence beyond that. Held-out R-loss needs no new justification -- it is the same
    # criterion AIREstimatorCV uses to select K.
    cate_val_frac=0.2,
    # Rescale t to [0,1] inside CATEMixture. On by default because it is what the
    # archived 100-seed run did and it is measurably better: with it off, the oracle
    # IST-3 arm sits well above the archive, with it on it reproduces it (0.147 vs
    # 0.146). Affects the MGD model only; CATETensorProduct always normalizes.
    mgd_normalize_time=True,
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
    # grf causal survival forest defaults. num.trees is grf's own default; 1 thread
    # keeps the recorded runtimes comparable to the single-core numbers the table
    # reports (override with --csf-threads for a faster local run).
    csf_num_trees=2000,
    csf_num_threads=1,
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
    # See QUICK_CONFIG: without this, `cate_patience` cannot fire and every fit
    # over-trains to `cate_epochs`.
    cate_val_frac=0.2,
    # See QUICK_CONFIG: matches the archived run and reproduces its oracle arm.
    mgd_normalize_time=True,
    # See QUICK_CONFIG: 1 rather than the archived run's 5, deliberately.
    n_restarts=1,
    K=2,
    encoder_hidden=(16,),
    n_time_basis=8,
    poly_degree=2,
    rsf_n_estimators=50,
    rsf_min_samples_leaf=15,
    # See QUICK_CONFIG. Left at grf's defaults deliberately, so the baseline is the
    # method as a practitioner would reach for it.
    csf_num_trees=2000,
    csf_num_threads=1,
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
    # The `rsf` slot doubles as the external-learner discriminator, so adding grf's
    # causal survival forest needs no new key threaded through run_all -> run_single
    # -> make_estimator. It also means CSF inherits the text-DGP policy below, where
    # a truthy `rsf` restricts the learner to the structured covariates.
    ("CSF",        dict(oracle=False, tensor=False, rsf="CSF")),
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
    "purple": "#6a6599",
}

METHOD_LABELS = {
    "AIR":         "Cross-fitted AIR w/ MGD",
    "Tensor-prod": "Cross-fitted AIR w/ TP",
    "AIR-oracle":  "AIR w/ MGD (oracle nuis.)",
    "TP-oracle":   "AIR w/ TP (oracle nuis.)",
    "RSF-T":       "Survival Forest T-learner",
    "RSF-S":       "Survival Forest S-learner",
    "CSF":         "Causal Survival Forest (grf)",
}

METHOD_STYLES = {
    "AIR":         dict(color=_JAMA["teal"],   marker="o", ls="-",  lw=2.2, ms=6, zorder=5),
    "Tensor-prod": dict(color=_JAMA["orange"], marker="s", ls="-",  lw=2.2, ms=6, zorder=4),
    "AIR-oracle":  dict(color=_JAMA["teal"],   marker="o", ls="--", lw=1.4, ms=4.5, zorder=3),
    "TP-oracle":   dict(color=_JAMA["orange"], marker="s", ls="--", lw=1.4, ms=4.5, zorder=3),
    "RSF-T":       dict(color=_JAMA["red"],    marker="^", ls="-",  lw=2.2, ms=6, zorder=4),
    "RSF-S":       dict(color=_JAMA["green"],  marker="D", ls="-",  lw=2.2, ms=6, zorder=4),
    "CSF":         dict(color=_JAMA["purple"], marker="v", ls="-",  lw=2.2, ms=6, zorder=4),
}

# Which metrics get a figure, and how each one is drawn. `key` indexes the per-seed
# result dicts written by run_single; `stem` names the output file. iMSE deliberately
# keeps the bare `simulation_convergence` stem, because that is the filename the paper
# includes.
#
# `log_y` is per metric rather than global: Excess is a *difference* of counts and goes
# negative for an oversmoothed method, which a log axis cannot show at all. Its
# reference line is y=0 (the truth's own reversal count) instead of the tv-ATE line,
# which is what `ref` selects.
# An axis label names the PER-REPLICATION scalar and nothing more. Each panel then
# takes the mean (top row) or the variance (bottom row) of that scalar ACROSS
# REPLICATIONS, which the figure caption states once instead of every label carrying
# it -- the convention the paper's prose already uses ("the mean iMSE over N
# replications", simulation_study.tex).
#
# Whether a label needs the word "mean" therefore depends on whether the metric's own
# name already contains one, and only two of the four do:
#
#   iMSE     the M *is* the average over the n_test test subjects -- that is how the
#            paper defines it (main.tex:501, the 1/n_test sum is inside iMSE(tau_hat)).
#            "Average iMSE" said it twice.
#   D, H1    an integral over t and an H^1 norm are per-SUBJECT quantities; neither
#            name contains an average, so both carry "Mean" explicitly.
#   Excess   a per-subject reversal count, likewise -- "Average" here plays the same
#            role "Mean" plays above.
#
# Every metric averages over a test set that is FIXED across methods, sample sizes and
# replications (_make_test_set is called once per DGP, outside the seed loop), so these
# are fixed-design averages, not fresh draws from P_X, and the variance panels are
# training-sample variability with no test-draw noise mixed in.
METRIC_SPECS = {
    "imse": dict(
        key="imse", stem="simulation_convergence",
        mean_label="iMSE", var_label="Variance of iMSE",
        log_y=True, ref="pop_avg",
    ),
    "dmse": dict(
        key="dmse", stem="simulation_convergence_dmse",
        mean_label=r"Mean $\int_0^{t_{\mathrm{end}}}"
                   r"(\partial_t\hat\tau - \partial_t\tau)^2\,\mathrm{d}t$",
        var_label=r"Variance of mean $\int_0^{t_{\mathrm{end}}}"
                  r"(\partial_t\hat\tau - \partial_t\tau)^2\,\mathrm{d}t$",
        log_y=True, ref="pop_avg",
    ),
    "h1": dict(
        key="h1", stem="simulation_convergence_h1",
        mean_label=r"Mean $\|\hat\tau - \tau\|_{H^1}$",
        var_label=r"Variance of mean $\|\hat\tau - \tau\|_{H^1}$",
        log_y=True, ref="pop_avg",
    ),
    "excess": dict(
        key="excess", stem="simulation_convergence_excess",
        mean_label="Average excess reversal count",
        var_label="Variance of excess reversal count",
        log_y=False, ref="zero",
        # The only metric that goes negative, and autoscale handles it badly: it ends
        # the axis just under the lowest curve, and the default locator anchors ticks
        # wherever it likes, so with curves spanning -0.6 to 15 the lowest tick lands
        # on 0. Every method that beats the truth's own reversal count -- AIR, at about
        # -0.5, which is the GOOD direction -- then sits in an unlabelled sliver with
        # nothing to read it against. Drop the floor to -2 and tick every 2 so the
        # negative region is labelled. min(), so a DGP running below -2 is not clipped;
        # the step is set explicitly because moving the limit alone does not move the
        # locator's anchor, and -2 would still go unlabelled.
        mean_ymin_at_most=-2.0, mean_ytick_step=2.0,
    ),
}

# The reversal thresholds the results files actually carry, in increasing order.
#
# Both entries are the SAME metric -- mean over test subjects of
# N(tau_hat) - N(tau_true) -- computed under a different rule for what counts as a
# direction reversal rather than against a different target, which is why they belong
# side by side on a shared y axis instead of in two unrelated figures.
#
# There are two and only two, because `curve_metrics` calls `n_sign_changes` exactly
# twice: once with no threshold, once at DEFAULT_THRESHOLD_FRAC. A third threshold is
# not a re-plot -- the result files keep these scalars, not the fitted curves, so it
# would mean re-running the simulation.
#
# `title` is what the column header says. Keeping the 5% out of the string and reading
# it from DEFAULT_THRESHOLD_FRAC means the header cannot drift from the constant the
# numbers underneath it were computed with.
EXCESS_THRESHOLD_VARIANTS = [
    dict(key="excess", title="No flatness threshold (any sign change)"),
    dict(key="excess_thresh",
         title=f"Flat below {DEFAULT_THRESHOLD_FRAC:.0%} of the curve's own range"),
]


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

    if rsf == "CSF":
        # One grf forest per horizon: the causal survival forest targets a scalar
        # effect at a single time, so the tv-CATE curve costs M fits. Hyperparameters
        # stay at grf's defaults.
        return CausalSurvivalForest(
            M=cfg["M"],
            num_trees=cfg.get("csf_num_trees", 2000),
            num_threads=cfg.get("csf_num_threads", 1),
            rscript=cfg.get("csf_rscript", "Rscript"),
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
        # 0 unless --cate-val-frac is passed, i.e. no split and `patience` stays inert.
        cate_val_frac=cfg.get("cate_val_frac", 0.0),
        # Off unless --cate-refit-full is passed. Requires cate_val_frac > 0: the
        # split picks the stopping point, then the final model is refit on all n rows
        # for the same number of optimizer steps. Not a config default until measured.
        cate_refit_full=cfg.get("cate_refit_full", False),
    )
    if nuis_hidden:
        common["weibull_config"] = {"hidden_sizes": nuis_hidden}
        common["propensity_config"] = {"hidden_sizes": nuis_hidden}
    # Off by default. Drops the *censoring* model's hidden layers, so its Weibull
    # shape and scale are each a softplus of a single linear index in x -- 32
    # weights on IST-3's p=15, against 3170 for the (64, 32) MLP. The event
    # models, the propensity and the CATE model are untouched.
    #
    # Why this and nothing else. The AIPCW pseudo-outcome divides by S_cens in all
    # three of its terms, so it is a near-cancellation: a smooth systematic error
    # in S_cens largely cancels, while an erratic per-subject error does not and
    # lands undamped in the tail of Y_tilde, where the quadratic R-loss is most
    # sensitive to it. Restricting the map from x is what removes the erratic part.
    # Bias in S_cens is cheap; variance is not -- the restricted model is 4x more
    # biased (-0.129 vs -0.033) and still has LOWER rmse (0.139 vs 0.168), and it
    # cuts max|Y_tilde| from 136-190 to 5.6-7.7 and r_loss_at_zero from 17-26 to
    # ~1.5 against an oracle 1.0.
    #
    # Two other respecifications were tried and are NOT applied, because the
    # ablation in examples/diagnose_ist3_nuisance.py measured them at zero:
    # `admin_as_censored` (0.1102 with, 0.1128 without) and `shared_shape` (0.1133
    # with, 0.1102 without). Both are still available on `AIREstimator` -- the
    # identification argument for the first is in `cross_fit_nuisances` and is
    # sound; it simply does not pay on this DGP. Turning either on WITHOUT this
    # flexibility restriction is actively harmful (0.3298 vs a 0.1830 baseline).
    if cfg.get("censoring_fix", False):
        common["censoring_config"] = {"hidden_sizes": cfg.get("censoring_hidden", ())}
    if tensor:
        common["cate_model_class"] = CATETensorProduct
        common["cate_model_kwargs"] = dict(
            n_time_basis=cfg["n_time_basis"],
            poly_degree=cfg["poly_degree"],
        )
    else:
        common["K"] = cfg.get("text_K", cfg["K"]) if is_text_dgp else cfg["K"]
        common["encoder_hidden_sizes"] = encoder_hidden
        # Off by default. The archived 100-seed run normalized t unconditionally;
        # this key exists so that behaviour can be A/B'd against the current one
        # without a code edit. See `normalize_time` in CATEMixture.
        common["cate_model_kwargs"] = dict(
            normalize_time=cfg.get("mgd_normalize_time", False),
            # Off by default. The archived run drew lambda_raw from N(0, 1); the
            # current default N(1.0, 0.5) has 2.4x less relative spread in the
            # resulting decay rates. See `legacy_lambda_init` in CATEMixture.
            legacy_lambda_init=cfg.get("mgd_legacy_lambda_init", False),
        )
    return AIREstimator(**common)


def run_single(dgp_class, cfg, n, seed, X_test, oracle, tensor, rsf, propensity_oracle=False):
    """Fit one method on one (n, seed) draw and return detailed results dict.

    Args:
        propensity_oracle: If True and oracle=False, fit with cross-fitted event/censoring
            survival but oracle propensity (e_hat = 0.5 for all). Useful for isolating
            whether propensity estimation is the bottleneck.
    """
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
    if cfg.get("legacy_seeding", False):
        # Diagnostic only (--legacy-seeding): exactly what the archived 100-seed run
        # did. Seeding numpy alone leaves the Keras generator advancing on its own,
        # which is the order-dependence described above -- but it is also the only
        # remaining difference between this harness and the archive on the
        # cross-fitted path, whose nuisance code is byte-identical and whose config
        # matches. Running both arms over the same seeds settles whether the seeding
        # accounts for the gap.
        np.random.seed(seed)
    else:
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
    time_grid, _ = build_time_grid(t_end, cfg["M"])

    if oracle:
        # Oracle needs original X (DGP formulas use stored pool indices)
        nuisances = build_oracle_nuisances(dgp_class, X_train_raw, Z, time_grid)
    elif propensity_oracle:
        # Cross-fitted event/censoring survival, but oracle propensity (0.5).
        #
        # Superseded: on IST-3 at n=20000 the estimated propensity is already
        # accurate (rmse 0.046 against truth, nothing clipped), and splicing the
        # oracle in moves iMSE from 0.242 to 0.258 -- i.e. very slightly worse,
        # within seed noise. The censoring survival is where the gap lives; see
        # examples/diagnose_ist3_nuisance.py. Kept only so the measurement is
        # reproducible.
        from air_cate._types import SurvivalData
        from air_cate.nuisance.cross_fit import cross_fit_nuisances
        data = SurvivalData(X=X_train, W=W, Z=Z, Delta=Delta)
        nuisances_cf = cross_fit_nuisances(
            data, time_grid,
            n_folds=cfg.get("n_folds", 5),
            censoring_config=est.censoring_config,
            admin_as_censored=est.admin_as_censored,
            t_end=t_end,
            rng_seed=seed,
        )
        nuisances = nuisances_cf._replace(
            e_hat=np.full(len(X_train), 0.5, dtype=np.float32)
        )

    est.fit(X_train, W, Z, Delta, t_end, nuisances=nuisances, verbose=0)

    tau_hat = est.predict_tau(X_test_est)
    t_grid = np.array(est.time_grid_)
    tau_true = dgp_class.true_tau(t_grid, X_test)  # uses stored pool indices
    delta = est.delta_

    # iMSE and mse_per_timepoint keep their exact previous definitions, so every
    # result file already on disk stays comparable cell for cell; the derivative,
    # H^1 and extrema metrics come from the same call. See air_cate.metrics.
    metrics = curve_metrics(tau_hat, tau_true, delta)
    runtime = time.time() - t_start

    out = {
        "seed": seed,
        **metrics,
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

    # CSF-only diagnostics, guarded the same way. Both are properties of the method
    # rather than of its accuracy, so they belong in the record: n_failed_horizons
    # counts grid points grf refused (horizon past the last observed failure time),
    # which the flat-tail fallback then papers over, and frac_out_of_range measures
    # how far an orthogonalized estimate strays outside [-1, 1] -- the structural
    # validity AIR has by construction and this baseline does not.
    n_failed = getattr(est, "n_failed_horizons_", None)
    if n_failed is not None:
        out["n_failed_horizons"] = int(n_failed)
    frac_oor = getattr(est, "frac_out_of_range_", None)
    if frac_oor is not None:
        out["frac_out_of_range"] = float(frac_oor)

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
            # Carry over any rung the prior file holds that this invocation was not
            # asked to run. `results` starts empty and only ever gains the requested
            # n_values, so without this a pass narrowed with --n-values REPLACES the
            # output file rather than adding to it -- save_results writes exactly the
            # rungs in `results`. That silently destroyed the CSF n<=10000 cells of the
            # metrics50 run, whose two-pass recipe ran the top rung alone as pass 2 and
            # believed resume would preserve the rest. Resume preserves a CELL it is
            # asked about; it never sees a rung outside n_values.
            if resume:
                for prior_n, prior_rung in (prior.get(dgp_name, {})
                                            .get(method_name, {}).items()):
                    if prior_n not in results[dgp_name][method_name]:
                        results[dgp_name][method_name][prior_n] = prior_rung
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
                        # Every metric key gets a nan, not just imse: the plotting
                        # code reads whichever metric it was asked for, and a missing
                        # key would raise only after the whole batch had been paid
                        # for. METRIC_KEYS is the single source of that list.
                        result = {
                            "seed": seed,
                            **{k: float("nan") for k in METRIC_KEYS},
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
    # A --output under a directory that does not exist yet is otherwise only
    # discovered here, after every fit has already been paid for.
    out_dir = os.path.dirname(results_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
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

def _pop_avg_metric(dgp_class, X_test, M, metric="imse"):
    """The named metric for the population-average CATE (no heterogeneity baseline).

    tau_bar(t) = mean_i tau_true(t, X_i) is broadcast as if it were an estimate and
    run through the same `curve_metrics` as any method, so every metric gets its
    reference line from one code path rather than a per-metric formula.
    """
    t_end = getattr(dgp_class, "t_end", 5.0)
    time_grid, delta = build_time_grid(t_end, M)
    t_grid = np.asarray(time_grid)
    tau_true = dgp_class.true_tau(t_grid, X_test)              # (n_test, M+1)
    tau_bar = np.broadcast_to(
        tau_true.mean(axis=0, keepdims=True), tau_true.shape
    )                                                          # (n_test, M+1)
    return curve_metrics(tau_bar, tau_true, delta)[metric]


def plot_convergence(results, n_values, cfg, X_tests, figures_dir=None,
                     metric="imse"):
    """Multi-panel convergence plot: metric mean (top) and variance (bottom) per DGP.

    `metric` names an entry in METRIC_SPECS. figures_dir defaults to FIGURES_DIR,
    which holds the tracked publication figures -- pass somewhere else for smoke runs
    so they are not overwritten.
    """
    if metric not in METRIC_SPECS:
        raise ValueError(
            f"unknown metric {metric!r}; valid: {sorted(METRIC_SPECS)}"
        )
    spec = METRIC_SPECS[metric]
    metric_key = spec["key"]
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

    # One column per DGP PRESENT IN THE RESULTS, in the canonical DGPS order -- not one
    # per entry of DGPS. A smoke run over two DGPs used to get a blank third column.
    dgps_present = [(n, c) for n, c in DGPS if n in results]
    if not dgps_present:
        raise ValueError(
            f"no DGP in the results matches DGPS; results hold {sorted(results)}, "
            f"expected some of {[n for n, _ in DGPS]}"
        )

    n_dgps = len(dgps_present)
    fig, axes = plt.subplots(2, n_dgps, figsize=(6.5 * n_dgps, 9),
                             sharex="col")
    if n_dgps == 1:
        axes = axes[:, None]  # ensure 2D
    ax_mean = axes[0]   # top row: metric mean
    ax_var = axes[1]     # bottom row: metric variance

    for col, (dgp_name, dgp_class) in enumerate(dgps_present):
        dgp_results = results[dgp_name]
        X_test = X_tests[dgp_name] if isinstance(X_tests, dict) else X_tests

        # --- Reference line (top row only) ---
        if spec["ref"] == "pop_avg":
            pa_value = _pop_avg_metric(dgp_class, X_test, cfg["M"], metric_key)
            ax_mean[col].axhline(pa_value, color="#888888", ls="--", linewidth=1.2,
                                 zorder=1, label="True tv-ATE")
        else:
            # Excess is a difference against the truth's own reversal count, so the
            # meaningful reference is exact agreement, not a competing estimator.
            ax_mean[col].axhline(0.0, color="#888888", ls="--", linewidth=1.2,
                                 zorder=1, label="True tv-CATE")

        # --- Method curves ---
        for method_name, _ in METHODS:
            if method_name not in dgp_results:
                continue
            vals = dgp_results[method_name]
            st = METHOD_STYLES[method_name]
            label = METHOD_LABELS[method_name]

            # Get n_values that exist in results for this method (skip all-NaN).
            #
            # .get, not [...]: result files written before these metrics existed
            # (full999_merged.json) carry only `imse`, and merging new results into a
            # copy of one must degrade to an empty curve rather than a KeyError.
            def _metric_vals(n):
                return [v.get(metric_key, float("nan")) for v in vals[n]]

            ns = [n for n in n_values if n in vals
                  and len(vals[n]) > 0
                  and np.any(np.isfinite(_metric_vals(n)))]
            if not ns:
                continue

            # --- Top row: metric mean with 95% CI (t-distribution) ---
            def _finite(arr):
                a = np.asarray(arr, dtype=float)
                return a[np.isfinite(a)]

            means = np.array([np.mean(_finite(_metric_vals(n))) for n in ns])
            ses = np.array([np.std(_finite(_metric_vals(n)), ddof=1)
                            / np.sqrt(len(_finite(_metric_vals(n))))
                            for n in ns])
            t_crit = np.array([t_dist.ppf(0.975, max(len(_finite(_metric_vals(n))) - 1, 1))
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

            # --- Bottom row: variance of the metric with 95% CI (chi-squared) ---
            variances = np.array([np.var(_finite(_metric_vals(n)), ddof=1) for n in ns])
            var_lo = np.array([
                (len(_finite(_metric_vals(n))) - 1) * np.var(_finite(_metric_vals(n)), ddof=1)
                / chi2_dist.ppf(0.975, max(len(_finite(_metric_vals(n))) - 1, 1))
                for n in ns
            ])
            var_hi = np.array([
                (len(_finite(_metric_vals(n))) - 1) * np.var(_finite(_metric_vals(n)), ddof=1)
                / chi2_dist.ppf(0.025, max(len(_finite(_metric_vals(n))) - 1, 1))
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
            row_axes[col].set_xticks(n_values)
            row_axes[col].set_xticklabels([str(n) for n in n_values], fontsize=12)
            row_axes[col].tick_params(axis="y", labelsize=12)
            row_axes[col].grid(True, alpha=0.12, linewidth=0.4)

        # The variance is a positive quantity for every metric, so its row stays log
        # even when the mean row cannot be (Excess goes negative).
        if spec["log_y"]:
            ax_mean[col].set_yscale("log")
        ax_var[col].set_yscale("log")

        # Optional floor and tick step for the mean row, used only by the one metric
        # whose panel autoscales badly (see the Excess entry in METRIC_SPECS). Runs
        # here, after every curve is drawn, because it reads the autoscaled limit.
        floor = spec.get("mean_ymin_at_most")
        if floor is not None:
            lo, hi = ax_mean[col].get_ylim()
            ax_mean[col].set_ylim(min(lo, floor), hi)
        step = spec.get("mean_ytick_step")
        if step is not None:
            ax_mean[col].yaxis.set_major_locator(MultipleLocator(step))

        ax_mean[col].set_title(f"{dgp_name} DGP", fontsize=14,
                                fontweight="bold", pad=8)
        ax_var[col].set_xlabel("Sample size", fontsize=15)

    # Labels come from METRIC_SPECS, where the naming convention is argued. Only the
    # leftmost column carries one: the panels in a row are the same quantity on
    # different DGPs, and repeating the label three times would cost the width the
    # curves need.
    ax_mean[0].set_ylabel(spec["mean_label"], fontsize=15)
    ax_var[0].set_ylabel(spec["var_label"], fontsize=15)

    # Shared legend below all panels.
    #
    # Ordered BY LABEL, not by position. A hardcoded index permutation
    # (order = [0, 1, 2, 5, 6, 3, 4]) worked only when all six methods were
    # plotted: any --methods subset yields fewer handles and index 5 raises
    # IndexError -- after the fits are done and the summary is printed, so the
    # expensive work is already complete when the plot dies.
    handles, labels = ax_mean[0].get_legend_handles_labels()
    # The reference line first, then the solid (cross-fitted) methods, then the dashed
    # oracle ones, so the legend reads in the same visual grouping as the panels.
    preferred = ["True tv-ATE", "True tv-CATE"] + [
        METHOD_LABELS[m]
        for m in ("AIR", "Tensor-prod", "RSF-T", "RSF-S", "CSF",
                  "AIR-oracle", "TP-oracle")
    ]
    rank = {label: i for i, label in enumerate(preferred)}
    # Unknown labels sort last in their existing order rather than being dropped:
    # a legend entry silently missing from a figure is worse than one out of place.
    order = sorted(range(len(labels)), key=lambda i: (rank.get(labels[i], len(rank)), i))
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=14, frameon=False, bbox_to_anchor=(0.5, -0.02),
               columnspacing=2.5, handletextpad=0.8, handlelength=2.8,
               labelspacing=0.8)

    fig.tight_layout(rect=[0, 0.06, 1, 1], w_pad=2.5, h_pad=1.5)
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{spec['stem']}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def _mean_ci_series(vals, n_values, metric_key):
    """(ns, means, halfwidths) for one method's `metric_key` across sample sizes.

    Exactly the top-row computation `plot_convergence` does inline: drop sample sizes
    with no finite replication, then a normal-theory 95% CI on the mean with the
    t critical value at (n_seeds - 1) degrees of freedom.

    `.get`, not `[...]`: result files written before these metrics existed carry only
    `imse`, and merging new results into a copy of one must degrade to an empty curve
    rather than a KeyError.
    """
    def _finite(n):
        a = np.array([v.get(metric_key, float("nan")) for v in vals[n]], dtype=float)
        return a[np.isfinite(a)]

    ns = [n for n in n_values
          if n in vals and len(vals[n]) > 0 and len(_finite(n)) > 0]
    if not ns:
        return [], np.array([]), np.array([])

    means = np.array([_finite(n).mean() for n in ns])
    ses = np.array([_finite(n).std(ddof=1) / np.sqrt(len(_finite(n))) for n in ns])
    t_crit = np.array([t_dist.ppf(0.975, max(len(_finite(n)) - 1, 1)) for n in ns])
    return ns, means, t_crit * ses


def _ordered_legend_handles(ax):
    """Legend handles/labels for `ax`, in the paper's reading order.

    Reference line first, then the solid cross-fitted methods, then the dashed oracle
    ones, so the legend groups the way the panels do. Ordered BY LABEL rather than by a
    hardcoded index permutation, which breaks under any --methods subset. Unknown labels
    sort last in their existing order rather than being dropped: a legend entry silently
    missing from a figure is worse than one out of place.
    """
    handles, labels = ax.get_legend_handles_labels()
    preferred = ["True tv-ATE", "True tv-CATE"] + [
        METHOD_LABELS[m]
        for m in ("AIR", "Tensor-prod", "RSF-T", "RSF-S", "CSF",
                  "AIR-oracle", "TP-oracle")
    ]
    rank = {label: i for i, label in enumerate(preferred)}
    order = sorted(range(len(labels)), key=lambda i: (rank.get(labels[i], len(rank)), i))
    return [handles[i] for i in order], [labels[i] for i in order]


def plot_excess_thresholds(results, n_values, figures_dir=None,
                           stem="simulation_convergence_excess_thresholds"):
    """Average excess reversal count under every recorded threshold, in one figure.

    One row per DGP, one COLUMN per entry of EXCESS_THRESHOLD_VARIANTS, so the two
    thresholds sit next to each other left to right and the left-to-right change within
    a row is the flatness rule and nothing else.

    Only the mean panel, deliberately: `simulation_convergence_excess` already carries
    the variance row at the default threshold, and what this figure exists to answer is
    whether a method's excess count survives being asked to ignore small wiggles.

    The y axis is shared ACROSS each row. Autoscaling the two columns independently
    would rescale away exactly the effect being shown -- a method whose count falls from
    13 to 6 under thresholding would draw two identical-looking panels.

    Unlike `plot_convergence`, this takes no X_tests: every panel's reference line is
    y = 0, so no population-average curve has to be evaluated. Note that y = 0 is the
    right reference in BOTH columns even though the truth's own reversal count differs
    between them -- each column's excess is a difference against the truth counted under
    that same column's threshold (metrics.py:171-177).
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

    # One row per DGP PRESENT IN THE RESULTS, in the canonical DGPS order -- not one per
    # entry of DGPS, so a two-DGP smoke run does not get a blank third row.
    dgps_present = [n for n, _ in DGPS if n in results]
    if not dgps_present:
        raise ValueError(
            f"no DGP in the results matches DGPS; results hold {sorted(results)}, "
            f"expected some of {[n for n, _ in DGPS]}"
        )

    n_rows = len(dgps_present)
    n_cols = len(EXCESS_THRESHOLD_VARIANTS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 4.3 * n_rows),
                             sharex="col", sharey="row", squeeze=False)

    for row, dgp_name in enumerate(dgps_present):
        dgp_results = results[dgp_name]

        for col, variant in enumerate(EXCESS_THRESHOLD_VARIANTS):
            ax = axes[row][col]
            ax.axhline(0.0, color="#888888", ls="--", linewidth=1.2, zorder=1,
                       label="True tv-CATE")

            for method_name, _ in METHODS:
                if method_name not in dgp_results:
                    continue
                ns, means, hw = _mean_ci_series(
                    dgp_results[method_name], n_values, variant["key"]
                )
                if not ns:
                    continue
                st = METHOD_STYLES[method_name]
                ax.plot(
                    ns, means,
                    marker=st["marker"], ls=st["ls"],
                    linewidth=st["lw"], markersize=st["ms"],
                    color=st["color"],
                    markeredgecolor="white", markeredgewidth=0.5,
                    label=METHOD_LABELS[method_name], zorder=st["zorder"],
                )
                ax.fill_between(ns, means - hw, means + hw,
                                color=st["color"], alpha=0.12, zorder=st["zorder"] - 1)

            ax.set_xscale("log")
            ax.set_xticks(n_values)
            ax.set_xticklabels([str(n) for n in n_values], fontsize=12)
            ax.tick_params(axis="y", labelsize=12)
            ax.grid(True, alpha=0.12, linewidth=0.4)

            # Column headers on the top row only: every panel below repeats the same
            # threshold, and the DGP is named by the row's y label.
            if row == 0:
                ax.set_title(variant["title"], fontsize=13, fontweight="bold", pad=10)
            if row == n_rows - 1:
                ax.set_xlabel("Sample size", fontsize=15)

        # Floor and tick step from the Excess spec, for the reason argued there: the
        # metric goes negative, and left alone the locator anchors its lowest tick on 0,
        # so every method that beats the truth's own count sits in an unlabelled sliver.
        # Applied to the left panel only -- sharey="row" carries it across the row.
        spec = METRIC_SPECS["excess"]
        lo, hi = axes[row][0].get_ylim()
        axes[row][0].set_ylim(min(lo, spec["mean_ymin_at_most"]), hi)
        axes[row][0].yaxis.set_major_locator(MultipleLocator(spec["mean_ytick_step"]))

        # The row label carries the DGP; the quantity itself is named once by supylabel,
        # since all six panels plot it.
        axes[row][0].set_ylabel(f"{dgp_name} DGP", fontsize=14, fontweight="bold")

    fig.supylabel("Average excess reversal count", fontsize=15)

    handles, labels = _ordered_legend_handles(axes[0][0])
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=14, frameon=False, bbox_to_anchor=(0.5, -0.02),
               columnspacing=2.5, handletextpad=0.8, handlelength=2.8,
               labelspacing=0.8)

    fig.tight_layout(rect=[0.02, 0.05, 1, 1], w_pad=2.5, h_pad=1.5)
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_all_metrics(results, n_values, cfg, X_tests, figures_dir=None, metrics=None):
    """Render one convergence figure per metric.

    The population-average reference line refits nothing, so looping here costs only
    the plotting itself -- there is no reason to make the caller choose a metric.
    """
    for metric in (metrics or list(METRIC_SPECS)):
        plot_convergence(results, n_values, cfg, X_tests,
                         figures_dir=figures_dir, metric=metric)


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
                        help="Comma-separated DGP names to run (e.g. 'Weibull,Mixture'). "
                             "Names contain no commas, so this is unambiguous.")
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
    # The two knobs that separate this package from the archived 100-seed run.
    # Both exist so that run can be reproduced from the command line, without
    # editing a config constant and risking it being committed by accident.
    parser.add_argument("--mgd-normalize-time", action="store_true",
                        help="Rescale t to [0,1] inside CATEMixture, as the archived "
                             "100-seed run did. Affects AIR and AIR-oracle only; the "
                             "tensor-product model always normalizes.")
    parser.add_argument("--n-restarts", type=int, default=None,
                        help="Independent CATE fits per cell, best training loss wins. "
                             "The archived 100-seed run used 5; the default is 1.")
    parser.add_argument("--mgd-legacy-lambda-init", action="store_true",
                        help="Draw CATEMixture's lambda_raw from N(0, 1), as the archived "
                             "100-seed run did, instead of the current N(1.0, 0.5). "
                             "Affects AIR and AIR-oracle only. The current default was "
                             "chosen against a constant initializer, not against this one, "
                             "and it starts the mixture components with 2.4x less relative "
                             "spread in their decay rates.")
    parser.add_argument("--mgd-k", type=int, default=None,
                        help="Number of MGD mixture components, overriding BOTH `K` and "
                             "`text_K` so it takes effect on the text DGP too (which reads "
                             "text_K, not K -- setting only one silently does nothing on "
                             "IST-3). Diagnostic: capacity is fixed at K=2 / text_K=4 while "
                             "n grows 40x across the ladder, and a saturating model is "
                             "indistinguishable from the plateau we are chasing. No effect "
                             "on the tensor-product or RSF methods.")
    parser.add_argument("--legacy-seeding", action="store_true",
                        help="Seed numpy only, as the archived 100-seed run did, instead "
                             "of also seeding the Keras global generator. Diagnostic: "
                             "isolates whether the seeding change explains the "
                             "cross-fitted IST-3 gap. Reintroduces the order-dependence "
                             "described at the top of run_single, so do not use it for "
                             "production runs.")
    parser.add_argument("--cate-epochs", type=int, default=None,
                        help="Epoch cap for the CATE stage (default 200). Diagnostic: now "
                             "that the t=0 NaN is fixed, nothing actually stops CATE "
                             "training -- patience monitors the *train* R-loss, which falls "
                             "monotonically, so every fit runs to the cap. Sweeping this "
                             "measures how much of the archived run's accuracy came from "
                             "its fits dying early rather than converging.")
    parser.add_argument("--cate-val-frac", type=float, default=None,
                        help="Hold out this fraction of rows from the CATE fit and early-stop "
                             "on their R-loss (default 0 = no split, train to the epoch cap). "
                             "This is the principled alternative to --cate-epochs: it makes "
                             "`patience` a real stopping rule instead of an inert one.")
    parser.add_argument("--cate-refit-full", action="store_true",
                        help="After the held-out split picks a stopping point, refit the "
                             "CATE model from scratch on ALL rows for the same number of "
                             "optimizer steps. Recovers the rows --cate-val-frac spends, "
                             "at the cost of a second CATE fit. Requires --cate-val-frac "
                             "> 0 (or a config that sets it).")
    parser.add_argument("--censoring-fix", action="store_true",
                        help="Fit the censoring nuisance with no hidden layers, so its "
                             "Weibull shape and scale are each a softplus of a single "
                             "linear index in x (32 weights on IST-3) instead of a "
                             "(64, 32) MLP (3170). On IST-3 at n=20000, 4 seeds, this "
                             "takes cross-fitted AIR from iMSE 0.183 to 0.110 against an "
                             "oracle-censoring 0.098 and RSF-T's 0.1239, by cutting the "
                             "pseudo-outcome noise floor ~15x. Affects the censoring model "
                             "only, so the oracle and RSF arms are unchanged and stay "
                             "valid as controls.")
    parser.add_argument("--metric", type=str, default=None,
                        help="Comma-separated metrics to plot (default: all of "
                             f"{','.join(METRIC_SPECS)}). Every metric is computed and "
                             "stored on every run regardless; this only selects figures.")
    parser.add_argument("--excess-thresholds", action="store_true",
                        help="With --plot-only: render ONLY the threshold-comparison "
                             "figure (average excess reversal count, one row per DGP, "
                             "one column per recorded threshold). Its reference line is "
                             "y=0, so unlike the per-metric figures it needs no test "
                             "set and runs without the gitignored IST-3 embedding "
                             "artifact.")
    parser.add_argument("--csf-rscript", type=str, default=None,
                        help="Path to the Rscript binary for the CSF baseline. Needed "
                             "wherever R is provided by a module rather than being on "
                             "the conda env's PATH (e.g. Sherlock).")
    parser.add_argument("--csf-threads", type=int, default=None,
                        help="Threads for grf's causal survival forest (default 1). "
                             "The CSF baseline refits one forest per horizon, so a "
                             "replication is ~M grf fits and raising this speeds a "
                             "local run up considerably. Runtimes recorded with more "
                             "than one thread are NOT comparable to the single-core "
                             "numbers in the results table.")
    args = parser.parse_args()

    # Validated before anything expensive: a typo here would otherwise surface only
    # after the fits, when the figures are written.
    if args.metric:
        plot_metrics = [m.strip() for m in args.metric.split(",") if m.strip()]
        unknown = [m for m in plot_metrics if m not in METRIC_SPECS]
        if unknown:
            parser.error(f"Unknown metric(s) {unknown}. Valid: {list(METRIC_SPECS)}")
    else:
        plot_metrics = None

    # --- Merge mode ---
    if args.merge:
        merge_path = args.output or RESULTS_PATH
        merge_partial_results(merge_path, args.merge)
        return

    # --- Plot-only mode ---
    if args.plot_only:
        print("Loading saved results...")
        results, n_values, cfg = load_results(args.output or RESULTS_PATH)
        # Handled before the test sets are built, which is the point of the flag: this
        # figure needs none, so it renders on a machine with no IST-3 embedding artifact.
        if args.excess_thresholds:
            print_summary(results, n_values)
            print("\nGenerating plots...")
            plot_excess_thresholds(results, n_values, figures_dir=args.figures_dir)
            print(f"Figure saved to {args.figures_dir or FIGURES_DIR}/")
            return
        # Recreate per-DGP test sets, for the DGPs the file actually holds. Building
        # all of DGPS unconditionally made re-plotting a two-DGP smoke run impossible
        # on any machine without the IST-3 embedding artifact, which is gitignored.
        X_tests = {}
        for dgp_name, dgp_class in DGPS:
            if dgp_name not in results:
                continue
            X_tests[dgp_name] = _make_test_set(dgp_class, cfg["n_test"])
        print_summary(results, n_values)
        print("\nGenerating plots...")
        plot_all_metrics(results, n_values, cfg, X_tests,
                         figures_dir=args.figures_dir, metrics=plot_metrics)
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

    # Applied after --n-values so both overrides can be combined. Each takes its own
    # shallow copy, which is cheap and keeps the module constants immutable whichever
    # subset of flags is passed.
    if args.mgd_normalize_time:
        cfg = dict(cfg)
        cfg["mgd_normalize_time"] = True
    if args.n_restarts is not None:
        if args.n_restarts < 1:
            parser.error(f"--n-restarts must be >= 1, got {args.n_restarts}")
        cfg = dict(cfg)
        cfg["n_restarts"] = args.n_restarts
    if args.mgd_legacy_lambda_init:
        cfg = dict(cfg)
        cfg["mgd_legacy_lambda_init"] = True
    if args.mgd_k is not None:
        if args.mgd_k < 1:
            parser.error(f"--mgd-k must be >= 1, got {args.mgd_k}")
        cfg = dict(cfg)
        # Both keys, deliberately: make_estimator reads text_K for the text DGP and
        # falls back to K otherwise, so setting one alone is a silent no-op on
        # whichever DGP the run actually targets.
        cfg["K"] = args.mgd_k
        cfg["text_K"] = args.mgd_k
    if args.legacy_seeding:
        cfg = dict(cfg)
        cfg["legacy_seeding"] = True
    if args.cate_epochs is not None:
        if args.cate_epochs < 1:
            parser.error(f"--cate-epochs must be >= 1, got {args.cate_epochs}")
        cfg = dict(cfg)
        cfg["cate_epochs"] = args.cate_epochs
    if args.cate_val_frac is not None:
        if not 0.0 <= args.cate_val_frac < 1.0:
            parser.error(
                f"--cate-val-frac must be in [0, 1), got {args.cate_val_frac}"
            )
        cfg = dict(cfg)
        cfg["cate_val_frac"] = args.cate_val_frac
    if args.cate_refit_full:
        # Checked here, against the *effective* config, rather than left to
        # AIREstimator's ctor: without a split there is no honest stopping point to
        # refit at, and a run that would die on fit #1 should die at argument-parse
        # time instead. Reads cfg, so it must follow the --cate-val-frac block.
        if cfg.get("cate_val_frac", 0.0) <= 0.0:
            parser.error(
                "--cate-refit-full needs a held-out split to select the stopping "
                "point, but the effective cate_val_frac is "
                f"{cfg.get('cate_val_frac', 0.0)}. Pass --cate-val-frac 0.2."
            )
        cfg = dict(cfg)
        cfg["cate_refit_full"] = True
    if args.censoring_fix:
        cfg = dict(cfg)
        cfg["censoring_fix"] = True
    if args.csf_threads is not None:
        if args.csf_threads < 1:
            parser.error(f"--csf-threads must be >= 1, got {args.csf_threads}")
        cfg = dict(cfg)
        cfg["csf_num_threads"] = args.csf_threads
    if args.csf_rscript:
        cfg = dict(cfg)
        cfg["csf_rscript"] = args.csf_rscript

    # --- DGP / method filters ---
    dgps = DGPS
    if args.dgp:
        # Comma-separated, matching --methods. Unknown names are named individually
        # rather than reported as "no matches", which is the difference between a
        # one-character typo and a wrong flag.
        dgp_names = [d.strip() for d in args.dgp.split(",") if d.strip()]
        valid = [n for n, _ in DGPS]
        unknown = [d for d in dgp_names if d not in valid]
        if unknown:
            parser.error(f"Unknown DGP(s) {unknown}. Valid: {valid}")
        dgps = [(n, c) for n, c in DGPS if n in dgp_names]

    methods = METHODS
    if args.methods:
        method_names = [m.strip() for m in args.methods.split(",")]
        methods = [(n, o) for n, o in METHODS if n in method_names]
        if not methods:
            valid = [n for n, _ in METHODS]
            parser.error(f"No matching methods. Valid: {valid}")

    # Fail fast on a missing grf. CSF is unconditionally in METHODS, so without this
    # check a run on a machine with no grf install fills every CSF cell with nan and
    # only says so in the summary -- after the whole batch has been paid for.
    if any(name == "CSF" for name, _ in methods):
        rscript = cfg.get("csf_rscript", "Rscript")
        if not grf_available(rscript=rscript):
            raise SystemExit(
                f"The CSF baseline needs {rscript!r} on PATH with the grf package "
                "installed, and one of the two is missing.\n"
                "Install it with:  Rscript -e 'install.packages(\"grf\")'\n"
                "Point at a specific R with --csf-rscript /path/to/Rscript, or drop "
                "CSF with e.g. --methods AIR,Tensor-prod,AIR-oracle,TP-oracle,RSF-T,RSF-S"
            )

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
    print("Convergence Simulation: trajectory metrics vs Sample Size")
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
        plot_all_metrics(results, n_values, cfg, X_tests,
                         figures_dir=args.figures_dir, metrics=plot_metrics)
        print(f"All figures saved to {args.figures_dir or FIGURES_DIR}/")


if __name__ == "__main__":
    main()
