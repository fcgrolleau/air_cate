"""Attribute the IST-3 cross-fitted AIR gap to individual nuisance components.

Motivation. On IST-3 the oracle arm is close to the archived 100-seed run
(0.0825-0.0873 at n=20000 against 0.0664) while the cross-fitted arm is nearly
twice that (0.164 mean / 0.149 median against 0.0854), and the overnight K sweep
showed the cross-fitted number is flat in capacity: 0.1714 / 0.1660 / 0.1649 /
0.1642 at K = 2 / 4 / 8 / 16. The whole defect therefore lives between the
observed data and the pseudo-outcomes, not in the CATE model. Cross-fitted iMSE
also *rises* from n=500 to n=5000 (0.31 -> 0.39) while the oracle arm falls
monotonically, which is the signature of a variance/tail problem rather than a
capacity one.

This script measures, on the same data draw, both nuisance sets side by side:

1. Component accuracy of each estimated nuisance against its true counterpart
   (IST-3 exposes exact per-subject survival, so the comparison is exact).
2. Censoring composition -- what share of Delta=0 is *administrative*, i.e.
   Z = t_end exactly. The censoring variable is min(C, t_end), so those rows are
   an atom at t_end that a Weibull AFT cannot represent; the oracle instead uses
   exp(-t / lambda_c(x)), the law of C alone. If that misspecification is the
   problem, it appears here as a bias in S_cens that does not shrink with n.
3. The fitted Weibull shape distribution, including the share pinned at the 0.3
   floor in `WeibullAFT._get_params`.
4. Pseudo-outcome tails under each nuisance set -- the R-loss is a squared loss,
   so a handful of rows with large |Y_tilde| dominates the gradient.
5. A paired ablation that swaps nuisance components in one at a time and refits
   the CATE stage, giving a direct iMSE attribution.

Ablation arms (all share one data draw, one estimated nuisance set, and one CATE
initialization per seed, so differences are the swapped component alone):

    est_all       what "AIR" does today
    orc_e         oracle propensity (exactly 0.5) only
    orc_Sev       oracle event survival only
    orc_Sce       oracle censoring survival only
    orc_Sev_Sce   both survivals oracle, propensity still estimated
    fix_Sce       estimated censoring, but fit treating Z = t_end as censored
                  rather than as an observed censoring event
    sh_Sce        estimated censoring with a single shared Weibull shape (the
                  canonical AFT specification) instead of a per-subject k(x)
    shfix_Sce     both censoring corrections together
    lin_Sce       shfix_Sce with log-lambda linear in x rather than an MLP, i.e.
                  the correctly specified censoring model for this DGP
    orc_all       what "AIR-oracle" does today

Usage:
  uv run python examples/diagnose_ist3_nuisance.py --n 5000 --seeds 6 \
      --output /tmp/ist3_diag_n5000.json
  uv run python examples/diagnose_ist3_nuisance.py --n 1000 --seeds 1 \
      --arms est_all,orc_all --output /tmp/smoke.json     # fast smoke test
"""

import argparse
import gc
import json
import os
import sys
import time

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax
import jax.numpy as jnp

from air_cate import AIREstimator  # sets KERAS_BACKEND=jax

import keras
from air_cate._types import NuisanceEstimates, SurvivalData
from air_cate.dgp import IST3TextDGP
from air_cate.nuisance.cross_fit import cross_fit_nuisances
from air_cate.nuisance.oracle import build_oracle_nuisances
from air_cate.nuisance.weibull_aft import WeibullAFT
from air_cate.pseudo_outcome import compute_pseudo_outcomes
from air_cate.utils import build_time_grid
from sklearn.model_selection import KFold

# Mirrors LARGE_CONFIG in plot_simulation.py. Kept as a literal rather than
# imported so this file does not drag in the plotting stack, but it must track
# it: an ablation run against a different CATE config would not be comparable
# to the gate numbers it is trying to explain.
CFG = dict(
    M=30,
    n_folds=3,
    eps=0.01,
    nuisance_epochs=200,
    nuisance_hidden=(64, 32),
    cate_epochs=200,
    cate_lr=1e-3,
    cate_patience=30,
    cate_val_frac=0.2,
    n_restarts=1,
    text_K=4,
    text_encoder_hidden=(16,),
    mgd_normalize_time=True,
)

# rng_seed default inside cross_fit_nuisances, which AIREstimator.fit does not
# override. The censoring-only refit below must reuse it or its folds would not
# line up with the ones the estimated nuisances came from.
CROSS_FIT_RNG_SEED = 42

ARMS = [
    "est_all",
    "orc_e",
    "orc_Sev",
    "orc_Sce",
    "orc_Sev_Sce",
    "fix_Sce",
    "sh_Sce",
    "shfix_Sce",
    "lin_Sce",
    "linonly_Sce",
    "linfree_Sce",
    "linplain_Sce",
    "orc_all",
]

# Censoring-only refits: arm name -> how to fit the censoring model.
CENS_REFITS = {
    "fix_Sce": dict(admin_as_censored=True, shared_shape=False),
    "sh_Sce": dict(admin_as_censored=False, shared_shape=True),
    "shfix_Sce": dict(admin_as_censored=True, shared_shape=True),
    # Same as shfix_Sce but with no hidden layers, so lambda is a function of a
    # single linear index in x rather than of an MLP.
    #
    # This is NOT the correctly specified model, despite the DGP's true scale
    # being lambda_c(x) = lambda_c_base * exp(-b_lambda * z_prog) with z_prog a
    # linear index: WeibullAFT._get_params reads lambda off the head with
    # softplus, not exp, so hidden_sizes=() gives softplus(linear), a different
    # family. Its measured bias is in fact the worst of any arm. What the
    # contrast isolates is the cost of surplus *flexibility*, not of
    # misspecification.
    "lin_Sce": dict(admin_as_censored=True, shared_shape=True, hidden_sizes=()),
    # Ablation of lin_Sce's three factors. lin_Sce turns on all of
    # admin_as_censored, shared_shape and hidden_sizes=() at once, and shfix_Sce
    # shows the first two alone are actively harmful, so these two arms drop one
    # factor each to find which are load-bearing.
    "linonly_Sce": dict(admin_as_censored=False, shared_shape=True, hidden_sizes=()),
    "linfree_Sce": dict(admin_as_censored=True, shared_shape=False, hidden_sizes=()),
    # The minimal one-knob change from the baseline: drop the censoring model's
    # hidden layers and nothing else. Measured at 0.110-0.113 alongside the three
    # arms above, none of which separate, so this is the spec to prefer -- it is
    # the smallest edit that gets the effect, and it leaves both `shared_shape`
    # and `admin_as_censored` off, neither of which pays for itself here.
    "linplain_Sce": dict(admin_as_censored=False, shared_shape=False, hidden_sizes=()),
}


# ---------------------------------------------------------------------------
# Nuisance surgery
# ---------------------------------------------------------------------------

def splice(est, orc, *, e=False, S_event=False, S_cens=False):
    """Return `est` with the requested component families taken from `orc`."""
    fields = {}
    if e:
        fields["e_hat"] = orc.e_hat
    if S_event:
        for f in ("S_event_0_grid", "S_event_1_grid",
                  "S_event_0_at_Z", "S_event_1_at_Z"):
            fields[f] = getattr(orc, f)
    if S_cens:
        for f in ("S_cens_0_grid", "S_cens_1_grid",
                  "S_cens_0_at_Z", "S_cens_1_at_Z"):
            fields[f] = getattr(orc, f)
    return est._replace(**fields)


class _GlobalScalar(keras.layers.Layer):
    """A single learnable scalar, broadcast over the batch and ignoring its input.

    Used to turn one output head of the Weibull MLP into a parameter that does
    not depend on x. The input is only read for its batch dimension.
    """

    def build(self, input_shape):
        self.b = self.add_weight(name="b", shape=(1,), initializer="zeros")

    def call(self, inputs):
        return keras.ops.zeros_like(inputs[:, :1]) + self.b


class SharedShapeWeibullAFT(WeibullAFT):
    """WeibullAFT with one shape parameter for the whole sample.

    `WeibullAFT` reads both k and lambda off the MLP, so the shape varies with x.
    That is not what "accelerated failure time" means -- an AFT model is
    log T = mu(x) + sigma * W with sigma *constant*, i.e. covariates shift the
    location of log T and nothing else. Letting k vary with x turns the model
    into a heteroscedastic Weibull regression, whose extra freedom lands exactly
    where the pseudo-outcomes are most fragile: a subject given a small lambda
    and a large k gets S(t) ~ 0 at late t, and 1/S(t) then enters the AIPCW terms.

    Here the shape comes from `_GlobalScalar` instead, so only the scale depends
    on x. Everything else -- the loss, the fit loop, `survival` -- is inherited,
    including `_get_params` reading column 0 as the shape.
    """

    def _build_model(self, input_dim: int) -> keras.Model:
        inputs = keras.Input(shape=(input_dim,))
        x = inputs
        for units in self.hidden_sizes:
            x = keras.layers.Dense(units, activation="relu")(x)
        lam_raw = keras.layers.Dense(1)(x)
        k_raw = _GlobalScalar()(lam_raw)
        return keras.Model(inputs, keras.layers.Concatenate()([k_raw, lam_raw]))


def cross_fit_censoring(
    data, time_grid, t_end, cfg, *,
    admin_as_censored: bool, shared_shape: bool, hidden_sizes=None,
):
    """Re-estimate only the censoring survival, under a modified specification.

    Two independent corrections, either or both:

    `admin_as_censored`: `cross_fit_nuisances` fits the censoring model with
    event indicator 1 - Delta. That indicator is 1 both when C < min(D, t_end) --
    a real censoring event -- and when min(D, C) >= t_end, where Z = t_end
    exactly and nothing was observed except that follow-up ended. The censoring
    variable is min(C, t_end), so those rows are a point mass at t_end, and a
    Weibull AFT fitting them has to buy that spike with a distorted S(t) at
    t < t_end. The AIPCW terms only ever need S*(t) for t <= t_end, and at
    t = t_end the numerator of the IPCW term is 1{Z > t_end} = 0 for every
    subject, so the atom carries no information the estimator can use; the
    quantity that matters on the interior is the law of C alone -- which is what
    `build_oracle_nuisances` supplies (exp(-t / lambda_c(x))). Treating
    Z = t_end as *censored* for the censoring model targets exactly that.

    `shared_shape`: use `SharedShapeWeibullAFT`, so k does not depend on x. See
    that class for why the per-subject shape is both off-spec and harmful here.

    `hidden_sizes`: width of the MLP behind log-lambda. Defaults to the harness
    setting; pass () for a log-linear scale.

    Returns {"fields": <NuisanceEstimates kwargs>, "shapes": <fitted k stats>}.
    """
    X = np.asarray(data.X, dtype=np.float32)
    W = np.asarray(data.W, dtype=np.float32)
    Z = np.asarray(data.Z, dtype=np.float32)
    Delta = np.asarray(data.Delta, dtype=np.float32)
    t_grid_np = np.asarray(time_grid, dtype=np.float32)
    n, M_plus_1 = X.shape[0], len(t_grid_np)

    # 1 - Delta, but administrative rows (Z at t_end) become censored, not events.
    # Float comparison against t_end: Z is float32 and min(D, C, t_end) returns
    # t_end bit-exactly on those rows, but compare with a tolerance anyway so a
    # future t_end that is not exactly representable does not silently drop the
    # whole correction.
    admin = Z >= t_end - 1e-6
    cens_event = (1.0 - Delta).astype(np.float32)
    if admin_as_censored:
        cens_event = (cens_event * (~admin)).astype(np.float32)

    model_class = SharedShapeWeibullAFT if shared_shape else WeibullAFT
    hidden = cfg["nuisance_hidden"] if hidden_sizes is None else hidden_sizes
    k_seen = []

    S_cens_0_grid = np.zeros((n, M_plus_1), dtype=np.float32)
    S_cens_1_grid = np.zeros((n, M_plus_1), dtype=np.float32)
    S_cens_0_at_Z = np.zeros(n, dtype=np.float32)
    S_cens_1_at_Z = np.zeros(n, dtype=np.float32)

    kf = KFold(n_splits=cfg["n_folds"], shuffle=True, random_state=CROSS_FIT_RNG_SEED)
    train_cfg = dict(epochs=cfg["nuisance_epochs"], batch_size=256, lr=1e-3, verbose=0)

    for train_idx, val_idx in kf.split(X):
        X_val = X[val_idx]
        Z_val = Z[val_idx]
        for w in (0, 1):
            arm = W[train_idx] == w
            model = model_class(hidden_sizes=hidden)
            model.fit(
                X[train_idx][arm], Z[train_idx][arm], cens_event[train_idx][arm],
                **train_cfg,
            )
            k_val, _ = WeibullAFT._get_params(model.model_(X_val, training=False))
            k_seen.append(np.asarray(k_val, dtype=np.float64))
            S_grid = np.asarray(model.survival(t_grid_np, X_val))
            S_at_Z = np.asarray(model.survival(Z_val, X_val))
            if w == 0:
                S_cens_0_grid[val_idx] = S_grid
                S_cens_0_at_Z[val_idx] = S_at_Z
            else:
                S_cens_1_grid[val_idx] = S_grid
                S_cens_1_at_Z[val_idx] = S_at_Z
            del model

    k_all = np.concatenate(k_seen)
    return {
        "fields": dict(
            S_cens_0_grid=jnp.array(S_cens_0_grid),
            S_cens_1_grid=jnp.array(S_cens_1_grid),
            S_cens_0_at_Z=jnp.array(S_cens_0_at_Z),
            S_cens_1_at_Z=jnp.array(S_cens_1_at_Z),
        ),
        # The truth is exponential (k = 1 exactly), so these say directly how far
        # the censoring model's shape wanders from the well-specified value.
        "shapes": {
            "k_mean": float(k_all.mean()),
            "k_sd": float(k_all.std()),
            "k_min": float(k_all.min()),
            "k_max": float(k_all.max()),
            "frac_k_at_floor": float(np.mean(k_all < 0.3 + 1e-3)),
        },
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def q(a, probs=(0.5, 0.9, 0.99, 1.0)):
    """Quantiles of |a| as plain floats, for JSON."""
    a = np.abs(np.asarray(a, dtype=np.float64)).ravel()
    return {f"q{int(p * 100)}": float(np.quantile(a, p)) for p in probs}


def observed_arm(nuis, W, field0, field1):
    """Pick the arm-matching field per subject, as the pseudo-outcome code does."""
    W_bool = np.asarray(W).astype(bool)
    a0 = np.asarray(getattr(nuis, field0))
    a1 = np.asarray(getattr(nuis, field1))
    if a0.ndim == 2:
        return np.where(W_bool[:, None], a1, a0)
    return np.where(W_bool, a1, a0)


def nuisance_report(est, orc, cens_refits, variants, data, time_grid, t_end, cfg):
    """Component accuracy, censoring composition, and pseudo-outcome tails."""
    W = np.asarray(data.W)
    Z = np.asarray(data.Z)
    Delta = np.asarray(data.Delta)
    t_grid = np.asarray(time_grid)
    eps = cfg["eps"]
    rep = {}

    # --- Censoring composition -------------------------------------------
    admin = Z >= t_end - 1e-6
    rep["censoring_composition"] = {
        "event_rate": float(Delta.mean()),
        "censored_rate": float((1 - Delta).mean()),
        "administrative_rate": float(admin.mean()),
        # Of everything the censoring model currently treats as an observed
        # censoring event, what share is the atom at t_end?
        "admin_share_of_cens_events": float(
            admin.sum() / max((1 - Delta).sum(), 1)
        ),
    }

    # --- Propensity -------------------------------------------------------
    e_est = np.asarray(est.e_hat)
    rep["propensity"] = {
        "mean": float(e_est.mean()),
        "sd": float(e_est.std()),
        "min": float(e_est.min()),
        "max": float(e_est.max()),
        "rmse_vs_truth": float(np.sqrt(np.mean((e_est - 0.5) ** 2))),
        # (W - e) is the R-loss weight on tau. Its scale sets how much signal
        # the CATE stage sees; at the truth it is exactly +-0.5.
        "mean_abs_W_minus_e": float(np.abs(W - e_est).mean()),
        "frac_clipped": float(np.mean((e_est < eps) | (e_est > 1 - eps))),
    }

    # --- Survival accuracy, on the observed arm --------------------------
    for name, f0, f1 in (
        ("S_event", "S_event_0_grid", "S_event_1_grid"),
        ("S_cens", "S_cens_0_grid", "S_cens_1_grid"),
    ):
        S_e = observed_arm(est, W, f0, f1)
        S_o = observed_arm(orc, W, f0, f1)
        err = S_e - S_o
        entry = {
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(err.mean()),
            # Same floor fraction for the truth. Without it, a high estimated
            # fraction is ambiguous: it could mean the estimate collapsed, or
            # that the true survival really is below eps there and the clip is
            # doing its job.
            "frac_at_eps_floor_truth": float(np.mean(S_o <= eps)),
            # Bias by time, on a few grid points. A misspecification shows up
            # as a stable sign here; pure estimation noise does not.
            "bias_by_t": {
                f"t={t_grid[i]:.2f}": float(err[:, i].mean())
                for i in (0, len(t_grid) // 4, len(t_grid) // 2,
                          3 * len(t_grid) // 4, len(t_grid) - 1)
            },
            "frac_at_eps_floor": float(np.mean(S_e <= eps)),
            # 1/S is what actually enters the pseudo-outcome.
            "mean_inv_ratio": float(
                np.mean(np.maximum(S_o, eps) / np.maximum(S_e, eps))
            ),
        }
        rep[name] = entry

    # Same accuracy summary for each alternative censoring specification, so the
    # iMSE of its arm can be read against how accurate its S_cens actually is.
    S_o = observed_arm(orc, W, "S_cens_0_grid", "S_cens_1_grid")
    for name, refit in cens_refits.items():
        S_f = np.where(
            W.astype(bool)[:, None],
            np.asarray(refit["fields"]["S_cens_1_grid"]),
            np.asarray(refit["fields"]["S_cens_0_grid"]),
        )
        err = S_f - S_o
        rep[f"S_cens_{name}"] = {
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(err.mean()),
            "frac_at_eps_floor": float(np.mean(S_f <= eps)),
            "mean_inv_ratio": float(
                np.mean(np.maximum(S_o, eps) / np.maximum(S_f, eps))
            ),
            "bias_by_t": {
                f"t={t_grid[i]:.2f}": float(err[:, i].mean())
                for i in (0, len(t_grid) // 4, len(t_grid) // 2,
                          3 * len(t_grid) // 4, len(t_grid) - 1)
            },
            "fitted_shape": refit["shapes"],
        }

    # --- Pseudo-outcome tails --------------------------------------------
    # Reported for every ablation arm, not just the two endpoints. `r_loss_at_zero`
    # is the R-loss the CATE stage would pay with tau == 0, i.e. the part of the
    # objective that carries no signal at all, so comparing it across arms
    # attributes the noise floor to individual nuisance components directly --
    # without having to re-derive the three AIPCW terms here.
    delta = float(t_grid[1] - t_grid[0])
    for label, nuis in variants.items():
        Y_tilde, S_marg, e_clip = compute_pseudo_outcomes(
            data, nuis, jnp.asarray(t_grid), delta, eps=eps
        )
        Y_tilde = np.asarray(Y_tilde)
        resid = Y_tilde - np.asarray(S_marg)
        rep[f"pseudo_outcome_{label}"] = {
            "Y_tilde_abs": q(Y_tilde),
            "residual_abs": q(resid),
            # The R-loss at tau = 0, i.e. how much of the objective is noise the
            # CATE model cannot explain. Directly comparable across arms.
            "r_loss_at_zero": float(np.mean(np.sum(delta * resid ** 2, axis=1))),
            "frac_Y_tilde_over_10": float(np.mean(np.abs(Y_tilde) > 10)),
        }

    return rep


def shape_report(data, cfg):
    """Fitted Weibull shape k(x) for the four survival models, on one fold split.

    Mirrors fold 0 of `cross_fit_nuisances` (same KFold seed and n_folds) rather
    than reaching into it, so the numbers describe models trained on the same
    rows the real ones were.
    """
    X = np.asarray(data.X, dtype=np.float32)
    W = np.asarray(data.W, dtype=np.float32)
    Z = np.asarray(data.Z, dtype=np.float32)
    Delta = np.asarray(data.Delta, dtype=np.float32)
    kf = KFold(n_splits=cfg["n_folds"], shuffle=True, random_state=CROSS_FIT_RNG_SEED)
    train_idx, val_idx = next(iter(kf.split(X)))
    train_cfg = dict(epochs=cfg["nuisance_epochs"], batch_size=256, lr=1e-3, verbose=0)

    out = {}
    for kind in ("event", "cens"):
        indicator = Delta if kind == "event" else 1.0 - Delta
        for w in (0, 1):
            arm = W[train_idx] == w
            model = WeibullAFT(hidden_sizes=cfg["nuisance_hidden"])
            model.fit(
                X[train_idx][arm], Z[train_idx][arm], indicator[train_idx][arm],
                **train_cfg,
            )
            raw = model.model_(X[val_idx], training=False)
            k, lam = WeibullAFT._get_params(raw)
            k = np.asarray(k, dtype=np.float64)
            lam = np.asarray(lam, dtype=np.float64)
            out[f"{kind}_arm{w}"] = {
                "k_mean": float(k.mean()),
                "k_median": float(np.median(k)),
                "k_min": float(k.min()),
                "k_max": float(k.max()),
                # softplus(raw) + 0.3, so "at the floor" means the MLP drove the
                # pre-activation to -inf and the shape carries no covariate
                # information for those rows.
                "frac_k_at_floor": float(np.mean(k < 0.3 + 1e-3)),
                "lam_median": float(np.median(lam)),
            }
            del model
    return out


# ---------------------------------------------------------------------------
# Per-seed driver
# ---------------------------------------------------------------------------

def make_estimator(cfg, seed):
    return AIREstimator(
        M=cfg["M"],
        n_folds=cfg["n_folds"],
        eps=cfg["eps"],
        K=cfg["text_K"],
        encoder_hidden_sizes=cfg["text_encoder_hidden"],
        nuisance_training_config={"epochs": cfg["nuisance_epochs"], "verbose": 0},
        weibull_config={"hidden_sizes": cfg["nuisance_hidden"]},
        propensity_config={"hidden_sizes": cfg["nuisance_hidden"]},
        cate_epochs=cfg["cate_epochs"],
        cate_lr=cfg["cate_lr"],
        patience=cfg["cate_patience"],
        n_restarts=cfg["n_restarts"],
        cate_val_frac=cfg["cate_val_frac"],
        rng_seed=seed,
        cate_model_kwargs=dict(normalize_time=cfg["mgd_normalize_time"]),
    )


def run_seed(n, seed, X_test, cfg, arms):
    dgp = IST3TextDGP
    keras.utils.set_random_seed(seed)
    X_raw, W, Z, Delta, t_end = dgp.generate(n=n, rng_seed=seed)

    # Same standardization the harness applies to this DGP.
    x_mean, x_std = X_raw.mean(axis=0), X_raw.std(axis=0) + 1e-8
    X = ((X_raw - x_mean) / x_std).astype(np.float32)
    X_test_est = ((X_test - x_mean) / x_std).astype(np.float32)

    time_grid, delta = build_time_grid(t_end, cfg["M"])
    data = SurvivalData(
        X=jnp.array(X), W=jnp.array(W), Z=jnp.array(Z), Delta=jnp.array(Delta)
    )

    # Oracle first: build_oracle_nuisances reads dgp._train_pool_idx, which the
    # generate() above just set. Anything that calls generate() again before this
    # point would silently pair this draw with another draw's true parameters.
    orc = build_oracle_nuisances(dgp, X_raw, Z, time_grid)

    t0 = time.time()
    est = cross_fit_nuisances(
        data, time_grid,
        n_folds=cfg["n_folds"],
        propensity_config={"hidden_sizes": cfg["nuisance_hidden"]},
        weibull_config={"hidden_sizes": cfg["nuisance_hidden"]},
        training_config={"epochs": cfg["nuisance_epochs"], "verbose": 0},
        rng_seed=CROSS_FIT_RNG_SEED,
    )
    t_nuis = time.time() - t0

    cens_refits = {
        name: cross_fit_censoring(data, time_grid, t_end, cfg, **kw)
        for name, kw in CENS_REFITS.items()
        if name in arms
    }

    variants = {
        "est_all": est,
        "orc_e": splice(est, orc, e=True),
        "orc_Sev": splice(est, orc, S_event=True),
        "orc_Sce": splice(est, orc, S_cens=True),
        "orc_Sev_Sce": splice(est, orc, S_event=True, S_cens=True),
        "orc_all": orc,
    }
    for name, refit in cens_refits.items():
        variants[name] = est._replace(**refit["fields"])

    out = {"seed": seed, "n": n, "nuisance_seconds": round(t_nuis, 1)}
    out["diagnostics"] = nuisance_report(
        est, orc, cens_refits, variants, data, time_grid, t_end, cfg
    )
    out["shapes"] = shape_report(data, cfg)

    tau_true = dgp.true_tau(np.asarray(time_grid), X_test)
    out["imse"] = {}
    for arm in arms:
        nuis = variants[arm]
        # Reseed before every fit so all arms share one CATE initialization and
        # one validation split: the comparison is then the swapped nuisance
        # component alone, not a different starting point.
        keras.utils.set_random_seed(seed)
        estimator = make_estimator(cfg, seed)
        t1 = time.time()
        estimator.fit(X, W, Z, Delta, t_end, nuisances=nuis, verbose=0)
        tau_hat = estimator.predict_tau(X_test_est)
        mse_t = np.mean((np.asarray(tau_hat) - tau_true) ** 2, axis=0)
        out["imse"][arm] = {
            "imse": float(np.sum(float(estimator.delta_) * mse_t)),
            "seconds": round(time.time() - t1, 1),
            "best_epoch": int(estimator.history_["best_epoch"])
            if estimator.history_ else None,
        }
        print(f"    {arm:12s} iMSE={out['imse'][arm]['imse']:.4f} "
              f"({out['imse'][arm]['seconds']:.0f}s, "
              f"best_epoch={out['imse'][arm]['best_epoch']})", flush=True)
        del estimator
        gc.collect()
        jax.clear_caches()

    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--seeds", type=int, default=6, help="number of seeds")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--arms", type=str, default=",".join(ARMS),
                   help=f"comma-separated subset of: {','.join(ARMS)}")
    p.add_argument("--output", type=str, required=True)
    args = p.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - set(ARMS)
    if unknown:
        p.error(f"unknown arm(s): {sorted(unknown)}; choose from {ARMS}")

    IST3TextDGP.load()
    X_test = IST3TextDGP.generate_test_X(args.n_test)

    results = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        print(f"[seed {seed}] n={args.n}", flush=True)
        t0 = time.time()
        results.append(run_seed(args.n, seed, X_test, CFG, arms))
        print(f"[seed {seed}] done in {time.time() - t0:.0f}s", flush=True)
        # Save after every seed: the nuisance stage is the expensive part and a
        # partial file is still worth reading.
        with open(args.output, "w") as fh:
            json.dump({"cfg": CFG, "arms": arms, "results": results}, fh, indent=2)

    print(f"\nWrote {args.output}")

    # Compact summary over completed seeds.
    print("\n=== mean iMSE by arm ===")
    for arm in arms:
        vals = [r["imse"][arm]["imse"] for r in results if arm in r["imse"]]
        if vals:
            print(f"  {arm:12s} {np.mean(vals):.4f}  "
                  f"(median {np.median(vals):.4f}, n_seeds={len(vals)})")


if __name__ == "__main__":
    main()
