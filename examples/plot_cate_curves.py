"""Plot predicted time-varying tv-CATE curves for test observations.

Fits 4 non-oracle methods on one training draw (Weibull DGP) and overlays their
predicted tau(t|x) against the true tv-CATE.

Two outputs:
  1. cate_curves_all_seed{SEED}.pdf  -- multi-page contact sheet with one panel
     per test observation, so a good-looking subject can be picked by eye.
  2. <out-stem>.{png,pdf}            -- full-size figure for a single subject
     (--obs IDX; defaults to the automatic pick).

Survival Forest predictions are step functions (cadlag / piecewise constant), so
they are drawn with steps-post rather than linear interpolation.

Reproducibility. Predictions depend on four independent random streams and all
four are pinned by --seed: `keras.utils.set_random_seed` covers Keras/JAX weight
initialisation (and the Python/NumPy global RNGs), `rng_seed=` covers the CATE
minibatch order, `random_state=` covers the survival forests, and the DGP takes
its own `rng_seed`. Nuisance cross-fitting splits are already deterministic
(rng_seed=42 inside cross_fit). The configuration that produced a cache is stored
inside the .npz, and --plot-only refuses a cache some other configuration wrote:
a stale cache does not error, it silently plots the wrong curves.

Usage:
  uv run --extra plots python examples/plot_cate_curves.py --seed 0
  uv run --extra plots python examples/plot_cate_curves.py --seed 0 --plot-only --obs 81

  # Presentation-size figure, in two steps. Step 1 fits (~4 min) and caches;
  # --no-single keeps it from overwriting the paper-size cate_curves.{png,pdf}.
  # Step 2 restyles from the cache in seconds, so the subject and the styling can
  # be re-picked without refitting.
  uv run --extra plots python examples/plot_cate_curves.py --seed 2 --no-single
  uv run --extra plots python examples/plot_cate_curves.py --seed 2 --plot-only \
      --obs 52 --preset presentation --out-stem cate_curves_seed2_obs52
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from air_cate import AIREstimator  # sets KERAS_BACKEND=jax; import before keras
from air_cate.baselines import SurvivalForestTLearner, SurvivalForestSLearner
from air_cate.cate_model.tensor_product import CATETensorProduct
from air_cate.plotting import PALETTE, apply_style
from conftest import WeibullDGP

import keras  # noqa: E402  -- must follow air_cate, which selects the JAX backend

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Style -- palette and rcParams come from air_cate.plotting, not redeclared here
# ---------------------------------------------------------------------------

# (label, color, linestyle, linewidth, drawstyle)
# Survival Forest predictions are piecewise constant: connect the grid points as
# a cadlag step function ("steps-post") instead of interpolating linearly. The
# staircase stays connected (vertical risers are drawn), so no visible gaps.
METHODS = [
    ("Cross-fitted AIR w/ MGD",    PALETTE["teal"],   "-",  2.2, "default"),
    ("Cross-fitted AIR w/ TP",     PALETTE["orange"], "-",  2.2, "default"),
    ("Survival Forest T-learner",  PALETTE["red"],    "-",  2.2, "steps-post"),
    ("Survival Forest S-learner",  PALETTE["green"],  "-",  2.2, "steps-post"),
]

# Legend text for the single-subject figure only. METHODS labels are also used
# for the contact-sheet titles and the console ranking table, both of which
# shorten a label via label.split()[-1] -- renaming there would yield "method)".
FIGURE_LABEL_OVERRIDES = {
    "Cross-fitted AIR w/ MGD": "Cross-fitted AIR w/ MGD (our method)",
}

# One short flag for the presentation-size figure, so it is reproduced by name
# rather than by retyping eight numbers. An explicit flag still wins.
PRESETS = {
    "presentation": dict(
        single_only=True, no_title=True, font_scale=2.0, lw_scale=1.3,
        legend_ncol=2, figsize="10,6.4", legend_pad=0.085, rect_bottom=0.10,
    ),
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_TRAIN = 2000
M = 50
N_TEST = 200
TEST_SEED = 9999

COMMON_AIR = dict(
    M=M, n_folds=2, eps=0.01,
    nuisance_training_config={"epochs": 50, "verbose": 0},
    cate_epochs=200, cate_lr=1e-3,
)
MGD_KWARGS = dict(K=3, encoder_hidden_sizes=(32, 16))
TP_KWARGS = dict(n_time_basis=8, poly_degree=2)
FOREST_KWARGS = dict(n_estimators=200, min_samples_leaf=10)


def run_config(seed, test_seed):
    """Everything that determines the predictions, as a JSON-safe dict.

    Written into the cache so --plot-only can check provenance. Anything that
    changes a curve belongs here; anything purely cosmetic (font sizes, which
    subject gets plotted) deliberately does not, so restyling never invalidates
    a cache.
    """
    return {
        "seed": seed,
        "test_seed": test_seed,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "M": M,
        "air": COMMON_AIR,
        "mgd": {**MGD_KWARGS,
                "encoder_hidden_sizes": list(MGD_KWARGS["encoder_hidden_sizes"])},
        "tp": TP_KWARGS,
        "forest": FOREST_KWARGS,
    }


def cache_path(seed):
    return os.path.join(FIGURES_DIR, f"cate_curves_seed{seed}.npz")


# ---------------------------------------------------------------------------
# Fit all methods, return predicted curves for ALL test subjects
# ---------------------------------------------------------------------------

def fit_all_methods(seed, test_seed):
    """Fit 4 methods on the Weibull DGP, return predictions for all test subjects.

    Seeding is exhaustive -- see the module docstring. `set_random_seed` also
    reseeds the Python and NumPy global RNGs, but the CATE minibatch order draws
    from its own generator, so `rng_seed=` has to be passed explicitly.
    """
    keras.utils.set_random_seed(seed)
    X_train, W, Z, Delta, t_end = WeibullDGP.generate(n=N_TRAIN, rng_seed=seed)

    rng = np.random.default_rng(test_seed)
    X_test = rng.uniform(-1, 1, (N_TEST, 5)).astype(np.float32)

    estimators = [
        AIREstimator(**COMMON_AIR, **MGD_KWARGS, rng_seed=seed),
        AIREstimator(**COMMON_AIR, rng_seed=seed,
                     cate_model_class=CATETensorProduct,
                     cate_model_kwargs=dict(TP_KWARGS)),
        SurvivalForestTLearner(M=M, random_state=seed, **FOREST_KWARGS),
        SurvivalForestSLearner(M=M, random_state=seed, **FOREST_KWARGS),
    ]

    tau_hats_all = []  # list of (N_TEST, M+1) arrays
    for est in estimators:
        print(f"  Fitting {est.__class__.__name__} ...", end=" ", flush=True)
        est.fit(X_train, W, Z, Delta, t_end, verbose=0)
        tau_hats_all.append(np.asarray(est.predict_tau(X_test)))
        print("done")

    t_grid = np.asarray(estimators[0].time_grid_)
    delta = float(estimators[0].delta_)
    tau_true = np.asarray(WeibullDGP.true_tau(t_grid, X_test))  # (N_TEST, M+1)

    return t_grid, delta, tau_true, tau_hats_all


def save_cache(seed, test_seed, t_grid, delta, tau_true, tau_hats_all):
    path = cache_path(seed)
    np.savez_compressed(
        path,
        t_grid=t_grid,
        delta=delta,
        tau_true=tau_true,
        tau_hats=np.stack(tau_hats_all),  # (n_methods, N_TEST, M+1)
        config=json.dumps(run_config(seed, test_seed), sort_keys=True),
    )
    print(f"Saved: {path}")


def load_cache(seed, test_seed, force=False):
    """Load cached predictions, refusing a cache a different configuration wrote."""
    path = cache_path(seed)
    if not os.path.exists(path):
        raise SystemExit(
            f"No cached predictions at {path}. Run without --plot-only first."
        )
    d = np.load(path)
    print(f"Loaded: {path}")

    want = run_config(seed, test_seed)
    got = json.loads(str(d["config"])) if "config" in d.files else None
    if got != want and not force:
        if got is None:
            detail = ("the cache records no configuration, so it predates this "
                      "script or came from another checkout")
        else:
            keys = sorted(set(want) | set(got))
            detail = "differing entries:\n" + "\n".join(
                f"    {k}: cached={got.get(k)!r} current={want.get(k)!r}"
                for k in keys if got.get(k) != want.get(k)
            )
        raise SystemExit(
            f"Cache at {path} does not match the current configuration, so its\n"
            f"  curves are not the ones this script would produce.\n  {detail}\n"
            "  Re-run without --plot-only to refit, or pass --force-cache to plot\n"
            "  it anyway."
        )

    return (d["t_grid"], float(d["delta"]), d["tau_true"],
            [d["tau_hats"][j] for j in range(d["tau_hats"].shape[0])])


# ---------------------------------------------------------------------------
# Per-subject diagnostics
# ---------------------------------------------------------------------------

def compute_imse(delta, tau_true, tau_hats_all):
    """iMSE per method per subject: (n_methods, n_test)."""
    return np.stack([
        np.sum(delta * (tau_hat - tau_true) ** 2, axis=1)
        for tau_hat in tau_hats_all
    ])


def report_fit_health(tau_true, tau_hats_all):
    """Flag degenerate fits.

    A model that collapses to tau_hat == 0 everywhere still scores a small iMSE
    for subjects whose true effect is near zero, so the ranking alone will not
    reveal it. Compare each method's median dynamic range against the truth's.
    """
    def med_range(a):
        return float(np.median(a.max(axis=1) - a.min(axis=1)))

    true_range = med_range(tau_true)
    print(f"\nFit health -- median dynamic range of tau over subjects "
          f"(truth: {true_range:.4f}):")
    for (label, *_), tau_hat in zip(METHODS, tau_hats_all):
        r = med_range(tau_hat)
        flag = "  <-- COLLAPSED to ~0" if r < 0.1 * true_range else ""
        print(f"  {label:<28} {r:.4f}{flag}")


def rank_observations(imse, tau_true, top=15):
    """Rank subjects where AIR w/ MGD wins, by true tv-CATE dynamic range."""
    tau_range = tau_true.max(axis=1) - tau_true.min(axis=1)
    air_best = imse[0] == imse.min(axis=0)
    score = np.where(air_best, tau_range, -np.inf)
    order = np.argsort(-score)

    n_win = int(air_best.sum())
    print(f"\nAIR w/ MGD has the lowest iMSE for {n_win}/{len(tau_range)} subjects.")
    print(f"Top {top} candidates (AIR wins, largest true tv-CATE range):")
    print(f"  {'obs':>5}  {'range':>7}  " +
          "  ".join(f"{lab.split()[-1][:7]:>7}" for lab, *_ in METHODS))
    for idx in order[:top]:
        if not np.isfinite(score[idx]):
            break
        print(f"  {idx:5d}  {tau_range[idx]:7.4f}  " +
              "  ".join(f"{imse[j, idx]:7.4f}" for j in range(imse.shape[0])))

    return int(order[0])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _draw_panel(ax, t_grid, tau_true_i, tau_hats_i, lw_scale=1.0):
    """Draw one subject's curves onto an existing axes."""
    ax.plot(t_grid, tau_true_i, color="#888888", ls="--", lw=1.4 * lw_scale,
            label="True tv-CATE", zorder=6)

    for (label, color, ls, lw, ds), tau_hat_i in zip(METHODS, tau_hats_i):
        ax.plot(t_grid, tau_hat_i, color=color, ls=ls, lw=lw * lw_scale,
                drawstyle=ds, label=label, zorder=4)

    ax.axhline(0, color="black", lw=0.4, zorder=0)
    ax.grid(True, alpha=0.12, linewidth=0.4)


def _legend_handles():
    handles = [Line2D([0], [0], color="#888888", ls="--", lw=1.4)]
    labels = ["True tv-CATE"]
    for label, color, ls, lw, ds in METHODS:
        handles.append(Line2D([0], [0], color=color, ls=ls, lw=lw, drawstyle=ds))
        labels.append(label)
    return handles, labels


def plot_all_observations(t_grid, tau_true, tau_hats_all, imse, seed,
                          n_cols=4, n_rows=5):
    """Multi-page contact sheet: one panel per test observation."""
    apply_style()

    n_test = tau_true.shape[0]
    per_page = n_cols * n_rows
    n_pages = int(np.ceil(n_test / per_page))
    handles, labels = _legend_handles()

    path = os.path.join(FIGURES_DIR, f"cate_curves_all_seed{seed}.pdf")
    with PdfPages(path) as pdf:
        for page in range(n_pages):
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(3.0 * n_cols, 2.1 * n_rows))
            axes = np.atleast_1d(axes).ravel()
            start = page * per_page

            for slot, ax in enumerate(axes):
                idx = start + slot
                if idx >= n_test:
                    ax.axis("off")
                    continue

                _draw_panel(ax, t_grid, tau_true[idx],
                            [th[idx] for th in tau_hats_all], lw_scale=0.65)
                winner = int(np.argmin(imse[:, idx]))
                ax.set_title(
                    f"obs {idx}  (best: {labels[winner + 1].split()[-1]}, "
                    f"AIR iMSE {imse[0, idx]:.3f})",
                    fontsize=7.5, pad=3)
                ax.tick_params(labelsize=6.5)
                if slot >= per_page - n_cols or start + slot + n_cols >= n_test:
                    ax.set_xlabel("Time $t$", fontsize=8)
                if slot % n_cols == 0:
                    ax.set_ylabel(r"$\hat{\tau}(t \mid x)$", fontsize=9)

            fig.suptitle(
                f"Weibull DGP (seed {seed}) -- test observations "
                f"{start}-{min(start + per_page, n_test) - 1} "
                f"of {n_test}   [page {page + 1}/{n_pages}]",
                fontsize=11, fontweight="bold", y=0.995)
            fig.legend(handles, labels, loc="lower center", ncol=5,
                       fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.002),
                       columnspacing=1.5, handletextpad=0.6)
            fig.tight_layout(rect=[0, 0.035, 1, 0.975])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved: {path}  ({n_pages} pages, {n_test} observations)")


def plot_curves(t_grid, tau_true_i, tau_hats_i, obs_idx, seed,
                out_stem="cate_curves", show_title=True, font_scale=1.0,
                lw_scale=1.0, legend_ncol=3, figsize=(7, 4.5),
                legend_pad=0.06, rect_bottom=0.10):
    """Full-size single-observation figure.

    The defaults reproduce the paper-size cate_curves.{png,pdf}. Pass
    font_scale > 1 (with a proportionally larger figsize and fewer legend
    columns) for a presentation-size variant -- `--preset presentation` is
    exactly that combination. Spine and tick widths scale with the fonts via
    apply_style(width_scale=...), so a hairline axis never sits under 20pt labels.
    """
    apply_style(width_scale=font_scale)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    _draw_panel(ax, t_grid, tau_true_i, tau_hats_i, lw_scale=lw_scale)

    ax.set_xlabel("Time $t$", fontsize=10.5 * font_scale)
    ax.set_ylabel(r"$\hat{\tau}(t \mid x)$", fontsize=11 * font_scale)
    if show_title:
        ax.set_title(f"Weibull DGP (seed {seed}) -- test observation {obs_idx}",
                     fontsize=11.5, fontweight="bold", pad=8)
    ax.tick_params(labelsize=8.5 * font_scale)

    handles, labels = ax.get_legend_handles_labels()
    labels = [FIGURE_LABEL_OVERRIDES.get(lab, lab) for lab in labels]
    fig.legend(handles, labels, loc="lower center", ncol=legend_ncol,
               fontsize=8 * font_scale, frameon=False,
               bbox_to_anchor=(0.5, -legend_pad),
               columnspacing=1.5, handletextpad=0.6)

    fig.tight_layout(rect=[0, rect_bottom, 1, 1])
    for ext in ("png", "pdf"):
        path = os.path.join(FIGURES_DIR, f"{out_stem}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    """Build the argument parser, with --preset rewriting style defaults.

    --preset only changes *defaults*, so it has to be read before the remaining
    arguments are parsed; an explicit flag on the command line then still wins.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", choices=sorted(PRESETS), default=None,
                     help="named bundle of figure-style defaults "
                          f"({', '.join(sorted(PRESETS))})")
    preset_args, _ = pre.parse_known_args()

    parser = argparse.ArgumentParser(
        parents=[pre], description="tv-CATE curves per test subject")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the training draw and every model's RNG "
                             "(default: 0)")
    parser.add_argument("--test-seed", type=int, default=TEST_SEED,
                        help=f"seed for the test covariates (default: {TEST_SEED})")
    parser.add_argument("--obs", type=int, default=None,
                        help="test observation for the single-subject figure "
                             "(default: automatic pick)")
    parser.add_argument("--plot-only", action="store_true",
                        help="reuse cached predictions for this seed")
    parser.add_argument("--force-cache", action="store_true",
                        help="plot a cache whose recorded configuration does not "
                             "match this script (inspection only -- the curves "
                             "are then not the ones documented here)")
    parser.add_argument("--no-single", action="store_true",
                        help="skip the single-subject figure")
    parser.add_argument("--single-only", action="store_true",
                        help="skip the multi-page contact sheet and only write "
                             "the single-subject figure")
    parser.add_argument("--out-stem", default="cate_curves",
                        help="basename (no extension) for the single-subject "
                             "figure (default: cate_curves)")
    parser.add_argument("--no-title", action="store_true",
                        help="omit the title on the single-subject figure")
    parser.add_argument("--font-scale", type=float, default=1.0,
                        help="multiply all font sizes on the single-subject "
                             "figure (default: 1.0)")
    parser.add_argument("--lw-scale", type=float, default=1.0,
                        help="multiply curve line widths on the single-subject "
                             "figure (default: 1.0)")
    parser.add_argument("--legend-ncol", type=int, default=3,
                        help="legend columns on the single-subject figure "
                             "(default: 3)")
    parser.add_argument("--figsize", default="7,4.5", metavar="W,H",
                        help="single-subject figure size in inches "
                             "(default: 7,4.5)")
    parser.add_argument("--legend-pad", type=float, default=0.06,
                        help="gap between axes and legend, in figure fractions "
                             "(default: 0.06)")
    parser.add_argument("--rect-bottom", type=float, default=None,
                        help="bottom margin reserved by tight_layout, in figure "
                             "fractions. The legend is drawn outside the canvas "
                             "and bbox_inches='tight' grows it to fit, so a "
                             "small value here just removes dead whitespace "
                             "(default: max(0.10, legend_pad + 0.02))")

    if preset_args.preset is not None:
        parser.set_defaults(**PRESETS[preset_args.preset])
    return parser


def main():
    args = build_parser().parse_args()

    try:
        w, h = (float(v) for v in args.figsize.split(","))
    except ValueError:
        raise SystemExit("--figsize must be two numbers, e.g. 10,6.4")
    figsize = (w, h)

    print(f"--- Weibull DGP (seed {args.seed}, test seed {args.test_seed}) ---")
    if args.plot_only:
        t_grid, delta, tau_true, tau_hats_all = load_cache(
            args.seed, args.test_seed, force=args.force_cache)
    else:
        t_grid, delta, tau_true, tau_hats_all = fit_all_methods(
            args.seed, args.test_seed)
        save_cache(args.seed, args.test_seed, t_grid, delta, tau_true,
                   tau_hats_all)

    imse = compute_imse(delta, tau_true, tau_hats_all)
    report_fit_health(tau_true, tau_hats_all)
    auto_idx = rank_observations(imse, tau_true)

    if not args.single_only:
        plot_all_observations(t_grid, tau_true, tau_hats_all, imse, args.seed)

    if args.no_single:
        return

    obs_idx = auto_idx if args.obs is None else args.obs
    if not 0 <= obs_idx < tau_true.shape[0]:
        raise SystemExit(f"--obs must be in [0, {tau_true.shape[0] - 1}]")
    print(f"\nSingle-subject figure: observation {obs_idx}"
          f"{' (automatic pick)' if args.obs is None else ''}")
    plot_curves(t_grid, tau_true[obs_idx], [th[obs_idx] for th in tau_hats_all],
                obs_idx, args.seed,
                out_stem=args.out_stem, show_title=not args.no_title,
                font_scale=args.font_scale, lw_scale=args.lw_scale,
                legend_ncol=args.legend_ncol, figsize=figsize,
                legend_pad=args.legend_pad,
                rect_bottom=(max(0.10, args.legend_pad + 0.02)
                             if args.rect_bottom is None else args.rect_bottom))


if __name__ == "__main__":
    main()
