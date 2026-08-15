"""Plot predicted time-varying tv-CATE curves for test observations.

Fits 5 non-oracle methods on one training draw (Weibull DGP) and overlays their
predicted tau(t|x) against the true tv-CATE.

Two outputs:
  1. cate_curves_all_seed{SEED}.pdf  -- multi-page contact sheet with one panel
     per test observation, so a good-looking subject can be picked by eye.
  2. <out-stem>.{png,pdf}            -- full-size figure for a single subject
     (--obs IDX; defaults to the automatic pick).

Survival Forest predictions are step functions (cadlag / piecewise constant), so
they are drawn with steps-post rather than linear interpolation.

This is the figure where the pointwise-horizon limitation is visible rather than
merely asserted. The causal survival forest is refit from scratch at every one of
the M grid points, with nothing tying neighbouring horizons together, so its
curve wanders between adjacent t while the AIR curves -- continuous in t by
construction -- do not. Requires R with the grf package; `grf_available` is
checked before any fitting starts rather than after the other four methods have
already been paid for.

Reproducibility. Predictions depend on five independent random streams and all
five are pinned by --seed: `keras.utils.set_random_seed` covers Keras/JAX weight
initialisation (and the Python/NumPy global RNGs), `rng_seed=` covers the CATE
minibatch order, `random_state=` covers the survival forests, `seed=` covers
grf's C++ generator, and the DGP takes its own `rng_seed`. Nuisance cross-fitting
splits are already deterministic (rng_seed=42 inside cross_fit). The
configuration that produced a cache is stored inside the .npz, and --plot-only
refuses a cache some other configuration wrote: a stale cache does not error, it
silently plots the wrong curves.

Usage:
  uv run --extra plots python examples/plot_cate_curves.py --seed 0
  uv run --extra plots python examples/plot_cate_curves.py --seed 0 --plot-only --obs 81

  # Presentation-size figure, in two steps. Step 1 fits (~9 min: about 4 for the
  # four in-process methods and 5 for the M single-horizon grf forests) and caches;
  # --no-single keeps it from overwriting the paper-size cate_curves.{png,pdf}.
  # Step 2 restyles from the cache in seconds, so the subject and the styling can
  # be re-picked without refitting.
  uv run --extra plots python examples/plot_cate_curves.py --seed 2 --no-single
  uv run --extra plots python examples/plot_cate_curves.py --seed 2 --plot-only \
      --obs 52 --preset presentation --out-stem cate_curves_seed2_obs52

  # Same figure for a non-technical audience: --preset explainer is
  # "presentation" plus the favors-intervention / favors-control arrows beside
  # the zero line and plain-language axis labels (y ticks in percentage points).
  uv run --extra plots python examples/plot_cate_curves.py --seed 2 --plot-only \
      --obs 24 --preset explainer --out-stem cate_curves_seed2_obs24_explainer
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
from matplotlib.ticker import FuncFormatter, MultipleLocator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from air_cate import AIREstimator  # sets KERAS_BACKEND=jax; import before keras
from air_cate.baselines import SurvivalForestTLearner, SurvivalForestSLearner
from air_cate.baselines_csf import CausalSurvivalForest, grf_available
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
#
# The causal survival forest gets "default" rather than "steps-post": it is not a
# step function but M independently fitted point estimates, one forest per
# horizon, so linear interpolation between them is the honest reading. Drawing it
# as a staircase would suggest a piecewise-constant structure the estimator never
# claims, and would visually smooth over the horizon-to-horizon variation that is
# the whole point of showing it here. Labels match METHOD_LABELS in
# plot_simulation.py so the two figures name the same method identically.
METHODS = [
    ("Cross-fitted AIR w/ MGD",      PALETTE["teal"],   "-",  2.2, "default"),
    ("Cross-fitted AIR w/ TP",       PALETTE["orange"], "-",  2.2, "default"),
    ("Survival Forest T-learner",    PALETTE["red"],    "-",  2.2, "steps-post"),
    ("Survival Forest S-learner",    PALETTE["green"],  "-",  2.2, "steps-post"),
    ("Causal Survival Forest (grf)", PALETTE["purple"], "-",  2.2, "default"),
]

# Legend text for the single-subject figure only.
FIGURE_LABEL_OVERRIDES = {
    "Cross-fitted AIR w/ MGD": "Cross-fitted AIR w/ MGD (our method)",
}

# Plain-language relabelling for a non-technical audience (--plain-labels). The
# axis labels say what the quantity *is*, because "$\hat{\tau}(t \mid x)$" says
# nothing to a reader who has not read the paper, and the y axis is rescaled to
# percentage points, which is how a difference of two survival probabilities is
# normally read. The legend overrides below only drop implementation detail; the
# method names keep matching the paper and plot_simulation.py.
PLAIN_LABEL_OVERRIDES = {
    # "TP" spelled out, and the grf package name dropped: both are
    # implementation detail to a reader outside the project. The methods
    # themselves are still named as the paper names them.
    "Cross-fitted AIR w/ TP": "Cross-fitted AIR w/ Tensor-Product",
    "Causal Survival Forest (grf)": "Causal Survival Forest",
}
PLAIN_XLABEL = "Time since treatment"
PLAIN_YLABEL = "Difference in survival probability"

# Text beside the two arrows flanking the zero line. tau = S_1 - S_0, so tau > 0
# means the treated arm has the higher survival probability -- up is the
# intervention, down is the control. Do not flip these without flipping the sign
# convention in WeibullDGP.true_tau.
FAVORS_UP_LABEL = "Favors intervention"
FAVORS_DOWN_LABEL = "Favors control"

# Abbreviations for the contact-sheet panel titles and the console ranking table,
# where a full label does not fit. The default takes the last word of the label,
# which is what those call sites did inline; only labels whose last word is
# uninformative ("(grf)") need an entry here.
SHORT_LABELS = {
    "Causal Survival Forest (grf)": "CSF",
}


def short_label(label):
    return SHORT_LABELS.get(label, label.split()[-1])

# One short flag for the presentation-size figure, so it is reproduced by name
# rather than by retyping eight numbers. An explicit flag still wins.
PRESETS = {
    "presentation": dict(
        single_only=True, no_title=True, font_scale=2.0, lw_scale=1.3,
        legend_ncol=2, figsize="10,6.4", legend_pad=0.085, rect_bottom=0.10,
    ),
    # "presentation" plus the two readability changes for a non-technical
    # audience: the favors-intervention / favors-control arrows on the right and
    # plain-language axis labels. Kept as its own preset so the paper and
    # presentation figures are byte-for-byte unaffected. rect_bottom drops to
    # 0.02 because the legend is drawn below the canvas and bbox_inches="tight"
    # grows it anyway -- the 0.10 "presentation" reserves is dead white space
    # between the x label and the legend, and the panel is easier to read with
    # that height given back to the axes.
    "explainer": dict(
        single_only=True, no_title=True, font_scale=2.0, lw_scale=1.3,
        legend_ncol=2, figsize="10,6.4", legend_pad=0.085, rect_bottom=0.02,
        favors_arrows=True, plain_labels=True, ylim="-0.15,0.26",
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
# grf defaults, matching the simulation harness. num_threads is pinned to 1 rather
# than left at grf's hardware-concurrency default: this script's contract is that a
# --seed reproduces a curve exactly, and single-threaded training is the
# configuration where that is unambiguous. It costs about five minutes at M=50,
# since every horizon is a separate forest.
CSF_KWARGS = dict(num_trees=2000, num_threads=1)


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
        "csf": CSF_KWARGS,
    }


def cache_path(seed):
    return os.path.join(FIGURES_DIR, f"cate_curves_seed{seed}.npz")


# ---------------------------------------------------------------------------
# Fit all methods, return predicted curves for ALL test subjects
# ---------------------------------------------------------------------------

def fit_all_methods(seed, test_seed):
    """Fit 5 methods on the Weibull DGP, return predictions for all test subjects.

    Seeding is exhaustive -- see the module docstring. `set_random_seed` also
    reseeds the Python and NumPy global RNGs, but the CATE minibatch order draws
    from its own generator, so `rng_seed=` has to be passed explicitly. The causal
    survival forest takes `seed=` explicitly too, rather than inheriting the global
    NumPy RNG the way it does under the simulation harness.
    """
    # Checked before anything is fitted. The causal survival forest is fitted last
    # and needs an external R process, so a missing grf would otherwise only be
    # discovered after the four in-process methods had already been paid for.
    if not grf_available():
        raise SystemExit(
            "The causal survival forest baseline needs R with the grf package, and "
            "it is not available.\n"
            "  Install it with:  Rscript -e 'install.packages(\"grf\")'"
        )

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
        CausalSurvivalForest(M=M, seed=seed, **CSF_KWARGS),
    ]

    tau_hats_all = []  # list of (N_TEST, M+1) arrays
    for est in estimators:
        is_csf = isinstance(est, CausalSurvivalForest)
        note = f" ({M} forests, one per horizon)" if is_csf else ""
        print(f"  Fitting {est.__class__.__name__}{note} ...", end=" ", flush=True)
        est.fit(X_train, W, Z, Delta, t_end, verbose=0)
        tau_hats_all.append(np.asarray(est.predict_tau(X_test)))
        print("done")
        if is_csf:
            # A horizon grf refuses (one past the last observed failure time) is
            # held forward from the previous one, which flattens the tail of the
            # drawn curve. Report it rather than letting the plot imply an estimate
            # was made there. The out-of-range fraction is the structural validity
            # AIR has by construction and this baseline does not.
            print(f"    horizons grf could not fit: "
                  f"{est.n_failed_horizons_}/{M};  "
                  f"|tau_hat| > 1 at "
                  f"{100 * est.frac_out_of_range_:.2f}% of grid points")

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
        print(f"  {label:<30} {r:.4f}{flag}")


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
          "  ".join(f"{short_label(lab)[:7]:>7}" for lab, *_ in METHODS))
    for idx in order[:top]:
        if not np.isfinite(score[idx]):
            break
        print(f"  {idx:5d}  {tau_range[idx]:7.4f}  " +
              "  ".join(f"{imse[j, idx]:7.4f}" for j in range(imse.shape[0])))

    return int(order[0])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _draw_panel(ax, t_grid, tau_true_i, tau_hats_i, lw_scale=1.0,
                emphasize_zero=False):
    """Draw one subject's curves onto an existing axes.

    emphasize_zero darkens the tau = 0 line. It is on whenever the favors
    arrows are drawn: those arrows point away from that line, so a 0.4pt
    hairline is too faint to read as the thing they are anchored to.
    """
    ax.plot(t_grid, tau_true_i, color="#888888", ls="--", lw=1.4 * lw_scale,
            label="True tv-CATE", zorder=6)

    for (label, color, ls, lw, ds), tau_hat_i in zip(METHODS, tau_hats_i):
        ax.plot(t_grid, tau_hat_i, color=color, ls=ls, lw=lw * lw_scale,
                drawstyle=ds, label=label, zorder=4)

    if emphasize_zero:
        ax.axhline(0, color="#555555", lw=0.9 * lw_scale, zorder=1)
    else:
        ax.axhline(0, color="black", lw=0.4, zorder=0)
    ax.grid(True, alpha=0.12, linewidth=0.4)


def _draw_favors_arrows(ax, up_label=FAVORS_UP_LABEL,
                        down_label=FAVORS_DOWN_LABEL, font_scale=1.0,
                        lw_scale=1.0, color="#333333"):
    """Draw two vertical arrows just outside the right spine, split at tau = 0.

    One points up from the zero line ("favors intervention"), one down ("favors
    control"), each with a rotated label alongside. This is what tells a reader
    who does not know what tau is which direction of the y axis is the good one.

    The two arrows are the same length, and that length is the length of the
    longer of the two labels, so each arrow reads as the bracket of its own text
    rather than as a span of the axes. That cannot be hard-coded: the label
    length in axes units depends on the font size, the figure size and the final
    axes box, so the labels are drawn first, measured against a real renderer,
    and the arrows sized from the measurement. Equal length forces the *longer*
    label to set both, which is why the shorter one sits centred in its arrow
    with a little slack at each end.

    Everything vertical is therefore in axes fractions rather than data units --
    the arrow length comes from a pixel measurement, and only the *split point*
    is a data coordinate, converted once from tau = 0 to its axes fraction.

    Call this AFTER fig.tight_layout(): the measurement is against the axes box,
    and tight_layout moves it. The arrows and labels sit outside the axes, so
    they are drawn unclipped; callers save with bbox_inches="tight", which grows
    the canvas to fit them.
    """
    fig = ax.figure
    y_lo, y_hi = ax.get_ylim()
    x_arrow, x_label = 1.03, 1.058
    # tau = 0 as a fraction of the axes height -- where the two arrows part.
    y_zero = (0.0 - y_lo) / (y_hi - y_lo)
    # Break at the zero line, as a fraction of the axes height. Without it the
    # two tails touch and the pair reads as one double-headed line spanning the
    # axes, which says nothing about zero being the dividing point.
    gap = 0.015

    texts = {}
    for sign, label in ((1.0, up_label), (-1.0, down_label)):
        texts[sign] = ax.text(
            x_label, y_zero, label, transform=ax.transAxes,
            rotation=90, ha="center", va="center",
            # Well below the tick-label size, because the arrow length *is* this
            # text's length: the font size is what sets how much of the axes
            # each arrow claims, and hence how far below zero the y axis has to
            # run to hold the downward one. At the tick-label size the pair
            # needs the axis down to -23% on a subject whose curves stop at -8%,
            # which is mostly empty panel; this size fits inside -15%.
            fontsize=6.5 * font_scale, color=color, clip_on=False, zorder=8,
        )

    # Rotated 90 degrees, so a label's on-screen *height* is its text length.
    # Measured in pixels and divided by the axes height in pixels, which makes
    # the result independent of the dpi the measurement happens at -- savefig
    # re-renders at its own dpi, and both terms scale with it.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_h = ax.get_window_extent(renderer).height
    length = max(t.get_window_extent(renderer).height for t in texts.values())
    length /= axes_h

    for sign, text in texts.items():
        tail = y_zero + sign * gap
        tip = y_zero + sign * (gap + length)
        ax.annotate(
            "", xy=(x_arrow, tip), xytext=(x_arrow, tail),
            xycoords=ax.transAxes, textcoords=ax.transAxes,
            annotation_clip=False, zorder=8,
            arrowprops=dict(arrowstyle="-|>", color=color,
                            lw=1.4 * lw_scale, shrinkA=0, shrinkB=0,
                            mutation_scale=9 * font_scale),
        )
        # Centred on its own arrow, so arrow and label share a midpoint.
        text.set_position((x_label, 0.5 * (tail + tip)))

    # The arrows are sized by the text, so it is the y limits that have to make
    # room for them, not the other way round. Say so rather than shipping a
    # clipped arrow, and say what limit would fit. `need` is the fraction of the
    # axes height one arrow claims, plus a little clearance; inverting
    # y_zero = -y_lo / (y_hi - y_lo) at that fraction gives the limit to ask for.
    need = gap + length + 0.01
    if y_zero < need:
        print(f"  note: the 'favors' arrows do not fit below zero at "
              f"--ylim {y_lo:.3f},{y_hi:.3f} -- each one needs "
              f"{100 * need:.0f}% of the axes height.\n"
              f"        try --ylim {-need * y_hi / (1 - need):.3f},{y_hi:.3f}")
    elif 1.0 - y_zero < need:
        print(f"  note: the 'favors' arrows do not fit above zero at "
              f"--ylim {y_lo:.3f},{y_hi:.3f} -- each one needs "
              f"{100 * need:.0f}% of the axes height.\n"
              f"        try --ylim {y_lo:.3f},{-need * y_lo / (1 - need):.3f}")


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
    # The labels are long enough that more than three per row overruns the page
    # width, so the legend wraps. The bottom margin tight_layout reserves has to
    # grow with the row count, or the last row of panels sits on top of it.
    legend_ncol = 3
    n_legend_rows = -(-len(labels) // legend_ncol)

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
                    f"obs {idx}  (best: {short_label(labels[winner + 1])}, "
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
            fig.legend(handles, labels, loc="lower center", ncol=legend_ncol,
                       fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.002),
                       columnspacing=1.5, handletextpad=0.6)
            fig.tight_layout(
                rect=[0, 0.014 + 0.022 * n_legend_rows, 1, 0.975])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved: {path}  ({n_pages} pages, {n_test} observations)")


def plot_curves(t_grid, tau_true_i, tau_hats_i, obs_idx, seed,
                out_stem="cate_curves", show_title=True, font_scale=1.0,
                lw_scale=1.0, legend_ncol=3, figsize=(7, 4.5),
                legend_pad=0.06, rect_bottom=0.10, favors_arrows=False,
                plain_labels=False, ylim=None):
    """Full-size single-observation figure.

    The defaults reproduce the paper-size cate_curves.{png,pdf}. Pass
    font_scale > 1 (with a proportionally larger figsize and fewer legend
    columns) for a presentation-size variant -- `--preset presentation` is
    exactly that combination. Spine and tick widths scale with the fonts via
    apply_style(width_scale=...), so a hairline axis never sits under 20pt labels.

    favors_arrows and plain_labels are the two non-technical-audience options;
    `--preset explainer` is "presentation" plus both. They are cosmetic -- the
    plotted curves are identical either way, and plain_labels rescales only the
    y *tick labels*, by formatter, never the data.

    ylim, when given, is an explicit (lo, hi) in tau units. The arrows read
    whatever the limits end up being, but autoscaling sizes each half from the
    curves alone, so a subject whose effect is almost entirely positive gets a
    down arrow too short to letter. Setting the limits is how the two halves are
    given deliberate proportions.
    """
    apply_style(width_scale=font_scale)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    _draw_panel(ax, t_grid, tau_true_i, tau_hats_i, lw_scale=lw_scale,
                emphasize_zero=favors_arrows)

    if ylim is not None:
        ax.set_ylim(*ylim)

    if plain_labels:
        ax.set_xlabel(PLAIN_XLABEL, fontsize=10.5 * font_scale)
        ax.set_ylabel(PLAIN_YLABEL, fontsize=10.5 * font_scale)
        # tau is a difference of two survival probabilities, so 0.25 is 25%.
        # Zero is left bare: "0%" invites reading the axis as a ratio.
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: "0" if abs(v) < 1e-9 else f"{100 * v:.0f}%"))
        # Fixed 5-point steps rather than the automatic locator. The favors
        # arrows need a y range roughly twice the data's, and over that wider
        # range the locator backs off to 10-point steps, which reads coarser
        # than the effects being shown.
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
    else:
        ax.set_xlabel("Time $t$", fontsize=10.5 * font_scale)
        ax.set_ylabel(r"$\hat{\tau}(t \mid x)$", fontsize=11 * font_scale)
    if show_title:
        ax.set_title(f"Weibull DGP (seed {seed}) -- test observation {obs_idx}",
                     fontsize=11.5, fontweight="bold", pad=8)
    ax.tick_params(labelsize=8.5 * font_scale)

    handles, labels = ax.get_legend_handles_labels()
    labels = [FIGURE_LABEL_OVERRIDES.get(lab, lab) for lab in labels]
    if plain_labels:
        labels = [PLAIN_LABEL_OVERRIDES.get(lab, lab) for lab in labels]
    # loc="lower center" anchors the legend's *bottom* edge, so a legend that wraps
    # to a second row grows upward into the axes and over the x label. Push the
    # anchor down by the extra rows' height so the clearance above the legend stays
    # the legend_pad that was asked for. Roughly 1.6 line-heights per row, 72pt to
    # the inch.
    n_legend_rows = -(-len(labels) // legend_ncol)
    row_height = 1.6 * 8 * font_scale / (72 * figsize[1])
    pad = legend_pad + (n_legend_rows - 1) * row_height
    fig.legend(handles, labels, loc="lower center", ncol=legend_ncol,
               fontsize=8 * font_scale, frameon=False,
               bbox_to_anchor=(0.5, -pad),
               columnspacing=1.5, handletextpad=0.6)

    fig.tight_layout(rect=[0, rect_bottom, 1, 1])

    # After tight_layout, which moves the axes box the arrow lengths are
    # measured against.
    if favors_arrows:
        _draw_favors_arrows(ax, font_scale=font_scale, lw_scale=lw_scale)

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
    parser.add_argument("--favors-arrows", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="draw arrows on the right of the single-subject "
                             "figure, up from the zero line for "
                             f"'{FAVORS_UP_LABEL}' and down for "
                             f"'{FAVORS_DOWN_LABEL}' (default: off; on under "
                             "--preset explainer)")
    parser.add_argument("--plain-labels", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="plain-language axis labels and y ticks in "
                             "percentage points instead of tau notation "
                             "(default: off; on under --preset explainer)")
    parser.add_argument("--legend-ncol", type=int, default=3,
                        help="legend columns on the single-subject figure "
                             "(default: 3)")
    parser.add_argument("--figsize", default="7,4.5", metavar="W,H",
                        help="single-subject figure size in inches "
                             "(default: 7,4.5)")
    parser.add_argument("--ylim", default=None, metavar="LO,HI",
                        help="explicit y limits in tau units, e.g. -0.12,0.26 "
                             "(default: autoscaled). Worth setting alongside "
                             "--favors-arrows, which sizes each arrow from "
                             "these limits")
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


def _parse_pair(value, flag, example):
    """Parse a "a,b" argument into a float pair, or None if value is None."""
    if value is None:
        return None
    try:
        a, b = (float(v) for v in value.split(","))
    except ValueError:
        raise SystemExit(f"{flag} must be two numbers, e.g. {example}")
    return (a, b)


def main():
    args = build_parser().parse_args()

    figsize = _parse_pair(args.figsize, "--figsize", "10,6.4")
    ylim = _parse_pair(args.ylim, "--ylim", "-0.12,0.26")
    if ylim is not None and ylim[0] >= ylim[1]:
        raise SystemExit("--ylim must be LO,HI with LO < HI")

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
                favors_arrows=args.favors_arrows,
                plain_labels=args.plain_labels, ylim=ylim,
                rect_bottom=(max(0.10, args.legend_pad + 0.02)
                             if args.rect_bottom is None else args.rect_bottom))


if __name__ == "__main__":
    main()
