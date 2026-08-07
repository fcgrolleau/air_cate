"""Plots for cross-validated K selection.

matplotlib is imported lazily inside each function: it ships only in the
optional [plots] extra, so importing this module must not require it.

Series identity is carried by BOTH color and linestyle (solid = training,
dashed = held out) so the figures stay readable in grayscale print and for
colorblind readers, and every multi-series axes carries a legend.

Terminology, kept consistent across every figure here: a *fold* is one outer
cross-validation split (a row of the convergence grid); a *block* is a row-set a
curve is computed on (training block = the other folds pooled, held-out block =
that row's own fold); and *K* is the number of clusters in the mixture of gamma
differences, never a fold count.
"""

import textwrap

# JAMA palette, shared with examples/plot_simulation.py and plot_cate_curves.py.
_JAMA = {
    "teal":   "#374e55",
    "orange": "#df8f44",
    "blue":   "#00a1d5",
    "red":    "#b24745",
    "green":  "#79af97",
}

_RC = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
}


def apply_style():
    """Apply the shared figure style (serif, CM math, thin recessive spines)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(_RC)


def _paired_stats(fold_loss, ref_index: int):
    """Per-K mean and SE of the *paired* difference in CV loss against a reference K.

    Within a fold, every K is scored against identical validation pseudo-outcomes,
    so fold identity is a blocking factor. Differencing against a reference K
    inside each fold removes the fold effect -- which here is large, because each
    held-out block carries its own nuisance noise level -- and leaves the K
    contrast, which is what selecting K actually depends on.

    Args:
        fold_loss: (G, J) array of per-(K, fold) validation losses.
        ref_index: Row index of the reference K.

    Returns:
        (diff, mean, se) where diff is (G, J) and mean/se are (G,).
    """
    import numpy as np

    diff = fold_loss - fold_loss[ref_index][None, :]
    mean = diff.mean(axis=1)
    n_folds = diff.shape[1]
    se = diff.std(axis=1, ddof=1) / np.sqrt(n_folds)
    return diff, mean, se


def plot_convergence(
    history: dict,
    ax=None,
    log_y: bool = False,
    mark_best: bool = True,
    title: str | None = "AIR CATE loss by training epoch",
    figsize: tuple[float, float] = (6.0, 3.8),
    **plot_kwargs,
):
    """Plot the AIR CATE loss against training epoch for a single fit.

    This is the single-fit counterpart to `plot_convergence_grid`: one panel for one
    model, rather than a grid over (K, fold). Use it to check that the CATE stage
    actually converged before reading anything off the fitted model.

    The x axis counts epochs from 1, while `history["best_epoch"]` is 0-indexed as
    `train_cate` returns it; the conversion happens here so callers never see it.

    Both curves are plotted on one shared axis, without the per-curve normalization
    `plot_convergence_grid` defaults to. That is safe here because a single fit's
    training and validation losses come from nuisances cross-fit on the same rows, so
    they sit at comparable levels -- unlike the grid, where the held-out block's
    nuisances are estimated from far fewer rows and the level gap swamps the
    trajectories. `AIREstimator` trains with no validation split at all, so in the
    usual case there is one curve.

    How to read it: a curve still descending at the last epoch means the budget was
    too small -- raise `cate_epochs`. A curve flat from early on means the budget was
    ample. When a validation curve is present, judge overfitting by its slope turning
    upward, and note that the epochs after the marked best one are the ones early
    stopping was measuring against; the fitted weights are the best epoch's, not the
    last one's.

    Args:
        history: The `history_` dict from AIREstimator (or one value of
            AIREstimatorCV's `histories_`), as returned by `train_cate`.
        ax: Optional existing axes to draw into.
        log_y: Use a log y scale. Useful when the first few epochs dwarf the rest.
        mark_best: Mark the best epoch on the monitored curve, and shade the epochs
            after it -- the stretch over which the loss failed to improve.
        title: Axes title. None suppresses it.
        figsize: Figure size when creating a new figure.
        **plot_kwargs: Forwarded to `ax.plot` for the training curve.

    Returns:
        (fig, ax)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    apply_style()

    train = np.asarray(history["train_loss"], dtype=float)
    if train.size == 0:
        raise ValueError("history['train_loss'] is empty; nothing to plot.")
    val = history.get("val_loss")
    val = None if val is None else np.asarray(val, dtype=float)
    epochs = np.arange(1, len(train) + 1)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    plot_kwargs.setdefault("lw", 2.0)
    ax.plot(
        epochs, train,
        color=_JAMA["teal"], ls="-",
        label="Training loss", zorder=3, **plot_kwargs,
    )
    if val is not None:
        ax.plot(
            epochs, val,
            color=_JAMA["orange"], ls="--", lw=2.0,
            label="Held-out loss", zorder=4,
        )

    # The best epoch is chosen on whichever curve `train_cate` monitored, so mark it
    # there: on a validation-monitored run the minimum of the training curve is a
    # different epoch, and marking that one would misreport which weights were kept.
    monitored = val if (history.get("monitor") == "val" and val is not None) else train
    if mark_best and "best_epoch" in history:
        best = int(history["best_epoch"])
        if 0 <= best < len(monitored):
            # Shade the tail first so the curves and marker draw over it.
            if best < len(monitored) - 1:
                ax.axvspan(
                    best + 1, len(monitored),
                    color="#888888", alpha=0.06, zorder=0,
                    label="No further improvement",
                )
            ax.axvline(best + 1, color="#888888", ls=":", lw=0.8, zorder=1)
            ax.plot(
                best + 1, monitored[best],
                marker="o", markersize=10, markerfacecolor="none",
                markeredgecolor=_JAMA["red"], markeredgewidth=1.6,
                linestyle="none", zorder=5,
                label=f"Best epoch ({best + 1}, loss {monitored[best]:.4f})",
            )

    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel("Epoch", fontsize=10.5)
    ax.set_ylabel(
        "AIR CATE loss" if val is not None else "Training AIR CATE loss",
        fontsize=10.5,
    )
    if title is not None:
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(labelsize=8.5)
    ax.grid(True, alpha=0.12, linewidth=0.4)
    # The best epoch is very often the last one (a monotone training curve never
    # trips patience), which puts the marker on the axes boundary and clips it.
    ax.margins(x=0.02, y=0.10)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8, frameon=False, loc="best")

    fig.tight_layout()
    return fig, ax


def plot_cv_curve(
    cv_results: dict,
    ax=None,
    show_folds: bool = False,
    one_se_rule: bool = False,
    title: str = (
        "Cross-validated AIR CATE loss by the number of clusters $K$ "
        "in the mixture of gamma differences"
    ),
    figsize: tuple[float, float] = (6.5, 4.2),
):
    """Plot the cross-validated AIR CATE loss against K, on the absolute scale.

    This panel shows the *level* of the loss and how much it varies between folds.
    It is usually NOT the panel to select K from: the fold-to-fold spread is
    typically an order of magnitude larger than the differences between K values,
    because each held-out block's nuisances are estimated within that block and so
    carry their own estimation error. That error dominates the level -- on
    WeibullMixtureDGP at n=1200 with three folds it accounts for roughly two thirds
    of it -- and is common to every K within a fold, so it inflates the plotted
    marginal SEs but cancels in any comparison of two K values. Use `plot_cv_paired`
    (or `plot_k_selection`, which shows both) to select K.

    `one_se_rule` defaults to False here for the same reason. The textbook
    one-standard-error rule uses the across-fold SE, which is the right error term
    when folds differ only by sampling noise; here folds differ by a level offset
    as well, so on this scale the threshold can sit above the entire curve and
    spuriously nominate K=1. The paired panel applies the rule with the correct
    error term.

    Args:
        cv_results: The `cv_results_` dict from AIREstimatorCV.
        ax: Optional existing axes to draw into.
        show_folds: Overlay one thin line per CV fold. Recommended -- it makes the
            fold-level offsets visible, which is why this scale is hard to read.
        one_se_rule: Draw a dashed line at (min mean loss + its marginal SE).
            Off by default; see above.
        title: Axes title.
        figsize: Figure size when creating a new figure.

    Returns:
        (fig, ax)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    apply_style()

    K_grid = np.asarray(cv_results["K_grid"])
    mean_loss = np.asarray(cv_results["mean_val_loss"], dtype=float)
    se_loss = np.asarray(cv_results["se_val_loss"], dtype=float)
    fold_loss = np.asarray(cv_results["fold_val_loss"], dtype=float)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    if show_folds:
        n_folds = fold_loss.shape[1]
        for j in range(n_folds):
            ax.plot(
                K_grid, fold_loss[:, j],
                color=_JAMA["teal"], lw=0.7, alpha=0.35,
                marker="", zorder=2,
                label="Individual CV folds" if j == 0 else None,
            )

    ax.errorbar(
        K_grid, mean_loss,
        yerr=se_loss,
        color=_JAMA["teal"], lw=2.0,
        marker="o", markersize=5.5,
        capsize=3, capthick=0.8, elinewidth=0.8,
        label="Mean CV loss $\\pm$ SE",
        zorder=4,
    )

    # Mark the minimum
    best_i = int(np.argmin(mean_loss))
    ax.plot(
        K_grid[best_i], mean_loss[best_i],
        marker="o", markersize=10, markerfacecolor="none",
        markeredgecolor=_JAMA["red"], markeredgewidth=1.6,
        linestyle="none",
        label=f"Minimum ($K={K_grid[best_i]}$)",
        zorder=5,
    )

    if one_se_rule and np.isfinite(se_loss[best_i]):
        threshold = mean_loss[best_i] + se_loss[best_i]
        ax.axhline(
            threshold, color=_JAMA["red"], ls=":", lw=1.0, zorder=1,
            label="One-SE threshold",
        )

    ax.set_xlabel("Number of clusters $K$", fontsize=10.5)
    ax.set_ylabel("CV AIR CATE loss", fontsize=10.5)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8)
    ax.set_xticks(K_grid)
    ax.tick_params(labelsize=8.5)
    ax.grid(True, alpha=0.12, linewidth=0.4)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8, frameon=False, loc="best")

    fig.tight_layout()
    return fig, ax


def plot_cv_paired(
    cv_results: dict,
    ref_K: int | None = None,
    ax=None,
    show_folds: bool = True,
    one_se_rule: bool = True,
    title: str = "Change in CV AIR loss relative to $K_{\\mathrm{ref}}$ (paired)",
    figsize: tuple[float, float] = (6.5, 4.2),
):
    """Plot the *paired* change in CV AIR loss against K. **Decide K from this.**

    Each fold's losses are differenced against the same fold's loss at `ref_K`
    before averaging, which removes the fold-level nuisance offset that dominates
    the absolute scale. The error bars are the SE of those paired differences, so
    they measure the precision of the K contrast itself -- typically an order of
    magnitude tighter than the marginal SEs in `plot_cv_curve`.

    How to read it: points below the zero line are better than the reference K.
    Pick the smallest K that is clearly below zero and not clearly worse than the
    minimum; the one-SE band marks that set. Points whose error bars straddle each
    other are not distinguishable by this screen.

    Args:
        cv_results: The `cv_results_` dict from AIREstimatorCV.
        ref_K: Reference K to difference against. Defaults to the smallest K in the
            grid (usually K=1, the no-heterogeneity model), so the plot reads as
            "how much does allowing K clusters buy over no heterogeneity".
        ax: Optional existing axes to draw into.
        show_folds: Overlay one thin line per CV fold.
        one_se_rule: Shade the band within one paired SE of the best mean
            difference and mark the smallest K inside it.
        title: Axes title.
        figsize: Figure size when creating a new figure.

    Returns:
        (fig, ax)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    apply_style()

    K_grid = np.asarray(cv_results["K_grid"])
    fold_loss = np.asarray(cv_results["fold_val_loss"], dtype=float)

    if ref_K is None:
        ref_index = 0
    else:
        matches = np.flatnonzero(K_grid == ref_K)
        if matches.size == 0:
            raise ValueError(f"ref_K={ref_K} is not in K_grid={list(K_grid)}")
        ref_index = int(matches[0])

    diff, mean_diff, se_diff = _paired_stats(fold_loss, ref_index)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    ax.axhline(0.0, color="#888888", ls="-", lw=0.8, zorder=1)

    if show_folds:
        for j in range(diff.shape[1]):
            ax.plot(
                K_grid, diff[:, j],
                color=_JAMA["teal"], lw=0.7, alpha=0.35, marker="",
                zorder=2,
                label="Individual CV folds" if j == 0 else None,
            )

    best_i = int(np.argmin(mean_diff))

    if one_se_rule and np.isfinite(se_diff[best_i]):
        threshold = mean_diff[best_i] + se_diff[best_i]
        ax.axhspan(
            mean_diff[best_i], threshold,
            color=_JAMA["red"], alpha=0.07, zorder=0,
            label="Within one paired SE of best",
        )
        within = np.flatnonzero(mean_diff <= threshold)
        chosen_i = int(within[0]) if within.size else best_i
    else:
        chosen_i = best_i

    ax.errorbar(
        K_grid, mean_diff,
        yerr=se_diff,
        color=_JAMA["teal"], lw=2.0,
        marker="o", markersize=5.5,
        capsize=3, capthick=0.8, elinewidth=0.8,
        label="Mean paired difference $\\pm$ SE",
        zorder=4,
    )

    ax.plot(
        K_grid[chosen_i], mean_diff[chosen_i],
        marker="o", markersize=10, markerfacecolor="none",
        markeredgecolor=_JAMA["red"], markeredgewidth=1.6,
        linestyle="none",
        label=f"Selected ($K={K_grid[chosen_i]}$)",
        zorder=5,
    )

    ax.set_xlabel("Number of clusters $K$", fontsize=10.5)
    ax.set_ylabel(
        f"CV AIR loss $-$ loss at $K={K_grid[ref_index]}$", fontsize=10.5
    )
    ax.set_title(title.replace("K_{\\mathrm{ref}}", f"K={K_grid[ref_index]}"),
                 fontsize=11.5, fontweight="bold", pad=8)
    ax.set_xticks(K_grid)
    ax.tick_params(labelsize=8.5)
    ax.grid(True, alpha=0.12, linewidth=0.4)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8, frameon=False, loc="best")

    fig.tight_layout()
    return fig, ax


def plot_k_selection(
    cv_results: dict,
    ref_K: int | None = None,
    figsize: tuple[float, float] = (11.5, 4.3),
):
    """Two-panel K-selection figure: absolute scale (left) and paired scale (right).

    The left panel is the honest picture of the loss level and its fold-to-fold
    spread; the right panel is the one to select K from. Two panels rather than two
    y-axes on one plot: the quantities differ by an order of magnitude, and a
    twin-axis chart would make their relative size meaningless.

    Why the left panel is labelled "held-out nuisance error dominates". Each outer
    fold re-cross-fits its nuisances inside its own held-out block, so each fold's
    validation pseudo-outcomes carry that block's nuisance estimation error. On
    WeibullMixtureDGP at n=1200 with three outer folds, recomputing the same three
    held-out blocks under one common set of nuisances fitted on all n rows moved the
    loss level from about 9.6 to about 3.4 and cut the fold-to-fold range from 1.67
    to 0.58. So the level is mostly held-out nuisance error rather than signal, and
    that error is also the majority driver of the offsets between folds; the
    remainder is ordinary subject sampling. Neither figure is harmed by this,
    because the offset is common to every K within a fold -- which is exactly what
    the paired panel differences away.

    Args:
        cv_results: The `cv_results_` dict from AIREstimatorCV.
        ref_K: Reference K for the paired panel. Defaults to the smallest K.
        figsize: Figure size.

    Returns:
        (fig, (ax_absolute, ax_paired))
    """
    import matplotlib.pyplot as plt

    apply_style()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_cv_curve(
        cv_results, ax=axes[0], show_folds=True, one_se_rule=False,
        title="Absolute scale -- held-out nuisance error dominates",
    )
    plot_cv_paired(
        cv_results, ref_K=ref_K, ax=axes[1], show_folds=True, one_se_rule=True,
        title="Paired scale -- select $K$ here",
    )

    fig.suptitle(
        "Cross-validated AIR CATE loss by the number of clusters $K$ "
        "in the mixture of gamma differences",
        fontsize=12.5, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig, (axes[0], axes[1])


def plot_convergence_grid(
    histories: dict,
    K_grid=None,
    n_cv_folds: int = None,
    sharey: str = "row",
    mark_best: bool = True,
    normalize: str = "first",
    title: str = (
        "AIR CATE loss convergence by the number of clusters $K$ in the mixture of "
        "gamma differences (columns)\nand cross-validation fold (rows)"
    ),
    panel_size: tuple[float, float] = (2.1, 1.75),
):
    """Grid of per-fold convergence curves: columns = K, rows = cross-validation fold.

    Each panel shows the AIR CATE loss against epoch for one (K, fold) fit. Within a
    row, the *training block* is the other n_cv_folds - 1 folds pooled, and the
    *held-out block* is that row's own fold. Both curves are labelled "block" rather
    than "fold" so that "fold" refers unambiguously to the row -- K counts mixture
    clusters, not folds.

    Why `normalize="first"` is the default. The held-out block's nuisances are
    cross-fit *within* that block, so they come from roughly 1/n_cv_folds as many
    rows as the training block's. Their noisier pseudo-outcomes carry a much larger
    irreducible residual, so the held-out curve sits far above the training curve --
    in practice a level difference several times the training loss itself, not a
    modest offset. On a shared absolute axis that gap swamps the epoch-to-epoch
    movement both curves are drawn to show, and each panel collapses into two flat
    lines. Plotting each curve minus its own epoch-1 value puts both trajectories on
    one axis at a readable scale, which is what a convergence plot is for. Pass
    `normalize="none"` to see the absolute levels instead.

    The level gap is NOT a measure of overfitting -- it is a property of the
    nuisance sample sizes and is common to every K within a row, so it does not
    distort the comparison across K. What diagnoses overfitting is the held-out
    curve *turning upward* as epochs increase. Judge by slope, never by the gap.

    Args:
        histories: The `histories_` dict from AIREstimatorCV, keyed by (K, fold).
        K_grid: Column order. Inferred from the keys if None.
        n_cv_folds: Number of rows. Inferred from the keys if None.
        sharey: Passed to plt.subplots. "row" keeps each fold on a common scale
            (comparable across K within a fold) while letting folds differ.
        mark_best: Mark the best held-out epoch in each panel.
        normalize: "first" (default) plots each curve minus its own epoch-1 value;
            "none" plots the raw loss.
        title: Figure suptitle.
        panel_size: (width, height) in inches per panel.

    Returns:
        (fig, axes) where axes is always a 2D array.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    apply_style()

    if K_grid is None:
        K_grid = sorted({k for k, _ in histories.keys()})
    else:
        K_grid = list(K_grid)
    if n_cv_folds is None:
        n_cv_folds = len({j for _, j in histories.keys()})

    n_cols = len(K_grid)
    fig, axes = plt.subplots(
        nrows=n_cv_folds,
        ncols=n_cols,
        figsize=(panel_size[0] * n_cols, panel_size[1] * n_cv_folds),
        sharex=True,
        sharey=sharey,
        squeeze=False,
    )

    for row in range(n_cv_folds):
        for col, K in enumerate(K_grid):
            ax = axes[row][col]
            history = histories.get((K, row))

            if history is None:
                ax.set_axis_off()
                continue

            train = np.asarray(history["train_loss"], dtype=float)
            if normalize == "first" and train.size:
                train = train - train[0]
            epochs = np.arange(1, len(train) + 1)

            if normalize == "first":
                ax.axhline(0.0, color="#bbbbbb", ls="-", lw=0.6, zorder=0)

            ax.plot(
                epochs, train,
                color=_JAMA["teal"], ls="-", lw=1.6,
                label="Training block", zorder=3,
            )

            val = history.get("val_loss")
            if val is not None:
                val = np.asarray(val, dtype=float)
                if normalize == "first" and val.size:
                    val = val - val[0]
                ax.plot(
                    epochs, val,
                    color=_JAMA["orange"], ls="--", lw=1.6,
                    label="Held-out block", zorder=4,
                )

                if mark_best:
                    best_epoch = int(history["best_epoch"])
                    ax.axvline(
                        best_epoch + 1, color="#888888", ls=":", lw=0.8, zorder=1,
                    )
                    ax.plot(
                        best_epoch + 1, val[best_epoch],
                        marker="o", markersize=4.5,
                        markerfacecolor="none", markeredgecolor=_JAMA["red"],
                        markeredgewidth=1.2, linestyle="none", zorder=5,
                        label="Best epoch",
                    )

            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.12, linewidth=0.4)

            if row == 0:
                ax.set_title(f"$K = {K}$", fontsize=10, fontweight="bold", pad=6)
            if col == 0:
                ax.set_ylabel(f"Fold {row + 1}", fontsize=9)
            if row == n_cv_folds - 1:
                ax.set_xlabel("Epoch", fontsize=9)

    fig.suptitle(title, fontsize=11.5, fontweight="bold", y=0.995)

    # One shared legend below the grid, taken from the first populated panel.
    handles, labels = [], []
    for row in range(n_cv_folds):
        for col in range(n_cols):
            h, lab = axes[row][col].get_legend_handles_labels()
            if h:
                handles, labels = h, lab
                break
        if handles:
            break

    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(labels),
        fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.012),
        columnspacing=1.5, handletextpad=0.6,
    )

    if normalize == "first":
        caption = (
            "Rows are cross-validation folds; columns are candidate numbers of "
            "clusters $K$ in the mixture of gamma differences. Each curve is shown "
            "as the change from its own epoch-1 value, so the two trajectories "
            "share one axis despite their very different levels. Judge overfitting "
            "by the held-out slope turning upward, never by the gap between curves."
        )
    else:
        caption = (
            "Rows are cross-validation folds; columns are candidate numbers of "
            "clusters $K$ in the mixture of gamma differences. The held-out curve "
            "sits above the training curve because its nuisances are estimated from "
            "fewer rows; judge overfitting by its slope, not the gap."
        )

    # Wrap the caption to the figure's own width. Callers save with
    # bbox_inches="tight", which grows the canvas to fit any overhanging artist -- so
    # an unwrapped caption pads the figure out to its own length and squashes the
    # panels. ~17 characters per inch at 7.5pt italic.
    fig.text(
        0.5, -0.045,
        textwrap.fill(caption, width=int(fig.get_figwidth() * 17)),
        ha="center", va="top", fontsize=7.5, color="#666666", style="italic",
        linespacing=1.5,
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    return fig, axes
