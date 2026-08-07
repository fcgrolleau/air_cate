"""Tests for air_cate.plotting.

These run on hand-built `cv_results` / `histories` dicts rather than a real fit, so
they are fast and independent of the estimator. matplotlib lives in the optional
[plots] extra, so the whole module skips when it is absent.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from air_cate.plotting import (  # noqa: E402
    _paired_stats,
    plot_convergence_grid,
    plot_cv_curve,
    plot_cv_paired,
    plot_k_selection,
)

K_GRID = [1, 2, 3, 4]
N_FOLDS = 3
N_EPOCHS = 12


def _fake_cv_results():
    """CV results with a large fold offset and a small, consistent K effect.

    Fold j carries a level offset of j, dwarfing the K differences -- which is
    exactly the regime the paired panel exists for. Within every fold the loss
    falls from K=1 to K=3 and ticks back up at K=4, so the "right" answer is K=3.
    """
    per_K = np.array([0.30, 0.10, 0.00, 0.04])
    offsets = np.arange(N_FOLDS, dtype=float)
    fold_loss = 8.0 + per_K[:, None] + offsets[None, :]

    return {
        "K_grid": list(K_GRID),
        "fold_val_loss": fold_loss,
        "mean_val_loss": fold_loss.mean(axis=1),
        "se_val_loss": fold_loss.std(axis=1, ddof=1) / np.sqrt(N_FOLDS),
        "fold_best_epoch": np.full((len(K_GRID), N_FOLDS), N_EPOCHS - 2, dtype=int),
    }


def _fake_histories(drop=()):
    """One decaying train/val trajectory per (K, fold); `drop` omits keys."""
    histories = {}
    for K in K_GRID:
        for j in range(N_FOLDS):
            if (K, j) in drop:
                continue
            epochs = np.arange(N_EPOCHS, dtype=float)
            train = 3.0 - 0.02 * K * epochs
            # Validation sits far above training, as it does in a real screen.
            val = 9.0 - 0.01 * K * epochs
            histories[(K, j)] = {
                "train_loss": train.tolist(),
                "val_loss": val.tolist(),
                "best_epoch": N_EPOCHS - 1,
                "best_val_loss": float(val[-1]),
                "monitor": "val",
            }
    return histories


# ---------------------------------------------------------------------------
# _paired_stats
# ---------------------------------------------------------------------------

def test_paired_stats_removes_the_fold_offset():
    """Differencing within folds cancels a pure level shift exactly."""
    cv = _fake_cv_results()
    diff, mean, se = _paired_stats(cv["fold_val_loss"], 0)

    assert diff.shape == (len(K_GRID), N_FOLDS)
    # Row 0 is the reference, so it is identically zero.
    np.testing.assert_allclose(diff[0], 0.0)
    # The fold offset is common to every K, so each row is constant across folds
    # and its SE is zero.
    np.testing.assert_allclose(mean, [0.0, -0.20, -0.30, -0.26], atol=1e-12)
    np.testing.assert_allclose(se, 0.0, atol=1e-12)


def test_paired_se_is_tighter_than_marginal_se():
    """The point of the paired panel: the marginal SE is the wrong error term."""
    cv = _fake_cv_results()
    _, _, se_paired = _paired_stats(cv["fold_val_loss"], 0)

    # Marginal SEs are inflated by the fold offsets; paired SEs are not.
    assert np.all(cv["se_val_loss"] > 0.4)
    assert np.all(se_paired < 1e-9)


def test_paired_stats_reference_choice_shifts_but_does_not_reorder():
    """Changing ref_K shifts every difference by a constant; the ranking holds."""
    cv = _fake_cv_results()
    _, mean_0, _ = _paired_stats(cv["fold_val_loss"], 0)
    _, mean_2, _ = _paired_stats(cv["fold_val_loss"], 2)

    np.testing.assert_allclose(mean_2 - mean_0, mean_2[0] - mean_0[0])
    assert np.argmin(mean_0) == np.argmin(mean_2) == 2


# ---------------------------------------------------------------------------
# plot_cv_curve
# ---------------------------------------------------------------------------

def test_plot_cv_curve_marks_the_minimum():
    cv = _fake_cv_results()
    fig, ax = plot_cv_curve(cv)
    try:
        assert ax.get_xticks().tolist() == [float(k) for k in K_GRID]
        assert "AIR" in ax.get_ylabel()
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("Minimum" in lab and "K=3" in lab for lab in labels), labels
    finally:
        plt.close(fig)


def test_plot_cv_curve_show_folds_adds_one_line_per_fold():
    cv = _fake_cv_results()
    fig, ax_off = plot_cv_curve(cv, show_folds=False)
    n_off = len(ax_off.lines)
    plt.close(fig)

    fig, ax_on = plot_cv_curve(cv, show_folds=True)
    try:
        assert len(ax_on.lines) == n_off + N_FOLDS
    finally:
        plt.close(fig)


def test_plot_cv_curve_one_se_rule_is_off_by_default():
    """On the absolute scale the marginal one-SE threshold is misleading, so it
    must be opt-in."""
    cv = _fake_cv_results()

    fig, ax = plot_cv_curve(cv)
    labels_default = [t.get_text() for t in ax.get_legend().get_texts()]
    plt.close(fig)
    assert not any("One-SE" in lab for lab in labels_default)

    fig, ax = plot_cv_curve(cv, one_se_rule=True)
    try:
        labels_on = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("One-SE" in lab for lab in labels_on)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_cv_paired
# ---------------------------------------------------------------------------

def test_plot_cv_paired_reference_row_is_zero():
    cv = _fake_cv_results()
    fig, ax = plot_cv_paired(cv)
    try:
        # The errorbar's data line carries the plotted means.
        y = None
        for line in ax.lines:
            xd, yd = line.get_data()
            if len(xd) == len(K_GRID) and line.get_marker() == "o":
                y = np.asarray(yd)
                break
        assert y is not None, "no errorbar data line found"
        assert y[0] == pytest.approx(0.0)
        assert np.argmin(y) == 2
        assert "K=1" in ax.get_ylabel()
    finally:
        plt.close(fig)


def test_plot_cv_paired_respects_ref_K():
    cv = _fake_cv_results()
    fig, ax = plot_cv_paired(cv, ref_K=3)
    try:
        assert "K=3" in ax.get_ylabel()
    finally:
        plt.close(fig)


def test_plot_cv_paired_rejects_ref_K_outside_the_grid():
    cv = _fake_cv_results()
    with pytest.raises(ValueError, match="not in K_grid"):
        plot_cv_paired(cv, ref_K=99)


def test_plot_cv_paired_one_se_rule_picks_the_smallest_K_in_the_band():
    """With zero paired SE the band collapses, so the selection is the argmin."""
    cv = _fake_cv_results()
    fig, ax = plot_cv_paired(cv, one_se_rule=True)
    try:
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("Selected" in lab and "K=3" in lab for lab in labels), labels
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_k_selection
# ---------------------------------------------------------------------------

def test_plot_k_selection_returns_two_distinct_panels():
    cv = _fake_cv_results()
    fig, (ax_abs, ax_paired) = plot_k_selection(cv)
    try:
        assert ax_abs is not ax_paired
        assert len(fig.axes) == 2
        # Left panel plots the level, right panel plots the difference from ref.
        assert "CV AIR CATE loss" == ax_abs.get_ylabel()
        assert "loss at $K=1$" in ax_paired.get_ylabel()
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_convergence_grid
# ---------------------------------------------------------------------------

def test_convergence_grid_layout_is_K_by_run():
    """Columns are K, rows are CV run -- the requested orientation."""
    histories = _fake_histories()
    fig, axes = plot_convergence_grid(
        histories, K_grid=K_GRID, n_cv_folds=N_FOLDS
    )
    try:
        assert axes.shape == (N_FOLDS, len(K_GRID))
        top_titles = [axes[0][c].get_title() for c in range(len(K_GRID))]
        assert top_titles == [f"$K = {k}$" for k in K_GRID]
        left_labels = [axes[r][0].get_ylabel() for r in range(N_FOLDS)]
        assert left_labels == [f"Run {r + 1}" for r in range(N_FOLDS)]
    finally:
        plt.close(fig)


def test_convergence_grid_infers_layout_from_keys():
    histories = _fake_histories()
    fig, axes = plot_convergence_grid(histories)
    try:
        assert axes.shape == (N_FOLDS, len(K_GRID))
    finally:
        plt.close(fig)


def test_convergence_grid_normalize_first_starts_both_curves_at_zero():
    """The default normalization is what puts train and val on one readable axis."""
    histories = _fake_histories()
    fig, axes = plot_convergence_grid(
        histories, K_grid=K_GRID, n_cv_folds=N_FOLDS, normalize="first"
    )
    try:
        ax = axes[0][0]
        curves = [
            np.asarray(line.get_ydata())
            for line in ax.lines
            if len(line.get_ydata()) == N_EPOCHS
        ]
        assert len(curves) == 2, "expected a training and a held-out curve"
        for y in curves:
            assert y[0] == pytest.approx(0.0)
    finally:
        plt.close(fig)


def test_convergence_grid_normalize_none_keeps_absolute_levels():
    histories = _fake_histories()
    fig, axes = plot_convergence_grid(
        histories, K_grid=K_GRID, n_cv_folds=N_FOLDS, normalize="none"
    )
    try:
        ax = axes[0][0]
        starts = sorted(
            float(line.get_ydata()[0])
            for line in ax.lines
            if len(line.get_ydata()) == N_EPOCHS
        )
        assert starts == pytest.approx([3.0, 9.0])
    finally:
        plt.close(fig)


def test_convergence_grid_disables_missing_panels():
    """A (K, fold) run absent from histories leaves an empty panel, not a crash."""
    histories = _fake_histories(drop=((2, 1),))
    fig, axes = plot_convergence_grid(
        histories, K_grid=K_GRID, n_cv_folds=N_FOLDS
    )
    try:
        missing = axes[1][K_GRID.index(2)]
        assert not missing.axison
        assert axes[0][0].axison
    finally:
        plt.close(fig)


def test_convergence_grid_handles_histories_without_validation():
    """train_cate can be called with no validation_data; the grid must cope."""
    histories = _fake_histories()
    for h in histories.values():
        h["val_loss"] = None
    fig, axes = plot_convergence_grid(
        histories, K_grid=K_GRID, n_cv_folds=N_FOLDS
    )
    try:
        ax = axes[0][0]
        curves = [
            line for line in ax.lines if len(line.get_ydata()) == N_EPOCHS
        ]
        assert len(curves) == 1
    finally:
        plt.close(fig)
