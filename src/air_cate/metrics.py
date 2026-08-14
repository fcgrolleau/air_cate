"""Trajectory-quality metrics for an estimated tv-CATE curve.

iMSE -- the integrated pointwise squared error the simulation study has always
reported -- is a delta-weighted Riemann sum, so it is essentially invariant to the
number of grid points M and completely blind to how the curve gets from one grid
point to the next. A method that estimates each horizon independently (the causal
survival forest refits one forest per horizon; the survival-forest T/S-learners read
a step function off one forest) can zigzag between neighbouring t and pay nothing for
it. This module adds the three metrics that do see that.

Everything is evaluated on the shared uniform grid `build_time_grid(t_end, M)` that
every method in the study already uses, with `e = tau_hat - tau_true`:

    iMSE   = mean_i sum_m delta * e_m^2
    D      = mean_i sum_m delta * (Delta e_m / delta)^2   [derivative MSE]
    H1     = mean_i sqrt(sum_m delta * e_m^2 + sum_m delta * (Delta e_m / delta)^2)
    Excess = mean_i [ N(tau_hat(.|x_i)) - N(tau_true(.|x_i)) ]

where N counts sign changes in the *compressed nonzero-sign* sequence of forward
differences -- see `n_sign_changes` for why the compression is not optional.

Two properties of these metrics to keep in view when reporting them:

  - **D is grid-dependent by nature.** It has no continuum limit for a method with no
    derivative: under independent per-horizon noise of variance v it grows like
    2*v*M^2/t_end, so whoever picks M picks the number. The study fixes M = 30 and
    states this. The honest reading is structural rather than competitive -- refining
    the grid makes a pointwise-horizon estimator's *trajectory* strictly worse while
    leaving its iMSE untouched.
  - **Excess does not penalize shrinkage.** A method that returns tau_hat == 0
    everywhere scores a perfect 0 (and can score negative, since the truth itself has
    monotone runs). It is only interpretable beside iMSE.

Pure NumPy: these are evaluation-side quantities, never part of the JAX training path.
"""

import numpy as np

# Fraction of a curve's own dynamic range below which a forward difference is treated
# as flat by the thresholded extrema count. 5% of range at M=30 is roughly the level
# at which a wiggle stops being visible in a printed figure.
DEFAULT_THRESHOLD_FRAC = 0.05

# Every scalar `curve_metrics` returns, in report order. Callers that need to
# nan-fill a failed cell iterate this instead of hardcoding the list.
METRIC_KEYS = (
    "imse",
    "dmse",
    "h1",
    "h1_sq",
    "excess",
    "excess_thresh",
    "n_hat_mean",
    "n_true_mean",
)


def range_threshold(curve, frac=DEFAULT_THRESHOLD_FRAC):
    """Per-subject flatness threshold: `frac` times each curve's own dynamic range.

    Args:
        curve: (n, M+1) array of curve values.
        frac: fraction of the range.

    Returns:
        (n,) array of thresholds.
    """
    curve = np.asarray(curve)
    span = curve.max(axis=1) - curve.min(axis=1)
    return frac * span


def n_sign_changes(curve, threshold=None):
    """Count sign changes in the compressed nonzero-sign sequence of forward diffs.

    This is the extrema count N(f): the number of times f reverses direction, i.e.
    one less than its number of monotone runs.

    The compression is load-bearing. The survival-forest baselines are step functions,
    so long stretches of their forward differences are exactly zero. Counting
    `sign(d_m) != sign(d_{m+1})` over the raw sequence would score every
    plateau boundary as a reversal in one direction and score nothing at all across a
    plateau that separates a genuine up-down pair -- systematically *undercounting*
    the roughest methods and making them look like the smoothest. Dropping the zeros
    first and comparing each nonzero difference to the last nonzero difference before
    it counts direction reversals and nothing else.

    Args:
        curve: (n, M+1) array of curve values. A (M+1,) input is treated as one row.
        threshold: None, a scalar, or an (n,) array. Forward differences with
            |d| < threshold are treated as zero (flat), which is what separates a
            real reversal from grid-scale jitter. `range_threshold` builds the
            per-subject version used by the study.

    Returns:
        (n,) integer array of counts.
    """
    curve = np.atleast_2d(np.asarray(curve))
    d = np.diff(curve, axis=1)

    if threshold is not None:
        thr = np.asarray(threshold, dtype=float)
        if thr.ndim == 1:
            thr = thr[:, None]
        d = np.where(np.abs(d) < thr, 0.0, d)

    s = np.sign(d)                                     # (n, M), zeros = flat
    n, m = s.shape
    if m == 0:
        return np.zeros(n, dtype=int)

    # Fill-forward the most recent nonzero sign. `np.maximum.accumulate` over the
    # column index of each nonzero entry (zeros contributing index 0) gives, at every
    # position, the column of the last nonzero sign at or before it.
    idx = np.where(s != 0, np.arange(m)[None, :], 0)
    last = np.maximum.accumulate(idx, axis=1)
    ff = np.take_along_axis(s, last, axis=1)

    # Shift by one so `prev` is the last nonzero sign *strictly* before each position.
    prev = np.concatenate([np.zeros((n, 1)), ff[:, :-1]], axis=1)

    reversal = (s != 0) & (prev != 0) & (s != prev)
    counts = reversal.sum(axis=1).astype(int)
    return counts


def curve_metrics(tau_hat, tau_true, delta, threshold_frac=DEFAULT_THRESHOLD_FRAC):
    """All trajectory metrics for one (n_test, M+1) pair of estimated and true curves.

    Args:
        tau_hat: (n_test, M+1) estimated tv-CATE on the shared grid.
        tau_true: (n_test, M+1) true tv-CATE on the same grid.
        delta: grid spacing t_end / M.
        threshold_frac: passed to `range_threshold` for `excess_thresh`.

    Returns:
        dict with the scalars in `METRIC_KEYS` plus `mse_per_timepoint` (a list of
        M+1 floats, the pointwise MSE profile the study already records).
    """
    tau_hat = np.asarray(tau_hat)
    tau_true = np.asarray(tau_true)
    if tau_hat.shape != tau_true.shape:
        raise ValueError(
            f"shape mismatch: tau_hat {tau_hat.shape} vs tau_true {tau_true.shape}"
        )
    delta = float(delta)

    err = tau_hat - tau_true

    # iMSE is computed exactly as it was before this module existed -- pointwise mean
    # first, then the delta-weighted sum -- so the values stay comparable, cell for
    # cell, with every result file already on disk.
    mse_per_timepoint = np.mean(err**2, axis=0)
    imse = float(np.sum(delta * mse_per_timepoint))

    # Forward-difference derivative. There are M differences and M * delta = t_end
    # exactly, so the quadrature covers the whole interval with no end correction.
    d_err = np.diff(err, axis=1) / delta
    dmse_per_subject = np.sum(delta * d_err**2, axis=1)
    dmse = float(np.mean(dmse_per_subject))

    # ||e||_{H1}^2 = ||e||_{L2}^2 + ||e'||_{L2}^2. `h1` is the mean of the per-subject
    # NORM (which is what "mean H1 norm" says) and is not recoverable from the two
    # means above; `h1_sq` is the mean of the squared norm and equals imse + dmse,
    # which is the additive decomposition worth reporting alongside it.
    imse_per_subject = np.sum(delta * err**2, axis=1)
    h1_sq_per_subject = imse_per_subject + dmse_per_subject
    h1 = float(np.mean(np.sqrt(h1_sq_per_subject)))
    h1_sq = float(np.mean(h1_sq_per_subject))

    n_hat = n_sign_changes(tau_hat)
    n_true = n_sign_changes(tau_true)
    excess = float(np.mean(n_hat - n_true))

    n_hat_t = n_sign_changes(tau_hat, range_threshold(tau_hat, threshold_frac))
    n_true_t = n_sign_changes(tau_true, range_threshold(tau_true, threshold_frac))
    excess_thresh = float(np.mean(n_hat_t - n_true_t))

    return {
        "imse": imse,
        "dmse": dmse,
        "h1": h1,
        "h1_sq": h1_sq,
        "excess": excess,
        "excess_thresh": excess_thresh,
        "n_hat_mean": float(np.mean(n_hat)),
        "n_true_mean": float(np.mean(n_true)),
        "mse_per_timepoint": [float(v) for v in mse_per_timepoint],
    }
