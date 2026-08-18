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

**Two different things get called a "reversal", and they are not variants.** Both are
counted here, at a grid of tolerances h, because the distinction is easy to lose:

  - A **sign reversal** (`n_sign_reversals`) is a zero crossing of the effect itself:
    the treatment goes from harmful to beneficial, or back. This is what a clinician
    means by "the effect reverses". A curve that stays positive throughout has none,
    however wiggly it is.
  - A **direction reversal** (`n_sign_changes`, `n_direction_reversals`) is a turning
    point of the trajectory -- a peak or a trough. This is what `Excess` has always
    measured. A curve going cleanly from -0.3 to +0.3 has none, though its effect
    reverses sign.

In this study's three DGPs the true tau has, per subject, exactly zero sign reversals
and about one direction reversal. That is structural, not incidental: all three define
tau = S_1 - S_0 as a difference of Weibull survivals that share a shape parameter and
differ only in scale, and two such survival curves are strictly ordered for every
t > 0. So each subject's tau rises from 0 to a peak and decays back toward 0 without
touching zero in between. The population still contains harmed and helped subjects
(Weibull splits 44.8/55.2 across subjects, IST-3 8.6/91.4) -- that is a
cross-sectional spread, not a temporal crossing. Any nonzero sign-reversal count on
these DGPs is therefore spurious by construction, which is what makes it worth
reporting; do not carry that guarantee over to a DGP outside this family.

Three properties of these metrics to keep in view when reporting them:

  - **D is grid-dependent by nature.** It has no continuum limit for a method with no
    derivative: under independent per-horizon noise of variance v it grows like
    2*v*M^2/t_end, so whoever picks M picks the number. The study fixes M = 30 and
    states this. The honest reading is structural rather than competitive -- refining
    the grid makes a pointwise-horizon estimator's *trajectory* strictly worse while
    leaving its iMSE untouched.
  - **Neither reversal count penalizes shrinkage.** A method that returns
    tau_hat == 0 everywhere scores a perfect 0 on both (and `Excess` can go negative,
    since the truth itself has monotone runs). They are only interpretable beside iMSE.
  - **`excess` and `excess_thresh` are frozen.** They are the only link back to the
    result files already archived, so their definitions must not move even though
    `excess_thresh`'s per-step threshold rule is superseded by
    `n_direction_reversals`'s swing filter for anything new.

Pure NumPy: these are evaluation-side quantities, never part of the JAX training path.
"""

import numpy as np

# Fraction of a curve's own dynamic range below which a forward difference is treated
# as flat by the thresholded extrema count. 5% of range at M=30 is roughly the level
# at which a wiggle stops being visible in a printed figure.
DEFAULT_THRESHOLD_FRAC = 0.05

# Floor on the sign-reversal tolerance. tau is a difference of two survival
# probabilities computed by subtracting exponentials, so a curve that touches zero
# analytically lands a few ULP either side of it numerically -- on the Mixture DGP's
# true tau the minority-sign excursion peaks at 2.22e-16. Without this floor, h = 0
# would report float round-off as sign reversals (0.093 per subject, all of it noise).
# 1e-9 is nine orders of magnitude below any effect the study can resolve and seven
# above the round-off it has to kill.
SIGN_EPS = 1e-9

# Tolerances at which both reversal counts are recorded. Stored densely because the
# whole run has to be repeated to add one: result files keep scalars, not curves, so a
# tolerance that is not recorded here cannot be recovered without refitting every cell.
# Read as absolute effect sizes on the tau scale (a difference of survival
# probabilities), i.e. 0.02 is "two percentage points"; the `rel` variants below
# multiply each subject's own true dynamic range by the same numbers instead.
H_GRID = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10)

# Suffix for each tolerance, in units of a percentage point, so the keys stay valid
# Python identifiers and sort readably: 0.075 -> "p75" reads as 7.5 points.
_H_LABELS = ("p0", "p1", "p2", "p3", "p5", "p75", "p10")

# rule x side x scale x tolerance. `flip` counts sign reversals of the effect, `rev`
# counts direction reversals (turning points) under the swing filter.
REVERSAL_KEYS = tuple(
    f"{rule}_{side}_{scale}_{label}"
    for rule in ("flip", "rev")
    for side in ("hat", "true")
    for scale in ("abs", "rel")
    for label in _H_LABELS
)

# Every scalar `curve_metrics` returns, in report order. Callers that need to
# nan-fill a failed cell iterate this instead of hardcoding the list. The original
# eight come first so report order is stable, and a result file written before
# REVERSAL_KEYS existed simply lacks those keys -- readers already `.get(key, nan)`.
METRIC_KEYS = (
    "imse",
    "dmse",
    "h1",
    "h1_sq",
    "excess",
    "excess_thresh",
    "n_hat_mean",
    "n_true_mean",
) + REVERSAL_KEYS


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


def _count_compressed_changes(s):
    """Count adjacent unequal pairs in each row's *nonzero* subsequence of `s`.

    The shared primitive behind both reversal counts. `s` is a (n, k) array of -1/0/+1
    states; the zeros are dropped and what remains is scanned for changes. Both callers
    need exactly this, differing only in what they take the state of: `n_sign_changes`
    the sign of the forward difference, `n_sign_reversals` the banded sign of the level.

    The compression is load-bearing in both cases. The survival-forest baselines are
    step functions, so long stretches of their forward differences are exactly zero,
    and a curve sitting inside the sign band contributes a run of zeros. Comparing
    `s_m != s_{m+1}` over the raw sequence would score every plateau boundary as a
    change and score nothing at all across a plateau separating a genuine pair --
    systematically *undercounting* the roughest methods and making them look like the
    smoothest. Comparing each nonzero state to the last nonzero state before it counts
    changes and nothing else.

    Args:
        s: (n, k) array of -1/0/+1 states.

    Returns:
        (n,) integer array of counts.
    """
    n, k = s.shape
    if k == 0:
        return np.zeros(n, dtype=int)

    # Fill-forward the most recent nonzero state. `np.maximum.accumulate` over the
    # column index of each nonzero entry (zeros contributing index 0) gives, at every
    # position, the column of the last nonzero state at or before it.
    idx = np.where(s != 0, np.arange(k)[None, :], 0)
    last = np.maximum.accumulate(idx, axis=1)
    ff = np.take_along_axis(s, last, axis=1)

    # Shift by one so `prev` is the last nonzero state *strictly* before each position.
    prev = np.concatenate([np.zeros((n, 1)), ff[:, :-1]], axis=1)

    return ((s != 0) & (prev != 0) & (s != prev)).sum(axis=1).astype(int)


def _as_column(h, n):
    """Broadcast a scalar or (n,) tolerance to an (n, 1) column for row-wise use."""
    thr = np.asarray(h, dtype=float)
    if thr.ndim == 0:
        return np.full((n, 1), float(thr))
    if thr.ndim == 1:
        return thr[:, None]
    raise ValueError(f"tolerance must be scalar or (n,), got shape {thr.shape}")


def n_sign_reversals(curve, h=0.0, eps=SIGN_EPS):
    """Count sign reversals of the effect: zero crossings with a hysteresis band.

    This is the count a clinician means by "the treatment effect reverses" -- the
    effect goes from harmful to beneficial, or back. It is *not* `n_sign_changes`,
    which counts turning points of the trajectory; see the module docstring.

    The curve is in the positive state while `tau > h` and the negative state while
    `tau < -h`, and neither while it sits inside the band. A reversal is counted each
    time the state changes from one to the other, which makes the band a hysteresis:
    noise skimming zero costs nothing, and only an excursion that clears `h` on the new
    side registers. The state starts as neither, so the structural `tau(0|x) = 0` that
    every method in the study enforces at the first grid point seeds nothing.

    Args:
        curve: (n, M+1) array of curve values. A (M+1,) input is treated as one row.
        h: tolerance, scalar or (n,). Absolute, on the same scale as `curve`. `h = 0`
            counts every crossing subject only to `eps`.
        eps: floor on the tolerance, guarding against float round-off around zero. See
            `SIGN_EPS` for why a bare `h = 0` is not safe on this study's true curves.

    Returns:
        (n,) integer array of counts.
    """
    curve = np.atleast_2d(np.asarray(curve, dtype=float))
    thr = np.maximum(_as_column(h, curve.shape[0]), eps)
    s = np.where(curve > thr, 1, np.where(curve < -thr, -1, 0))
    return _count_compressed_changes(s)


def n_direction_reversals(curve, h=0.0):
    """Count direction reversals (turning points) under a swing filter of size `h`.

    The tolerant version of `n_sign_changes`, and the one to use for anything new. It
    walks each curve holding the running extremum since the last confirmed turn, and
    confirms a new turn only once the curve has retraced more than `h` from that
    extremum. So a reversal has to be worth `h` *in total* to count.

    That total is the point of difference from `n_sign_changes`'s `threshold`
    argument, which zeroes forward differences one at a time. A dip of 0.20 spread over
    ten steps of 0.02 is a real 0.20 reversal, but no single step of it clears a
    threshold of 0.05: the per-step rule scores 0, this scores 2. The per-step rule is
    kept only because `excess_thresh` is already on disk in archived result files.

    At `h = 0` this reduces exactly to `n_sign_changes(curve)` -- both then count a
    reversal at every strict move against the last strict move -- which the test suite
    asserts on random curves.

    Note the initial extremum is the first grid point and is not updated until a
    direction is confirmed, so a curve that wanders inside the band before breaking out
    has its first swing measured from `curve[:, 0]` rather than from the local extremum
    it happened to reach. That is conservative (it can only delay a first confirmation,
    never invent one) and it is what keeps the `h = 0` reduction exact.

    Args:
        curve: (n, M+1) array of curve values. A (M+1,) input is treated as one row.
        h: minimum retrace to confirm a turn, scalar or (n,). Absolute, on the same
            scale as `curve`.

    Returns:
        (n,) integer array of counts.
    """
    curve = np.atleast_2d(np.asarray(curve, dtype=float))
    n, m = curve.shape
    if m == 0:
        return np.zeros(n, dtype=int)
    thr = _as_column(h, n)[:, 0]

    direction = np.zeros(n, dtype=int)      # -1 down, +1 up, 0 not yet established
    extremum = curve[:, 0].copy()           # running extremum since the last turn
    counts = np.zeros(n, dtype=int)

    for j in range(1, m):
        v = curve[:, j]
        # A break out of the band on either side. Mutually exclusive for h >= 0, and
        # gated on `direction` so a move *with* the current direction is an extension
        # of the current run rather than a new swing.
        up = (direction <= 0) & (v > extremum + thr)
        down = (direction >= 0) & (v < extremum - thr)

        # Only a break that contradicts an established direction is a turning point;
        # the first break of all merely establishes one.
        counts += ((up & (direction == -1)) | (down & (direction == 1))).astype(int)

        broke = up | down
        extremum = np.where(
            broke,
            v,
            np.where(
                direction == 1,
                np.maximum(extremum, v),
                np.where(direction == -1, np.minimum(extremum, v), extremum),
            ),
        )
        direction = np.where(up, 1, np.where(down, -1, direction))

    return counts


def n_sign_changes(curve, threshold=None):
    """Count sign changes in the compressed nonzero-sign sequence of forward diffs.

    This is the extrema count N(f): the number of times f reverses direction, i.e.
    one less than its number of monotone runs. See `_count_compressed_changes` for why
    dropping the zeros first is not optional.

    Frozen: `excess` and `excess_thresh` are computed from this and are the only link
    back to archived result files, so neither this function nor its per-step
    `threshold` rule may change. For new work use `n_direction_reversals`, whose swing
    filter measures a reversal's total size rather than its largest single step, or
    `n_sign_reversals` if what you want is a zero crossing of the effect.

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

    return _count_compressed_changes(np.sign(d))       # (n, M), zeros = flat


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

    out = {
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

    # Both reversal rules at every tolerance in `H_GRID`, on both scales, for the
    # estimate and for the truth. The truth's counts are recorded per cell rather than
    # once per DGP so a figure's target line comes out of the same file as its curves,
    # with no second evaluation of the DGP needed to draw it.
    #
    # `abs` reads h as an absolute effect size on the tau scale. `rel` reads it as a
    # fraction of each subject's own *true* dynamic range, and deliberately uses that
    # same yardstick for both sides: scaling the estimate by its own range would score
    # a shrunk-to-zero curve against a tolerance shrunk with it, so its jitter would
    # count as full-size reversals.
    for scale in ("abs", "rel"):
        for frac, label in zip(H_GRID, _H_LABELS):
            h = frac if scale == "abs" else range_threshold(tau_true, frac)
            out[f"flip_hat_{scale}_{label}"] = float(
                np.mean(n_sign_reversals(tau_hat, h))
            )
            out[f"flip_true_{scale}_{label}"] = float(
                np.mean(n_sign_reversals(tau_true, h))
            )
            out[f"rev_hat_{scale}_{label}"] = float(
                np.mean(n_direction_reversals(tau_hat, h))
            )
            out[f"rev_true_{scale}_{label}"] = float(
                np.mean(n_direction_reversals(tau_true, h))
            )

    return out
