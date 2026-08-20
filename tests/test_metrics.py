"""Tests for air_cate.metrics.

Every case is closed-form on a hand-built curve pair, so nothing here fits a model.
The point is that the definitions are pinned: these metrics are meant to be quoted in
the paper, and a sign-counting convention that drifts silently would change a
published number without breaking anything.
"""

import numpy as np
import pytest

from air_cate.metrics import (
    H_GRID,
    METRIC_KEYS,
    curve_metrics,
    n_direction_reversals,
    n_sign_changes,
    n_sign_reversals,
    range_threshold,
)

M = 20
T_END = 5.0
DELTA = T_END / M
GRID = np.linspace(0.0, T_END, M + 1)


def _flat(n=4, value=0.0):
    return np.full((n, M + 1), value, dtype=float)


def _step_with_jitter(step=1.0, jitter=0.001):
    """One large jump, which sets the dynamic range, plus a tiny alternating wiggle.

    This is the shape the thresholded extrema count exists for: the wiggle is what
    the raw count sees and the threshold is meant to ignore, and it has to be small
    relative to the curve's *range* rather than to its local slope -- which is why a
    noisy ramp will not do, since a ramp's own slope swamps any jitter small enough
    to fall under the threshold.
    """
    m = np.arange(M + 1)
    return (np.where(m > M // 2, step, 0.0) + jitter * (m % 2))[None, :]


# ---------------------------------------------------------------------------
# n_sign_changes
# ---------------------------------------------------------------------------

def _naive_sign_changes(curve, threshold=None):
    """Reference implementation: literally build the compressed sequence and count.

    Deliberately a slow Python loop. The shipped version is a vectorized
    fill-forward, which is the kind of code that is easy to get subtly wrong at the
    boundaries, so it is cross-checked against this on random input below.
    """
    curve = np.atleast_2d(np.asarray(curve))
    out = []
    for i, row in enumerate(curve):
        d = np.diff(row)
        if threshold is not None:
            thr = np.asarray(threshold, dtype=float)
            h = float(thr[i]) if thr.ndim == 1 else float(thr)
            d = np.where(np.abs(d) < h, 0.0, d)
        signs = [s for s in np.sign(d) if s != 0]
        out.append(sum(1 for a, b in zip(signs, signs[1:]) if a != b))
    return np.array(out, dtype=int)


def test_monotone_curve_has_no_reversals():
    assert n_sign_changes(GRID[None, :]) == 0
    assert n_sign_changes(-GRID[None, :]) == 0


def test_constant_curve_has_no_reversals():
    """All forward differences are zero, so the compressed sequence is empty."""
    assert n_sign_changes(_flat(1, 0.3)) == 0


def test_single_peak_is_one_reversal():
    """Up then down: one direction change, i.e. two monotone runs."""
    curve = np.concatenate([np.arange(11.0), np.arange(9.0, -1.0, -1.0)])[None, :]
    assert curve.shape == (1, M + 1)
    assert n_sign_changes(curve) == 1


def test_saw_tooth_counts_every_reversal():
    """Alternating up/down over the whole grid: M-1 reversals."""
    curve = (np.arange(M + 1) % 2).astype(float)[None, :]
    assert n_sign_changes(curve) == M - 1


def test_plateaus_do_not_hide_a_reversal():
    """The step-function case, and the reason the zeros are compressed out.

    Up, flat, down is one reversal. Comparing raw consecutive signs would see
    (+, 0) then (0, -) -- two changes of the *raw* sign sequence for one actual
    change of direction, and a long enough plateau would separate the + from the -
    entirely.
    """
    curve = np.array([[0.0, 1.0, 1.0, 1.0, 1.0, 0.0]])
    assert n_sign_changes(curve) == 1

    # And a plateau inside a monotone run is still zero reversals.
    curve = np.array([[0.0, 1.0, 1.0, 1.0, 2.0, 3.0]])
    assert n_sign_changes(curve) == 0


def test_leading_plateau_does_not_seed_a_phantom_reversal():
    """A curve that starts flat has no nonzero sign to compare its first move to."""
    curve = np.array([[0.0, 0.0, 0.0, 1.0, 2.0]])
    assert n_sign_changes(curve) == 0


def test_rows_are_counted_independently():
    peak = np.concatenate([np.arange(11.0), np.arange(9.0, -1.0, -1.0)])
    saw = (np.arange(M + 1) % 2).astype(float)
    counts = n_sign_changes(np.stack([GRID, peak, saw]))
    np.testing.assert_array_equal(counts, [0, 1, M - 1])


def test_matches_the_naive_reference_on_random_curves():
    """Random walks with ties, which is where a boundary bug would show up."""
    rng = np.random.default_rng(0)
    steps = rng.choice([-1.0, 0.0, 0.0, 1.0], size=(200, M))
    curves = np.concatenate([np.zeros((200, 1)), np.cumsum(steps, axis=1)], axis=1)
    np.testing.assert_array_equal(n_sign_changes(curves), _naive_sign_changes(curves))


def test_threshold_suppresses_sub_threshold_jitter():
    """A jitter of amplitude below the threshold is counted raw and dropped when
    thresholded -- the headline/diagnostic pair the study records."""
    curve = _step_with_jitter()

    assert n_sign_changes(curve) > 0
    assert n_sign_changes(curve, range_threshold(curve)) == 0


def test_threshold_accepts_a_scalar_and_a_per_row_array():
    curves = np.stack([
        0.01 * (np.arange(M + 1) % 2),   # small wiggle
        1.00 * (np.arange(M + 1) % 2),   # large wiggle
    ])
    scalar = n_sign_changes(curves, 0.5)
    np.testing.assert_array_equal(scalar, [0, M - 1])

    per_row = n_sign_changes(curves, np.array([0.5, 2.0]))
    np.testing.assert_array_equal(per_row, [0, 0])


def test_range_threshold_is_a_fraction_of_each_curves_own_span():
    curves = np.stack([np.zeros(M + 1), GRID, 2.0 * GRID])
    np.testing.assert_allclose(
        range_threshold(curves, 0.05), [0.0, 0.05 * T_END, 0.10 * T_END]
    )


# ---------------------------------------------------------------------------
# n_direction_reversals -- the swing filter that supersedes the per-step threshold
# ---------------------------------------------------------------------------

def test_swing_filter_reduces_to_the_raw_count_at_zero_tolerance():
    """The anchor: at h = 0 the new rule must be the old rule, exactly.

    Same random-walk-with-ties battery the naive cross-check uses, because ties are
    where a boundary bug between the two implementations would show up.
    """
    rng = np.random.default_rng(0)
    steps = rng.choice([-1.0, 0.0, 0.0, 1.0], size=(200, M))
    curves = np.concatenate([np.zeros((200, 1)), np.cumsum(steps, axis=1)], axis=1)
    np.testing.assert_array_equal(
        n_direction_reversals(curves, 0.0), n_sign_changes(curves)
    )


def test_swing_filter_sees_a_reversal_the_per_step_rule_misses():
    """The whole reason the swing filter exists, and why h > 0 is a *new* quantity.

    A dip of 0.20 spread over ten steps of 0.02 is a real 0.20 reversal. The per-step
    rule zeroes each 0.02 step against a tolerance of 0.05 and so erases the dip
    entirely; the swing filter asks whether the retrace *totals* more than 0.05.
    """
    curve = np.concatenate([
        np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),        # rise, steps of 0.20
        1.0 - 0.02 * np.arange(1, 11),                    # dip, ten steps of 0.02
        np.array([0.90, 1.00, 1.10, 1.20, 1.30]),         # rise, steps of 0.10
    ])[None, :]
    assert curve.shape == (1, M + 1)
    assert curve.min() == 0.0 and curve.max() == pytest.approx(1.30)

    # Untouched, both rules see the down-then-up pair.
    assert n_sign_changes(curve) == 2
    assert n_direction_reversals(curve, 0.0) == 2

    # At the same tolerance the two disagree, and the swing filter is the one that is
    # right about the dip's size.
    assert n_sign_changes(curve, 0.05) == 0
    assert n_direction_reversals(curve, 0.05) == 2


def test_swing_filter_still_drops_a_genuinely_small_reversal():
    """Monotone in the other direction too: a dip worth less than h is not a reversal."""
    curve = np.concatenate([
        np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        1.0 - 0.002 * np.arange(1, 11),                   # dip totalling only 0.02
        np.array([1.00, 1.10, 1.20, 1.30, 1.40]),
    ])[None, :]
    assert curve.shape == (1, M + 1)

    assert n_direction_reversals(curve, 0.0) == 2
    assert n_direction_reversals(curve, 0.05) == 0        # 0.02 total < 0.05


# ---------------------------------------------------------------------------
# n_sign_reversals -- zero crossings of the effect, which is a different quantity
# ---------------------------------------------------------------------------

def test_a_clean_crossing_reverses_sign_but_never_changes_direction():
    """The pair of curves that shows the two counts are not variants of each other."""
    crossing = np.linspace(-0.3, 0.3, M + 1)[None, :]
    assert n_sign_reversals(crossing) == 1
    assert n_sign_changes(crossing) == 0                  # monotone: no turning point

    wiggly_positive = (1.0 + 0.5 * (np.arange(M + 1) % 2))[None, :]
    assert n_sign_reversals(wiggly_positive) == 0          # never approaches zero
    assert n_sign_changes(wiggly_positive) == M - 1


def test_monotone_curve_has_no_sign_reversals():
    assert n_sign_reversals(GRID[None, :]) == 0
    assert n_sign_reversals(-GRID[None, :]) == 0
    assert n_sign_reversals(_flat(1, 0.3)) == 0
    assert n_sign_reversals(_flat(1, -0.3)) == 0


def test_alternating_sign_counts_every_crossing():
    curve = ((-1.0) ** np.arange(M + 1))[None, :]
    assert n_sign_reversals(curve) == M


def test_the_structural_zero_at_the_origin_seeds_no_sign_reversal():
    """Every method in the study enforces tau(0|x) = 0, so the first grid point sits
    exactly on the boundary. It must establish no state at all, or every curve would
    be credited with a reversal it did not make."""
    rising = np.concatenate([[0.0], np.linspace(0.1, 1.0, M)])[None, :]
    assert n_sign_reversals(rising) == 0

    falling = np.concatenate([[0.0], np.linspace(-0.1, -1.0, M)])[None, :]
    assert n_sign_reversals(falling) == 0

    # And a curve that starts at zero, rises, then genuinely crosses is exactly one.
    crosses = np.concatenate([[0.0], np.linspace(0.5, -0.5, M)])[None, :]
    assert n_sign_reversals(crosses) == 1


def test_the_band_is_a_hysteresis_not_a_clip():
    """An excursion has to clear h on the *new* side to count, so noise skimming zero
    costs nothing while a real crossing of the same width counts."""
    m = np.arange(M + 1)
    jitter = (0.01 * (-1.0) ** m)[None, :]                # +-0.01 about zero
    assert n_sign_reversals(jitter, 0.0) == M             # every crossing, no band
    assert n_sign_reversals(jitter, 0.05) == 0            # none clears the band

    real = (0.20 * (-1.0) ** m)[None, :]
    assert n_sign_reversals(real, 0.05) == M              # all of them clear it


def test_sign_reversals_accept_a_scalar_and_a_per_row_array():
    curves = np.stack([
        0.01 * (-1.0) ** np.arange(M + 1),                # small oscillation
        1.00 * (-1.0) ** np.arange(M + 1),                # large oscillation
    ])
    np.testing.assert_array_equal(n_sign_reversals(curves, 0.05), [0, M])
    np.testing.assert_array_equal(n_sign_reversals(curves, np.array([0.005, 2.0])), [M, 0])


def test_the_epsilon_floor_kills_float_round_off_at_zero_tolerance():
    """Why h = 0 is not literally zero. tau is a difference of exponentials, so a curve
    that touches zero analytically lands a few ULP either side of it. On the Mixture
    DGP's true tau those excursions peak at 2.22e-16 and would be reported as sign
    reversals by a bare h = 0."""
    ulp = (2.22e-16 * (-1.0) ** np.arange(M + 1))[None, :]

    assert n_sign_reversals(ulp, 0.0) == 0                # SIGN_EPS absorbs it
    assert n_sign_reversals(ulp, 0.0, eps=0.0) == M       # and it is the eps doing it


@pytest.mark.parametrize("counter", [n_sign_reversals, n_direction_reversals])
def test_both_counts_are_monotone_non_increasing_in_the_tolerance(counter):
    """Raising the tolerance may only ever remove reversals. Without this a column of
    the figure could exceed the column to its left, which would be meaningless."""
    rng = np.random.default_rng(7)
    steps = rng.choice([-1.0, 0.0, 0.0, 1.0], size=(200, M))
    curves = np.concatenate([np.zeros((200, 1)), np.cumsum(steps, axis=1)], axis=1)

    previous = counter(curves, 0.0)
    for h in (0.25, 0.5, 1.0, 2.0, 4.0):
        current = counter(curves, h)
        assert np.all(current <= previous), f"tolerance {h} increased a count"
        previous = current


# ---------------------------------------------------------------------------
# The DGP invariant the sign-reversal figure rests on
# ---------------------------------------------------------------------------

def _true_tau_on_the_study_grid(dgp_class, n_test=500, m=30, rng_seed=12345):
    """The true tv-CATE on the shared grid, built the way the simulation harness does.

    Mirrors `_make_test_set` in examples/plot_simulation.py: a DGP supplies its own
    test covariates if it has `generate_test_X` (IST-3 does, and it must be called
    before `true_tau` because it selects the super-population rows), otherwise the
    fallback is uniform on [-1, 1]^5.
    """
    from air_cate.utils import build_time_grid

    t_end = getattr(dgp_class, "t_end", 5.0)
    time_grid, _ = build_time_grid(t_end, m)
    if hasattr(dgp_class, "generate_test_X"):
        X = dgp_class.generate_test_X(n_test, rng_seed)
    else:
        X = np.random.default_rng(rng_seed).uniform(-1, 1, (n_test, 5)).astype(np.float32)
    return np.asarray(dgp_class.true_tau(np.asarray(time_grid), X))


def _assert_truth_never_reverses_sign(dgp_class):
    tau_true = _true_tau_on_the_study_grid(dgp_class)

    for h in H_GRID:
        counts = n_sign_reversals(tau_true, h)
        assert counts.max() == 0, (
            f"{dgp_class.__name__}: {np.count_nonzero(counts)} of {len(counts)} true "
            f"curves reverse sign at h={h}; the figure's zero target line assumes none do"
        )

    # Not a vacuous assertion about a flat truth: the curves do have turning points,
    # which is exactly the distinction the two counts draw.
    assert n_sign_changes(tau_true).max() >= 1


def test_the_true_tau_never_reverses_sign_on_the_synthetic_dgps():
    """The claim the sign-reversal figure rests on, and the reason its target is zero.

    No subject's true effect crosses zero on the study grid, so any nonzero count on
    these DGPs is spurious -- which is what makes the metric worth reporting. The
    reason differs by DGP. WeibullDGP puts both arms on a shared shape with differing
    scales, and two such survival curves are strictly ordered for every t > 0, so its
    zero holds at any horizon. WeibullMixtureDGP is not in that family (arm-specific
    mixture weights over components of shape 1, 2, 3) and its true tau does cross, at
    t ~ 16.3 for x_0 = 0, x_1 = -1 -- past its t_end of 5, which is why the assertion
    below, evaluated on the study grid, still holds. So `_true_tau_on_the_study_grid`
    has to keep using each DGP's own t_end for this test to mean what it says.

    If this test ever fails, the figure needs a real target line rather than a zero one.
    """
    from conftest import WeibullDGP, WeibullMixtureDGP

    _assert_truth_never_reverses_sign(WeibullDGP)
    _assert_truth_never_reverses_sign(WeibullMixtureDGP)


@pytest.mark.slow
def test_the_true_tau_never_reverses_sign_on_ist3():
    """Same invariant for IST-3, split out because loading its super-population reads
    a 100k-subject artifact off disk."""
    from air_cate.dgp import IST3TextDGP

    _assert_truth_never_reverses_sign(IST3TextDGP)


# ---------------------------------------------------------------------------
# curve_metrics
# ---------------------------------------------------------------------------

def test_perfect_recovery_is_zero_everywhere():
    rng = np.random.default_rng(1)
    tau = rng.normal(size=(8, M + 1))
    out = curve_metrics(tau, tau, DELTA)

    for key in ("imse", "dmse", "h1", "h1_sq", "excess", "excess_thresh"):
        assert out[key] == pytest.approx(0.0)
    # The extrema counts themselves are not zero -- the truth has its own reversals.
    assert out["n_hat_mean"] == out["n_true_mean"]


def test_constant_error_is_invisible_to_the_derivative_metric():
    """e(t) = c: D = 0 because e has no slope, whatever c is.

    The iMSE value pins the study's existing quadrature convention, which sums
    delta * MSE over all M+1 grid points *including* t=0 -- so a constant error
    integrates to c^2 * t_end * (M+1)/M, not c^2 * t_end. That extra delta has been
    in every reported number since the first run and is left alone here; the point of
    asserting it is that this module did not quietly change it.
    """
    c = 0.2
    tau_true = np.tile(GRID / T_END, (5, 1))
    tau_hat = tau_true + c
    out = curve_metrics(tau_hat, tau_true, DELTA)

    assert out["imse"] == pytest.approx(c**2 * DELTA * (M + 1))
    assert out["dmse"] == pytest.approx(0.0)


def test_error_linear_in_t_gives_the_exact_slope_energy():
    """e(t) = c*t: D = integral of c^2 over [0, t_end] = c^2 * t_end, exactly, since
    the forward difference of a linear function is its slope."""
    c = 0.3
    tau_true = np.zeros((5, M + 1))
    tau_hat = np.tile(c * GRID, (5, 1))
    out = curve_metrics(tau_hat, tau_true, DELTA)

    assert out["dmse"] == pytest.approx(c**2 * T_END)


def test_h1_squared_decomposes_additively():
    """||e||_H1^2 = ||e||_L2^2 + ||e'||_L2^2, which is what makes h1_sq worth
    reporting next to the two pieces."""
    rng = np.random.default_rng(2)
    tau_true = rng.normal(size=(30, M + 1)) * 0.1
    tau_hat = tau_true + rng.normal(size=(30, M + 1)) * 0.1
    out = curve_metrics(tau_hat, tau_true, DELTA)

    assert out["h1_sq"] == pytest.approx(out["imse"] + out["dmse"], rel=1e-6)


def test_h1_norm_is_the_mean_of_norms_not_the_norm_of_the_mean():
    """Jensen: mean sqrt <= sqrt mean, strictly when subjects differ. The headline
    h1 is deliberately the former ("mean H1 norm")."""
    tau_true = np.zeros((2, M + 1))
    tau_hat = np.stack([np.full(M + 1, 0.01), np.full(M + 1, 1.0)])
    out = curve_metrics(tau_hat, tau_true, DELTA)

    assert out["h1"] < np.sqrt(out["h1_sq"])


def test_grid_noise_inflates_the_derivative_metric_as_M_squared():
    """The scaling that makes D worth reporting, and the reason M is fixed at 30.

    A method whose per-horizon errors are independent has D ~ 2*v*M^2/t_end, so
    doubling the grid quadruples the number. A method with a smooth error does not
    move. This is asserted as a ratio so it documents the behaviour rather than
    pinning a constant.
    """
    rng = np.random.default_rng(3)
    ratios = []
    for m in (20, 40):
        grid = np.linspace(0.0, T_END, m + 1)
        delta = T_END / m
        tau_true = np.tile(grid / T_END, (200, 1))
        noisy = tau_true + rng.normal(scale=0.05, size=(200, m + 1))
        smooth = tau_true + 0.05          # constant offset, same iMSE scale
        ratios.append((
            curve_metrics(noisy, tau_true, delta)["dmse"],
            curve_metrics(smooth, tau_true, delta)["dmse"],
        ))

    (d_noisy_20, d_smooth_20), (d_noisy_40, d_smooth_40) = ratios
    assert d_noisy_40 / d_noisy_20 == pytest.approx(4.0, rel=0.15)
    assert d_smooth_20 == pytest.approx(0.0)
    assert d_smooth_40 == pytest.approx(0.0)


def test_excess_is_negative_for_an_oversmoothed_estimate():
    """tau_hat == 0 scores a *perfect* extrema count against a wiggly truth, which is
    exactly why Excess is only interpretable beside iMSE."""
    tau_true = np.tile((np.arange(M + 1) % 2).astype(float), (5, 1))
    tau_hat = np.zeros((5, M + 1))
    out = curve_metrics(tau_hat, tau_true, DELTA)

    assert out["excess"] == pytest.approx(-(M - 1))
    assert out["imse"] > 0.0


def test_excess_reaches_the_iid_ceiling_for_pure_noise():
    """Independent per-horizon estimates have E[N] = (2/3)(M-1).

    Standard turning-point count: among any three consecutive i.i.d. continuous
    values the middle one is the largest or the smallest -- i.e. a reversal -- in 4 of
    the 6 orderings. There are M-1 consecutive difference pairs on an (M+1)-point
    grid. This is the ceiling a method with no continuity mechanism is pushed toward,
    and it is why an *unthresholded* count compresses differences between rough
    methods once they are near it.
    """
    rng = np.random.default_rng(4)
    tau_true = np.zeros((2000, M + 1))
    tau_hat = rng.normal(size=(2000, M + 1))
    out = curve_metrics(tau_hat, tau_true, DELTA)

    assert out["n_hat_mean"] == pytest.approx((2.0 / 3.0) * (M - 1), rel=0.05)
    assert out["n_true_mean"] == 0.0


def test_thresholded_excess_drops_a_sub_threshold_saw_tooth():
    """The recorded diagnostic variant: same curves, jitter below 5% of range
    ignored. The truth here is monotone, so both counts are measuring tau_hat."""
    tau_hat = _step_with_jitter()
    tau_true = np.where(np.arange(M + 1) > M // 2, 1.0, 0.0)[None, :]
    out = curve_metrics(tau_hat, tau_true, DELTA)

    assert out["excess"] > 0.0
    assert out["excess_thresh"] == pytest.approx(0.0)


def test_mse_per_timepoint_is_the_profile_imse_integrates():
    rng = np.random.default_rng(5)
    tau_true = rng.normal(size=(12, M + 1))
    tau_hat = tau_true + rng.normal(size=(12, M + 1)) * 0.2
    out = curve_metrics(tau_hat, tau_true, DELTA)

    profile = np.asarray(out["mse_per_timepoint"])
    assert profile.shape == (M + 1,)
    assert out["imse"] == pytest.approx(float(np.sum(DELTA * profile)))


def test_all_documented_keys_are_present_and_json_safe():
    """`run_all` nan-fills a failed cell from METRIC_KEYS, and every value is written
    straight to JSON, so the contract is both the key set and the types."""
    rng = np.random.default_rng(6)
    tau_true = rng.normal(size=(6, M + 1))
    out = curve_metrics(tau_true + 0.1, tau_true, DELTA)

    assert set(METRIC_KEYS) <= set(out)
    for key in METRIC_KEYS:
        assert isinstance(out[key], float)
    assert all(isinstance(v, float) for v in out["mse_per_timepoint"])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        curve_metrics(np.zeros((3, M + 1)), np.zeros((3, M)), DELTA)
