"""AIPCW pseudo-outcome computation (pure JAX, no Keras)."""

import jax.numpy as jnp

from air_cate._types import NuisanceEstimates, SurvivalData
from air_cate.utils import clip_propensity, clip_survival


def compute_pseudo_outcomes(
    data: SurvivalData,
    nuisances: NuisanceEstimates,
    time_grid: jnp.ndarray,
    delta: float,
    eps: float = 0.01,
    eps_prop: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute discretized AIPCW pseudo-outcomes on the time grid.

    Args:
        data: SurvivalData with (X, W, Z, Delta).
        nuisances: Cross-fitted nuisance estimates.
        time_grid: (M+1,) uniform grid on [0, t_end].
        delta: Grid step size.
        eps: Clipping bound for the SURVIVAL floors -- event and censoring, on the
            grid and at Z -- and the default for the propensity clip.
        eps_prop: Separate clipping bound for the propensity, defaulting to `eps` so
            existing callers are unaffected.

            Split out because the two floors want OPPOSITE values and one knob cannot
            serve both. Measured on a Stanford CAD cohort (n = 2909 training rows,
            P(W=1) = 0.717): raising eps repairs the censoring tail -- the IPCW
            identity mean(1{Z>t}/S_cens) / S_KM(t) runs 2.04 at eps = 0.05 but
            1.03-1.22 at eps = 0.20 -- while simultaneously wrecking the propensity,
            because clipping an imbalanced cohort into [eps, 1-eps] drags mean(e) away
            from mean(W) and makes mean(W - e) grow 0.014 -> 0.075 -> 0.217.

            That matters because the R-loss's covariate-independent minimiser is the
            UNCENTRED projection sum_i (W-e) r / sum_i (W-e)^2, which decomposes as
            ( cov(W-e, r) + mean(W-e)*mean(r) ) / ( var(W-e) + mean(W-e)^2 ). The
            second numerator term has no causal content, and at eps = 0.20 it was
            0.0868 of a 0.0785 total -- more than the whole estimate.

    Returns:
        Y_tilde: (n, M+1) pseudo-outcome matrix.
        S_marg: (n, M+1) marginal survival S†(t|X).
        e_clipped: (n,) clipped propensity scores.
    """
    W = data.W
    Z = data.Z
    Delta = data.Delta
    n = W.shape[0]
    M = len(time_grid) - 1

    # Clip nuisances. The propensity floor is deliberately separable from the survival
    # floors below -- see `eps_prop`. None reproduces the single-knob behaviour.
    eps_prop = eps if eps_prop is None else eps_prop
    e_clipped = clip_propensity(nuisances.e_hat, eps_prop)

    # Select arm-specific survivals based on treatment assignment
    # W is (n,), broadcast to select arm-specific arrays
    W_bool = W.astype(bool)

    # Event survival on grid: S†_Wi(t_m | X_i), shape (n, M+1)
    S_ev_grid = jnp.where(W_bool[:, None], nuisances.S_event_1_grid, nuisances.S_event_0_grid)
    S_ev_grid = clip_survival(S_ev_grid, eps)

    # Event survival at Z_i: S†_Wi(Z_i | X_i), shape (n,)
    S_ev_at_Z = jnp.where(W_bool, nuisances.S_event_1_at_Z, nuisances.S_event_0_at_Z)
    S_ev_at_Z = clip_survival(S_ev_at_Z, eps)

    # Censoring survival on grid: S*_Wi(t_m | X_i), shape (n, M+1)
    S_ce_grid = jnp.where(W_bool[:, None], nuisances.S_cens_1_grid, nuisances.S_cens_0_grid)
    S_ce_grid = clip_survival(S_ce_grid, eps)

    # Censoring survival at Z_i: S*_Wi(Z_i | X_i), shape (n,)
    S_ce_at_Z = jnp.where(W_bool, nuisances.S_cens_1_at_Z, nuisances.S_cens_0_at_Z)
    S_ce_at_Z = clip_survival(S_ce_at_Z, eps)

    # --- Term 1: IPCW term ---
    # 1{Z_i > t_m} / S*_Wi(t_m | X_i)
    alive_mask = (Z[:, None] > time_grid[None, :]).astype(jnp.float32)  # (n, M+1)
    term1 = alive_mask / S_ce_grid  # (n, M+1)

    # --- Term 2: Censored augmentation ---
    # 1{Z_i <= t_m, Delta_i=0} * S†_Wi(t_m|X_i) / (S†_Wi(Z_i|X_i) * S*_Wi(Z_i|X_i))
    censored_before = ((Z[:, None] <= time_grid[None, :]) & (Delta[:, None] == 0)).astype(jnp.float32)
    term2 = censored_before * S_ev_grid / (S_ev_at_Z[:, None] * S_ce_at_Z[:, None])  # (n, M+1)

    # --- Term 3: Integral augmentation (efficient prefix-sum) ---
    # Censoring hazard at interior grid points m=1,...,M-1
    # lambda*_W(t_m|X) = (log S*_W(t_{m-1}|X) - log S*_W(t_{m+1}|X)) / (2*delta)
    log_S_ce = jnp.log(S_ce_grid)  # (n, M+1)
    # Interior points only: indices 1 to M-1
    cens_hazard_interior = (log_S_ce[:, :-2] - log_S_ce[:, 2:]) / (2.0 * delta)  # (n, M-2)
    # Ensure non-negative hazard
    cens_hazard_interior = jnp.maximum(cens_hazard_interior, 0.0)

    # Summand at interior points: delta * lambda * S†(t_m'|X) / (S†(t_m'|X) * S*(t_m'|X))
    # Wait - the S† in the summand is S†_Wi(t_m'|X_i) (the inner grid point), not the outer t
    # g[i, m'] = delta * lambda*_Wi(t_{m'+1}|X_i) / (S†_Wi(t_{m'+1}|X_i) * S*_Wi(t_{m'+1}|X_i))
    # where m' indexes into interior points 1..M-1
    S_ev_interior = S_ev_grid[:, 1:-1]  # (n, M-2) at grid points 1..M-2
    # Wait, we need interior points 1..M-1 which is M-1 points
    # cens_hazard_interior has shape (n, M-2) because log_S_ce[:, :-2] is indices 0..M-2
    # and log_S_ce[:, 2:] is indices 2..M, giving centered diff at indices 1..M-1
    # Actually: log_S_ce[:, :-2] has length M-1 (indices 0 to M-2)
    # log_S_ce[:, 2:] has length M-1 (indices 2 to M)
    # Their difference gives centered diff at indices 1 to M-1: that's M-1 points
    # Let me recheck: if grid has M+1 points (indices 0..M):
    # :-2 gives indices 0..M-2 (length M-1)
    # 2: gives indices 2..M (length M-1)
    # centered diff at indices 1..M-1: that's M-1 interior points. Correct.

    # Actually the shapes: log_S_ce is (n, M+1)
    # log_S_ce[:, :-2] is (n, M-1), log_S_ce[:, 2:] is (n, M-1)
    # So cens_hazard_interior is (n, M-1) — at interior indices 1..M-1. Good.

    S_ev_interior = S_ev_grid[:, 1:M]  # (n, M-1) at indices 1..M-1
    S_ce_interior = S_ce_grid[:, 1:M]  # (n, M-1) at indices 1..M-1

    # g[i, j] for j-th interior point (j=0 maps to grid index 1, etc.)
    g = delta * cens_hazard_interior / (S_ev_interior * S_ce_interior)  # (n, M-1)

    # Prefix sum: G[i, j] = sum_{j'=0}^{j-1} g[i, j']
    # We need: for each (i, t_m), sum over interior points where t_{m'} <= min(Z_i, t_m)
    # Interior grid index m' corresponds to time t_{m'} = (m') * delta, where m' in {1,...,M-1}
    # For a given upper bound u = min(Z_i, t_m), the number of interior points included is
    # the count of m' in {1,...,M-1} such that m'*delta <= u
    # = min(floor(u/delta), M-1), but m' starts at 1, so count = min(floor(u/delta), M-1)
    # In the prefix sum array (0-indexed), this means we want G_cumsum[count]

    G_cumsum = jnp.cumsum(g, axis=1)  # (n, M-1)
    # Prepend zero column: G_padded[i, 0] = 0, G_padded[i, j] = sum of first j terms
    G_padded = jnp.concatenate([jnp.zeros((n, 1)), G_cumsum], axis=1)  # (n, M)

    # For each (i, m), compute min(Z_i, t_m) and determine the prefix index
    upper_bound = jnp.minimum(Z[:, None], time_grid[None, :])  # (n, M+1)
    # Number of interior points <= upper_bound: floor(upper_bound / delta) clipped to [0, M-1]
    count = jnp.floor(upper_bound / delta).astype(jnp.int32)
    count = jnp.clip(count, 0, M - 1)  # (n, M+1)

    # Look up prefix sums using advanced indexing
    row_idx = jnp.arange(n)[:, None] * jnp.ones((1, M + 1), dtype=jnp.int32)
    prefix = G_padded[row_idx, count]  # (n, M+1)

    # Volume correction and outer factor
    correction = M / max(M - 2, 1)
    term3 = correction * S_ev_grid * prefix  # (n, M+1)

    # --- Combine: Y_tilde = term1 + term2 - term3 ---
    Y_tilde = term1 + term2 - term3

    # --- Marginal survival ---
    S_ev_0_grid = clip_survival(nuisances.S_event_0_grid, eps)
    S_ev_1_grid = clip_survival(nuisances.S_event_1_grid, eps)
    S_marg = S_ev_1_grid * e_clipped[:, None] + S_ev_0_grid * (1.0 - e_clipped[:, None])

    return Y_tilde, S_marg, e_clipped
