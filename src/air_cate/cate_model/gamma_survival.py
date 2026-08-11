"""Gamma survival function for treatment effect profiles."""

import jax
import jax.numpy as jnp


def gamma_survival(t: jnp.ndarray, alpha: jnp.ndarray, lam: jnp.ndarray) -> jnp.ndarray:
    """Compute Gamma survival function G(t; alpha, lambda) = P(T > t).

    G(t; alpha, lambda) = gammaincc(alpha, t / lambda)

    where gammaincc is the regularized upper incomplete gamma function,
    and T ~ Gamma(shape=alpha, scale=lambda).

    Args:
        t: Time points. Shape is flexible for broadcasting.
        alpha: Shape parameter (must be > 0).
        lam: Scale parameter (must be > 0).

    Returns:
        Survival probabilities with shape determined by broadcasting.
    """
    x = t / lam

    # G(0) = 1 for every (alpha, lam), so the exact derivative of G at t = 0 is
    # zero in both parameters. The autodiff value there is NaN rather than zero:
    #
    #   d/dx gammaincc(a, x) = -x^(a-1) e^(-x) / Gamma(a)
    #
    # diverges at x = 0 whenever a <= 1, and the chain rule to lam multiplies that
    # infinity by dx/dlam = -t/lam^2 = 0, giving 0 * inf = NaN. t = 0 is always the
    # first point of the time grid, so a single alpha <= 1 poisons the whole
    # gradient: AdamW writes NaN into every lambda and the model is destroyed.
    #
    # That was not hypothetical. alpha_bar = softplus(alpha_raw) + 1e-4 starts at
    # 1.313 and drifts steadily *down* under training, so on the IST-3 DGP every fit
    # eventually walked off this cliff -- measured at epoch 60-120 of 200. The NaN
    # then propagated into lambda and training was effectively over, which is why
    # the last recorded alpha_bar always sat within ~0.002 of 1.0 (1.0008, 1.0018,
    # 0.9993 on three seeds): that is where the run died, not where it converged.
    #
    # Whether the caller *saw* the failure was a race. train_cate snapshots the best
    # weights at the end of an epoch, while the epoch's loss averages pre-update
    # minibatch losses, so a NaN born in an epoch's final batch got snapshotted
    # under a finite loss. With n=2000 and batch 256 that is ~1 epoch in 8, and
    # indeed ~10% of fits returned iMSE = nan while the rest silently early-stopped.
    #
    # The double-where is the standard remedy: evaluate at a dummy x = 1 wherever
    # x == 0 so no infinity is ever produced, then substitute the exact value 1
    # back. Both branches stay finite, so the gradient is the true one -- exactly
    # zero at t = 0, and for a > 1 identical to what autodiff already produced
    # (verified: grad wrt lam is 5.401754 at a = 1.0 both with and without the t = 0
    # point). The gradient at a fixed parameter point is therefore unchanged, but
    # trajectories are not: the NaN was acting as an accidental early stop, so fits
    # now run to the epoch cap and alpha is free to settle below 1 (0.89-0.94
    # measured). Expect CATE-stage numbers to move even where no NaN was reported.
    safe_x = jnp.where(x == 0, jnp.ones_like(x), x)
    return jnp.where(
        x == 0,
        jnp.ones_like(x),
        jax.scipy.special.gammaincc(alpha, safe_x),
    )
