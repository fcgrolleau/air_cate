"""One-time training script for the IST-3 Text-Enriched DGP (two-index design).

Usage:
    uv run python -m air_cate.dgp.train_text_dgp

Reads:
    IST_VAE/sampled_data.csv           (100K TVAE rows)

Writes:
    artifacts_text/super_population.npz (X, E, lam0, lam1, D0, D1, C, alpha, k_c, lambda_c)
    artifacts_text/config.json          (DGP parameters and validation metrics)

Design (two-index DGP with one-factor embeddings):
  1. Load structured covariates X (p=10) from IST-3 TVAE pool.
  2. Build z_prog = standardize(PC1(X)).
  3. Sample z_treat ~ N(0,1) independently of X.
  4. Generate one-factor embeddings: E_j | z_treat ~ iid N(z_treat, sigma^2).
  5. Map (z_prog, z_treat) to per-subject Weibull parameters:
       alpha(x)     = exp(log(alpha_0) + a_alpha * z_prog)      [shape]
       lam0(x)      = lambda_base * exp(-a_lambda * z_prog)      [control scale]
       lam1(x,e)    = lam0(x) * exp(gamma_0 + beta * z_treat)   [treated scale]
  6. Draw correlated counterfactual event times using per-subject alpha.
  7. Draw covariate-dependent censoring (exponential):
       lambda_c(x)  = lambda_c_base * exp(-b_lambda * z_prog)
       C ~ Exp(1/lambda_c(x))
"""

import json
import time

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

from air_cate.dgp.encoding import encode_tvae

ARTIFACT_DIR = Path(__file__).parent / "artifacts_text"
TVAE_CSV = Path(__file__).parents[3] / "IST_VAE" / "sampled_data.csv"

# -- Event mechanism ----------------------------------------------------------
ALPHA_0 = 1.5           # baseline Weibull shape
A_ALPHA = 0.3            # shape sensitivity to z_prog
LAMBDA_BASE = 3.0        # baseline Weibull scale
A_LAMBDA = 1.0           # scale sensitivity to z_prog
GAMMA_INTERCEPT = 0.7    # average treatment benefit (positive = treatment helps)
BETA = 0.5               # text-driven treatment-effect heterogeneity

# -- Censoring mechanism (covariate-dependent exponential) --------------------
LAMBDA_CENS_BASE = 5.0   # baseline censoring scale
B_LAMBDA_CENS = 0.3      # censoring scale sensitivity to z_prog

# -- Embeddings ---------------------------------------------------------------
P_EMBED = 5              # embedding dimension d
SIGMA_EMBED = 1.0        # noise std: E_j | z_treat ~ N(z_treat, sigma^2)

T_END = 5.0


def build_super_population(X, E, z_prog, z_treat):
    """Build super-population with per-subject Weibull parameters.

    Event mechanism (two-index):
      alpha(x)  = exp(log(alpha_0) + a_alpha * z_prog)
      lam0(x)   = lambda_base * exp(-a_lambda * z_prog)
      lam1(x,e) = lam0(x) * exp(gamma_0 + beta * z_treat)
      S_w(t|x,e) = exp(-(t / lam_w)^{alpha(x)})

    Censoring (covariate-dependent exponential):
      lambda_c(x) = lambda_c_base * exp(-b_lambda * z_prog)
      C ~ Exp(1/lambda_c(x))
    """
    rng = np.random.default_rng(123)
    n = len(X)

    # Per-subject event Weibull shape
    alpha = np.exp(np.log(ALPHA_0) + A_ALPHA * z_prog).astype(np.float32)

    # Weibull scale parameters
    lam0 = (LAMBDA_BASE * np.exp(-A_LAMBDA * z_prog)).astype(np.float32)
    gamma = (GAMMA_INTERCEPT + BETA * z_treat).astype(np.float32)
    lam1 = (lam0 * np.exp(gamma)).astype(np.float32)

    print(f"  z_prog: mean={z_prog.mean():.4f}, std={z_prog.std():.4f}")
    print(f"  z_treat: mean={z_treat.mean():.4f}, std={z_treat.std():.4f}")
    print(f"  corr(z_prog, z_treat): {np.corrcoef(z_prog, z_treat)[0, 1]:.6f}")
    print(f"  alpha: mean={alpha.mean():.3f}, std={alpha.std():.3f}, "
          f"[{np.percentile(alpha, 5):.2f}, {np.percentile(alpha, 95):.2f}]")
    print(f"  lam0: mean={lam0.mean():.3f}, std={lam0.std():.3f}, "
          f"[{np.percentile(lam0, 5):.2f}, {np.percentile(lam0, 95):.2f}]")
    print(f"  gamma: mean={gamma.mean():.4f}, std={gamma.std():.4f}")
    print(f"  lam1: mean={lam1.mean():.3f}, std={lam1.std():.3f}, "
          f"[{np.percentile(lam1, 5):.2f}, {np.percentile(lam1, 95):.2f}]")

    # Correlated event times via shared uniforms: D_w = lam_w * (-ln U)^{1/alpha}
    U = rng.uniform(1e-10, 1.0 - 1e-10, n).astype(np.float64)
    neg_log_U = -np.log(U)
    D0 = (lam0 * neg_log_U ** (1.0 / alpha)).astype(np.float32)
    D1 = (lam1 * neg_log_U ** (1.0 / alpha)).astype(np.float32)

    # Covariate-dependent censoring: C ~ Exp(1/lambda_c(x))
    lambda_c = (LAMBDA_CENS_BASE * np.exp(-B_LAMBDA_CENS * z_prog)).astype(np.float32)
    C = (rng.exponential(1.0, n) * lambda_c).astype(np.float32)
    print(f"  lambda_c: mean={lambda_c.mean():.3f}, std={lambda_c.std():.3f}")
    print(f"  D0: mean={D0.mean():.2f}, median={np.median(D0):.2f}")
    print(f"  D1: mean={D1.mean():.2f}, median={np.median(D1):.2f}")
    print(f"  C:  mean={C.mean():.2f}, median={np.median(C):.2f}")

    # Effective censoring rate
    W_sim = rng.binomial(1, 0.5, n)
    D_sim = np.where(W_sim, D1, D0)
    event_rate = float((D_sim < np.minimum(C, T_END)).mean())
    print(f"  Event rate (sim): {event_rate:.3f}")

    return {
        "X": X, "E": E,
        "lam0": lam0, "lam1": lam1,
        "D0": D0, "D1": D1, "C": C,
        "alpha": alpha,
        "lambda_c": lambda_c,
    }


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # -- 1. Load structured covariates -----------------------------------------
    print("Step 1: Loading structured covariates...")
    df = pd.read_csv(TVAE_CSV)
    X = encode_tvae(df)
    n = len(X)
    print(f"  X: {X.shape}")

    # -- 2. Build prognostic index from X --------------------------------------
    print("\nStep 2: Building prognostic index...")
    pca_x = PCA(n_components=1, random_state=0)
    z_prog_raw = pca_x.fit_transform(X).ravel()
    z_prog = ((z_prog_raw - z_prog_raw.mean()) / (z_prog_raw.std() + 1e-8)).astype(np.float32)
    prog_weights = pca_x.components_[0].astype(np.float32)
    print(f"  X PC1 explained variance: {pca_x.explained_variance_ratio_[0]:.1%}")

    # -- 3. Sample treatment modifier independently ----------------------------
    print("\nStep 3: Sampling z_treat ~ N(0,1)...")
    rng = np.random.default_rng(42)
    z_treat = rng.standard_normal(n).astype(np.float32)
    # Standardize (should already be ~N(0,1), but enforce exactly)
    z_treat = ((z_treat - z_treat.mean()) / (z_treat.std() + 1e-8)).astype(np.float32)
    print(f"  z_treat: mean={z_treat.mean():.6f}, std={z_treat.std():.6f}")
    print(f"  corr(z_prog, z_treat): {np.corrcoef(z_prog, z_treat)[0, 1]:.6f}")

    # -- 4. Generate one-factor embeddings -------------------------------------
    print(f"\nStep 4: Generating embeddings (d={P_EMBED}, sigma={SIGMA_EMBED})...")
    E = rng.normal(
        loc=z_treat[:, None],
        scale=SIGMA_EMBED,
        size=(n, P_EMBED),
    ).astype(np.float32)
    print(f"  E shape: {E.shape}")
    # Check: mean of E should recover z_treat
    z_hat = E.mean(axis=1)
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression().fit(z_hat.reshape(-1, 1), z_treat)
    r2 = lr.score(z_hat.reshape(-1, 1), z_treat)
    print(f"  R2(z_treat ~ mean(E)): {r2:.4f}")

    # -- 5. Build super-population ---------------------------------------------
    print("\nStep 5: Building super-population...")
    super_pop = build_super_population(X, E, z_prog, z_treat)

    # -- Save artifacts --------------------------------------------------------
    config = {
        "p_struct": int(X.shape[1]),
        "p_embed": P_EMBED,
        "alpha_0": ALPHA_0,
        "a_alpha": A_ALPHA,
        "lambda_base": LAMBDA_BASE,
        "a_lambda": A_LAMBDA,
        "gamma_intercept": GAMMA_INTERCEPT,
        "beta": BETA,
        "lambda_cens_base": LAMBDA_CENS_BASE,
        "b_lambda_cens": B_LAMBDA_CENS,
        "sigma_embed": SIGMA_EMBED,
        "t_end": T_END,
        "corr_z_prog_z_treat": float(np.corrcoef(z_prog, z_treat)[0, 1]),
        "n_subjects": n,
        "mechanism": "two-index-one-factor",
    }
    (ARTIFACT_DIR / "config.json").write_text(json.dumps(config, indent=2))

    np.savez_compressed(
        str(ARTIFACT_DIR / "super_population.npz"),
        X=super_pop["X"], E=super_pop["E"],
        lam0=super_pop["lam0"], lam1=super_pop["lam1"],
        D0=super_pop["D0"], D1=super_pop["D1"],
        C=super_pop["C"],
        alpha=super_pop["alpha"],
        lambda_c=super_pop["lambda_c"],
        prog_weights=prog_weights,
    )

    size_mb = (ARTIFACT_DIR / "super_population.npz").stat().st_size / 1e6
    elapsed = time.time() - t0
    print(f"\nDone! Artifacts saved to {ARTIFACT_DIR} ({size_mb:.1f} MB)")
    print(f"Total time: {elapsed:.1f}s")

    # -- Sanity checks ---------------------------------------------------------
    print("\n-- Sanity checks --")
    t_grid = np.linspace(0.01, T_END, 20)
    alpha_sample = super_pop["alpha"][:100]
    lam0_sample = super_pop["lam0"][:100]
    lam1_sample = super_pop["lam1"][:100]
    S0 = np.exp(-((t_grid[None, :] / lam0_sample[:, None]) ** alpha_sample[:, None]))
    S1 = np.exp(-((t_grid[None, :] / lam1_sample[:, None]) ** alpha_sample[:, None]))
    tau = S1 - S0
    mid = len(t_grid) // 2
    print(f"  CATE at t={t_grid[mid]:.1f}: "
          f"mean={tau[:, mid].mean():.4f}, std={tau[:, mid].std():.4f}")
    print(f"  CATE at t={t_grid[-1]:.1f}: "
          f"mean={tau[:, -1].mean():.4f}, std={tau[:, -1].std():.4f}")
    print(f"  |CATE| at t={t_grid[mid]:.1f}: mean={np.abs(tau[:, mid]).mean():.4f}")


if __name__ == "__main__":
    main()
