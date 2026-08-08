"""IST-3 Text-Enriched Data-Generating Process (two-index design).

Covariates: structured X (p=10) from IST-3 TVAE + one-factor embeddings E (p=20).

Two latent indices:
  z_prog  = standardize(PC1(X))           (prognostic index from X)
  z_treat ~ N(0,1) independent of X       (treatment modifier, planted in E)

Embeddings: E_j | z_treat ~ iid N(z_treat, sigma^2) — one-factor measurement model.

Survival mechanism (covariate-dependent shape and scale):
  alpha(x)  = exp(log(alpha_0) + a_alpha * z_prog)
  lam0(x)   = lambda_base * exp(-a_lambda * z_prog)
  lam1(x,e) = lam0(x) * exp(gamma_0 + beta * z_treat)
  S_w(t|x,e) = exp(-(t / lam_w)^{alpha(x)})

Censoring (covariate-dependent exponential):
  lambda_c(x) = lambda_c_base * exp(-b_lambda * z_prog)
  C | X ~ Exp(1/lambda_c(x))

Treatment assignment: balanced (e(x)=0.5), matching IST-3's 1:1 randomisation.

All per-subject parameters are pre-computed and stored in
artifacts_text/super_population.npz — only closed-form Weibull survival
is needed at runtime.
"""

import json

import numpy as np
from pathlib import Path

ARTIFACT_DIR = Path(__file__).parent / "artifacts_text"


class IST3TextDGP:
    """Two-index text-enriched DGP with covariate-dependent shape and censoring.

    AIR methods receive X_full = [X, E] (p_struct + p_embed features).
    RSF baselines receive X_struct = X_full[:, :p_struct] (structured only).

    Baseline survival depends on z_prog (from X only) through both shape
    alpha(x) and scale lam0(x). Treatment effect heterogeneity depends on
    z_treat (planted in E via one-factor model).

    Censoring is exponential with covariate-dependent rate:
      lambda_c(x) = lambda_c_base * exp(-b_lambda * z_prog)
      C | X ~ Exp(1/lambda_c(x))

    Pre-computed per-subject parameters (alpha, lam0, lam1, lambda_c)
    are stored in the super-population. true_tau() and oracle nuisance
    methods use stored pool indices — no model evaluation at runtime.
    """

    t_end = 5.0
    cens_scale = 5.0  # kept for backwards compat; actual censoring is per-subject
    p_struct = 10  # first p_struct columns of X_full are structured covariates

    # Class-level state (loaded once)
    _super_pop = None
    _config = None
    _loaded = False
    _train_pool_idx = None  # set by generate(), used by oracle nuisances
    _test_pool_idx = None   # set by generate_test_X(), used by true_tau

    @classmethod
    def load(cls, artifact_dir=None):
        """Load pre-computed super-population and DGP configuration."""
        adir = Path(artifact_dir) if artifact_dir else ARTIFACT_DIR

        with open(adir / "config.json") as f:
            cls._config = json.load(f)

        cls.t_end = cls._config["t_end"]
        cls.p_struct = cls._config["p_struct"]

        data = np.load(str(adir / "super_population.npz"))
        cls._super_pop = {
            "X": data["X"],              # (N, p_struct)
            "E": data["E"],              # (N, p_embed)
            "lam0": data["lam0"],        # (N,) control-arm Weibull scale
            "lam1": data["lam1"],        # (N,) treated-arm Weibull scale
            "D0": data["D0"],            # (N,) pre-drawn control event times
            "D1": data["D1"],            # (N,) pre-drawn treated event times
            "C": data["C"],              # (N,) pre-drawn censoring times
            "alpha": data["alpha"],      # (N,) per-subject event Weibull shape
            "lambda_c": data["lambda_c"],  # (N,) per-subject censoring scale
        }

        n = len(cls._super_pop["X"])
        p_embed = cls._super_pop["E"].shape[1]
        cls._loaded = True
        print(f"IST3TextDGP loaded: {n} subjects, "
              f"p_struct={cls.p_struct}, p_embed={p_embed}, "
              f"p_total={cls.p_struct + p_embed}")

    @classmethod
    def propensity(cls, X):
        """P(W=1|X) = 0.5 (balanced RCT). Returns (n,) array."""
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full(n, 0.5, dtype=np.float32)

    @staticmethod
    def _weibull_survival(t, k, lam):
        """S(t|k,lam) = exp(-(t/lam)^k). Handles grid and pointwise.

        Args:
            t: (T,) time grid or (n,) per-subject times
            k: (n,) per-subject shape parameter
            lam: (n,) per-subject scale

        Returns:
            (n, T) if t is a grid, (n,) if t is per-subject
        """
        t = np.asarray(t, dtype=np.float32)
        k = np.asarray(k, dtype=np.float32)
        if t.ndim == 1 and len(t) != len(lam):
            # Grid: t is (T,), k is (n,), lam is (n,) -> (n, T)
            log_ratio = (np.log(np.maximum(t[None, :], 1e-30))
                         - np.log(lam[:, None]))
            return np.exp(-np.exp(k[:, None] * log_ratio)).astype(np.float32)
        else:
            # Pointwise: t, k, lam all (n,) -> (n,)
            log_ratio = (np.log(np.maximum(t, 1e-30))
                         - np.log(lam))
            return np.exp(-np.exp(k * log_ratio)).astype(np.float32)

    # ── Oracle event survival methods (use _train_pool_idx) ──────────────

    @classmethod
    def _control_survival_grid(cls, t_grid, X):
        """S_0(t|X,E) on grid. Returns (n, T)."""
        idx = cls._train_pool_idx
        lam0 = cls._super_pop["lam0"][idx]
        alpha = cls._super_pop["alpha"][idx]
        return cls._weibull_survival(t_grid, alpha, lam0)

    @classmethod
    def _control_survival_pointwise(cls, t, X):
        """S_0(t_i|X_i,E_i) per subject. Returns (n,)."""
        idx = cls._train_pool_idx
        lam0 = cls._super_pop["lam0"][idx]
        alpha = cls._super_pop["alpha"][idx]
        return cls._weibull_survival(t, alpha, lam0)

    @classmethod
    def _treated_survival_grid(cls, t_grid, X):
        """S_1(t|X,E) on grid. Returns (n, T)."""
        idx = cls._train_pool_idx
        lam1 = cls._super_pop["lam1"][idx]
        alpha = cls._super_pop["alpha"][idx]
        return cls._weibull_survival(t_grid, alpha, lam1)

    @classmethod
    def _treated_survival_pointwise(cls, t, X):
        """S_1(t_i|X_i,E_i) per subject. Returns (n,)."""
        idx = cls._train_pool_idx
        lam1 = cls._super_pop["lam1"][idx]
        alpha = cls._super_pop["alpha"][idx]
        return cls._weibull_survival(t, alpha, lam1)

    # ── Oracle censoring survival methods (use _train_pool_idx) ──────────

    @classmethod
    def _censoring_survival_grid(cls, t_grid, X):
        """S_C(t|x) = exp(-t/lambda_c(x)) on grid. Returns (n, T)."""
        idx = cls._train_pool_idx
        lam_c = cls._super_pop["lambda_c"][idx]
        t_grid = np.asarray(t_grid, dtype=np.float32)
        return np.exp(-t_grid[None, :] / lam_c[:, None]).astype(np.float32)

    @classmethod
    def _censoring_survival_pointwise(cls, t, X):
        """S_C(t_i|x_i) = exp(-t_i/lambda_c(x_i)) per subject. Returns (n,)."""
        idx = cls._train_pool_idx
        lam_c = cls._super_pop["lambda_c"][idx]
        t = np.asarray(t, dtype=np.float32)
        return np.exp(-t / lam_c).astype(np.float32)

    # ── Public API ────────────────────────────────────────────────────────

    @classmethod
    def true_tau(cls, t, X_full):
        """True CATE: tau(t|X,E) = S_1(t|X,E) - S_0(t|X,E). Returns (n, T).

        Uses _test_pool_idx set by generate_test_X().
        """
        if not cls._loaded:
            cls.load()
        t = np.atleast_1d(np.asarray(t, dtype=np.float32))

        idx = cls._test_pool_idx
        lam0 = cls._super_pop["lam0"][idx]
        lam1 = cls._super_pop["lam1"][idx]
        alpha = cls._super_pop["alpha"][idx]

        S0 = cls._weibull_survival(t, alpha, lam0)
        S1 = cls._weibull_survival(t, alpha, lam1)
        return S1 - S0

    @classmethod
    def generate(cls, n, p=None, t_end=None, rng_seed=0):
        """Sample n observations from the text-enriched IST-3 DGP.

        Returns:
            X_full: (n, p_struct + p_embed) float32 — [structured, embeddings]
            W: (n,) float32 treatment indicator {0, 1}
            Z: (n,) float32 observed time
            Delta: (n,) float32 event indicator {0, 1}
            t_end: float
        """
        if not cls._loaded:
            cls.load()
        if t_end is None:
            t_end = cls.t_end

        rng = np.random.default_rng(rng_seed)
        pool_size = len(cls._super_pop["X"])
        idx = rng.integers(0, pool_size, size=n)

        # Store for oracle nuisance lookups
        cls._train_pool_idx = idx

        X = cls._super_pop["X"][idx]
        E = cls._super_pop["E"][idx]
        X_full = np.hstack([X, E]).astype(np.float32)

        # Balanced treatment assignment
        W = rng.binomial(1, 0.5, n).astype(np.float32)

        # Per-subject parameters
        alpha = cls._super_pop["alpha"][idx]
        lam0 = cls._super_pop["lam0"][idx]
        lam1 = cls._super_pop["lam1"][idx]

        # Event times: D_w = lam_w * (-ln U)^{1/alpha}
        U = rng.uniform(1e-10, 1.0 - 1e-10, n)
        neg_log_U = -np.log(U)
        D0 = (lam0 * neg_log_U ** (1.0 / alpha)).astype(np.float32)
        D1 = (lam1 * neg_log_U ** (1.0 / alpha)).astype(np.float32)
        D = np.where(W == 1, D1, D0)

        # Covariate-dependent exponential censoring: C ~ Exp(1/lambda_c(x))
        lambda_c = cls._super_pop["lambda_c"][idx]
        C = (rng.exponential(1.0, n) * lambda_c).astype(np.float32)

        # Observed data
        Z = np.minimum(D, np.minimum(C, t_end)).astype(np.float32)
        Delta = (D < np.minimum(C, t_end)).astype(np.float32)

        return X_full, W, Z, Delta, t_end

    @classmethod
    def generate_test_X(cls, n_test, rng_seed=9999):
        """Sample test covariates from the super-population.

        Stores pool indices internally for true_tau() lookups.
        """
        if not cls._loaded:
            cls.load()
        rng = np.random.default_rng(rng_seed)
        pool_size = len(cls._super_pop["X"])
        idx = rng.integers(0, pool_size, size=n_test)

        cls._test_pool_idx = idx

        X = cls._super_pop["X"][idx]
        E = cls._super_pop["E"][idx]
        return np.hstack([X, E]).astype(np.float32)
