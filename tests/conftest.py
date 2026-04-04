"""Shared fixtures and synthetic DGPs for testing."""

import numpy as np
import pytest


class WeibullDGP:
    """DGP 1: Simple Weibull with heterogeneous treatment effects.

    X ~ Uniform(-1, 1)^p
    e(x) = sigmoid(0.5 * x_0)
    W ~ Bernoulli(e(X))
    D | X, W=w ~ Weibull(shape=1.5, scale=exp(1 + 0.3*x_0 + 0.5*w*x_1))
    C | X ~ Weibull(shape=1.0, scale=5.0)

    True CATE: tau(t|x) = S_1(t|x) - S_0(t|x) where S_w = exp(-(t/scale_w)^shape)
    """

    shape = 1.5
    cens_scale = 5.0

    @staticmethod
    def propensity(X):
        return 1.0 / (1.0 + np.exp(-0.5 * X[:, 0]))

    @staticmethod
    def scale(X, w):
        return np.exp(1.0 + 0.3 * X[:, 0] + 0.5 * w * X[:, 1])

    @classmethod
    def true_tau(cls, t, X):
        """True CATE at time(s) t for covariates X.

        Args:
            t: (T,) time grid or scalar.
            X: (n, p) covariates.

        Returns:
            (n, T) CATE values.
        """
        t = np.atleast_1d(t)
        s1 = cls.scale(X, 1)[:, None]  # (n, 1)
        s0 = cls.scale(X, 0)[:, None]  # (n, 1)
        S1 = np.exp(-((t[None, :] / s1) ** cls.shape))
        S0 = np.exp(-((t[None, :] / s0) ** cls.shape))
        return S1 - S0

    @classmethod
    def generate(cls, n, p=5, t_end=5.0, rng_seed=0):
        rng = np.random.default_rng(rng_seed)
        X = rng.uniform(-1, 1, (n, p)).astype(np.float32)
        prob = cls.propensity(X)
        W = rng.binomial(1, prob).astype(np.float32)

        scale_w = cls.scale(X, W)
        D = scale_w * (-np.log(rng.uniform(0, 1, n))) ** (1.0 / cls.shape)
        C = cls.cens_scale * (-np.log(rng.uniform(0, 1, n)))  # Exponential
        Z = np.minimum(D, C).astype(np.float32)
        Delta = (D <= C).astype(np.float32)

        return X, W, Z, Delta, t_end


class WeibullMixtureDGP:
    """DGP 3: 3-component Weibull mixture survival in each arm.

    X ~ Uniform(-1, 1)^p
    e(x) = sigmoid(0.5 * x_0)
    W ~ Bernoulli(e(X))

    Each arm's survival is a 3-component Weibull mixture:
        S_w(t|x) = sum_k pi_k^(w)(x) * exp(-(t / lam_k^(w)(x))^alpha_k)

    Shapes:  alpha = (1.0, 2.0, 3.0)
    Base scales:  mu = (0.8, 1.2, 1.8)

    Control arm (w=0):
        pi^(0) = (1/3, 1/3, 1/3)
        lam_k^(0)(x) = exp(mu_k + 0.2*x_0)

    Treatment arm (w=1):
        pi^(1)(x) = softmax(0.8*x_1, 0, -0.8*x_1)
        lam_k^(1)(x) = exp(mu_k + 0.2*x_0 + gamma_k*x_1)
        gamma = (0.5, 0.0, -0.3)

    True CATE: tau(t|x) = S_1(t|x) - S_0(t|x)
    """

    shapes = np.array([1.0, 2.0, 3.0])
    mu = np.array([0.8, 1.2, 1.8])
    gamma = np.array([0.5, 0.0, -0.3])
    cens_scale = 5.0

    @staticmethod
    def propensity(X):
        return 1.0 / (1.0 + np.exp(-0.5 * X[:, 0]))

    @classmethod
    def _scales(cls, X, w):
        """Component scales (n, 3) for arm w.  w can be scalar or (n,) array."""
        x0 = X[:, 0:1]  # (n, 1)
        x1 = X[:, 1:2]  # (n, 1)
        w = np.asarray(w)
        if w.ndim == 0:
            w_col = float(w)
        else:
            w_col = w[:, None]  # (n, 1)
        log_scales = cls.mu[None, :] + 0.2 * x0 + w_col * cls.gamma[None, :] * x1
        return np.exp(log_scales)  # (n, 3)

    @classmethod
    def _weights(cls, X, w):
        """Mixture weights (n, 3) for arm w.  w can be scalar or (n,) array."""
        n = X.shape[0]
        w = np.asarray(w)
        if w.ndim == 0:
            # Scalar: all subjects in the same arm
            if float(w) == 0:
                return np.full((n, 3), 1.0 / 3.0)
            logits = np.stack([0.8 * X[:, 1], np.zeros(n), -0.8 * X[:, 1]], axis=1)
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            return exp_logits / exp_logits.sum(axis=1, keepdims=True)
        # Per-subject w vector: blend between control (uniform) and treated weights
        treated_logits = np.stack([0.8 * X[:, 1], np.zeros(n), -0.8 * X[:, 1]], axis=1)
        treated_logits -= treated_logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(treated_logits)
        treated_weights = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        control_weights = np.full((n, 3), 1.0 / 3.0)
        w_col = w[:, None]  # (n, 1)
        return (1 - w_col) * control_weights + w_col * treated_weights

    @classmethod
    def _survival(cls, t, X, w):
        """Mixture survival S_w(t|x), shape (n, T)."""
        t = np.atleast_1d(t)
        scales = cls._scales(X, w)    # (n, 3)
        weights = cls._weights(X, w)  # (n, 3)
        S = np.zeros((X.shape[0], len(t)))
        for k in range(3):
            lam_k = scales[:, k:k+1]         # (n, 1)
            pi_k = weights[:, k:k+1]         # (n, 1)
            S += pi_k * np.exp(-((t[None, :] / lam_k) ** cls.shapes[k]))
        return S

    @classmethod
    def true_tau(cls, t, X):
        """True CATE: S_1(t|x) - S_0(t|x), shape (n, T)."""
        return cls._survival(t, X, 1) - cls._survival(t, X, 0)

    @classmethod
    def generate(cls, n, p=5, t_end=5.0, rng_seed=0):
        rng = np.random.default_rng(rng_seed)
        X = rng.uniform(-1, 1, (n, p)).astype(np.float32)
        prob = cls.propensity(X)
        W = rng.binomial(1, prob).astype(np.float32)

        # Sample component per subject
        weights = cls._weights(X, W)   # (n, 3)
        cumprob = np.cumsum(weights, axis=1)
        u_comp = rng.uniform(0, 1, n)
        components = (u_comp[:, None] >= cumprob).sum(axis=1).astype(int)  # (n,)

        # Per-subject shape and scale from selected component
        scales = cls._scales(X, W)     # (n, 3)
        shape_i = cls.shapes[components]                       # (n,)
        scale_i = scales[np.arange(n), components]             # (n,)

        # Weibull event time: D = scale * (-log(U))^(1/shape)
        D = scale_i * (-np.log(rng.uniform(0, 1, n))) ** (1.0 / shape_i)

        # Independent censoring
        C = cls.cens_scale * (-np.log(rng.uniform(0, 1, n)))
        Z = np.minimum(D, C).astype(np.float32)
        Delta = (D <= C).astype(np.float32)

        return X, W, Z, Delta, t_end


class NullDGP:
    """DGP 2: Null treatment effect.

    X ~ Uniform(-1, 1)^p
    e(x) = 0.5
    D | X ~ Weibull(shape=2, scale=3)  (same for both arms)
    C ~ Exponential(rate=0.2)

    True CATE: tau(t|x) = 0 for all t, x.
    """

    shape = 2.0
    scale = 3.0
    cens_rate = 0.2

    @classmethod
    def true_tau(cls, t, X):
        t = np.atleast_1d(t)
        return np.zeros((X.shape[0], len(t)), dtype=np.float32)

    @classmethod
    def generate(cls, n, p=5, t_end=5.0, rng_seed=0):
        rng = np.random.default_rng(rng_seed)
        X = rng.uniform(-1, 1, (n, p)).astype(np.float32)
        W = rng.binomial(1, 0.5, n).astype(np.float32)

        D = cls.scale * (-np.log(rng.uniform(0, 1, n))) ** (1.0 / cls.shape)
        C = rng.exponential(1.0 / cls.cens_rate, n)
        Z = np.minimum(D, C).astype(np.float32)
        Delta = (D <= C).astype(np.float32)

        return X, W, Z, Delta, t_end


@pytest.fixture
def weibull_dgp_data():
    return WeibullDGP.generate(n=1500, rng_seed=42)


@pytest.fixture
def null_dgp_data():
    return NullDGP.generate(n=1000, rng_seed=42)
