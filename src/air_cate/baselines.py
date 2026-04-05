"""Baseline CATE estimators using Random Survival Forests.

T-learner: fit separate forests per arm, tau = S_1 - S_0.
S-learner: fit one forest with treatment as a feature, tau = S(x,1) - S(x,0).
"""

import numpy as np
from sksurv.ensemble import RandomSurvivalForest

from air_cate.utils import build_time_grid


def _make_surv_y(Z, Delta):
    """Convert (Z, Delta) arrays to scikit-survival structured array."""
    return np.array(
        [(bool(d), float(z)) for d, z in zip(Delta, Z)],
        dtype=[("event", bool), ("time", float)],
    )


def _predict_survival_on_grid(forest, X, time_grid):
    """Predict survival curves from a fitted RSF and evaluate on a time grid.

    Returns:
        S: (n, M+1) survival probabilities on the grid.
    """
    surv_fns = forest.predict_survival_function(X)
    t_grid = np.asarray(time_grid)
    n = X.shape[0]
    T = len(t_grid)
    S = np.ones((n, T), dtype=np.float32)
    for i, fn in enumerate(surv_fns):
        S[i] = fn(t_grid)
    return S


class SurvivalForestTLearner:
    """T-learner: two separate Random Survival Forests (one per arm).

    tau_hat(t|x) = S_hat_1(t|x) - S_hat_0(t|x)

    Matches the AIREstimator interface (fit / predict_tau / time_grid_ / delta_).
    """

    def __init__(self, M=30, n_estimators=100, max_depth=None, min_samples_leaf=15,
                 random_state=None):
        self.M = M
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.forest_0_ = None
        self.forest_1_ = None
        self.time_grid_ = None
        self.delta_ = None

    def fit(self, X, W, Z, Delta, t_end, nuisances=None, verbose=0):
        self.time_grid_, self.delta_ = build_time_grid(t_end, self.M)

        rsf_kwargs = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )

        mask0 = W.ravel() == 0
        mask1 = W.ravel() == 1

        self.forest_0_ = RandomSurvivalForest(**rsf_kwargs)
        self.forest_0_.fit(X[mask0], _make_surv_y(Z[mask0], Delta[mask0]))

        self.forest_1_ = RandomSurvivalForest(**rsf_kwargs)
        self.forest_1_.fit(X[mask1], _make_surv_y(Z[mask1], Delta[mask1]))

        return self

    def predict_tau(self, X):
        t_grid = np.asarray(self.time_grid_)
        S1 = _predict_survival_on_grid(self.forest_1_, X, t_grid)
        S0 = _predict_survival_on_grid(self.forest_0_, X, t_grid)
        return S1 - S0


class SurvivalForestSLearner:
    """S-learner: one Random Survival Forest with treatment as a covariate.

    tau_hat(t|x) = S_hat(t | x, W=1) - S_hat(t | x, W=0)

    Matches the AIREstimator interface (fit / predict_tau / time_grid_ / delta_).
    """

    def __init__(self, M=30, n_estimators=100, max_depth=None, min_samples_leaf=15,
                 random_state=None):
        self.M = M
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.forest_ = None
        self.time_grid_ = None
        self.delta_ = None

    def fit(self, X, W, Z, Delta, t_end, nuisances=None, verbose=0):
        self.time_grid_, self.delta_ = build_time_grid(t_end, self.M)

        X_aug = np.column_stack([X, W.ravel().astype(np.float32)])

        self.forest_ = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.forest_.fit(X_aug, _make_surv_y(Z, Delta))

        return self

    def predict_tau(self, X):
        t_grid = np.asarray(self.time_grid_)
        n = X.shape[0]

        X1 = np.column_stack([X, np.ones(n, dtype=np.float32)])
        X0 = np.column_stack([X, np.zeros(n, dtype=np.float32)])

        S1 = _predict_survival_on_grid(self.forest_, X1, t_grid)
        S0 = _predict_survival_on_grid(self.forest_, X0, t_grid)
        return S1 - S0
