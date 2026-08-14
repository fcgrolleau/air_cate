"""Causal survival forest baseline (grf), refit at every horizon.

The causal survival forest estimates a treatment effect at one time point: with
grf's ``target = "survival.probability"`` and ``horizon = t`` it targets
P[T(1) > t | X] - P[T(0) > t | X], which is exactly the time-fixed CATE
tau_t(x). Building the whole tv-CATE curve out of it therefore means refitting the
forest at every point of the time grid -- the per-horizon cost that motivates the
paper's `Pointwise horizon targets` discussion.

The loop over horizons lives in the companion R script, so one fitted model costs
one R process rather than one per horizon. This module is importable without R;
the subprocess is only spawned by ``predict_tau``.

grf hyperparameters are left at their defaults, including ``W.hat = NULL`` so the
forest estimates its own propensity -- the simulation DGPs have
e(x) = sigmoid(0.5 x_1), not a constant, so pinning it would be wrong.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np

from air_cate.utils import build_time_grid

_R_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines_csf.R")

# R's integers are 32-bit. Seeds are drawn below this less the grid size, so that
# the per-horizon offset the R script adds cannot overflow.
_R_INT_MAX = 2**31 - 1


def grf_available(rscript="Rscript"):
    """True when Rscript is on PATH and the grf package loads.

    Intended for pytest skip guards, so a machine without R still runs the suite.
    """
    if shutil.which(rscript) is None:
        return False
    try:
        proc = subprocess.run(
            [rscript, "-e", 'cat(requireNamespace("grf", quietly = TRUE))'],
            capture_output=True, text=True, timeout=300,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return proc.returncode == 0 and "TRUE" in proc.stdout


class CausalSurvivalForest:
    """grf causal survival forest, one fit per horizon of the time grid.

    tau_hat(t|x) = P[T(1) > t | x] - P[T(0) > t | x], estimated separately at each
    t_m, with tau_hat(0|x) = 0 imposed exactly.

    Matches the AIREstimator interface (fit / predict_tau / time_grid_ / delta_).

    Predictions are deliberately not clipped to [-1, 1]: the causal survival forest
    is an orthogonalized estimator with no range guarantee, and clipping would hide
    the structural-validity gap the paper documents. The excursion is recorded in
    ``frac_out_of_range_`` instead.
    """

    def __init__(self, M=30, num_trees=2000, num_threads=1, rscript="Rscript",
                 seed=None, timeout=None):
        self.M = M
        self.num_trees = num_trees
        self.num_threads = num_threads
        self.rscript = rscript
        self.seed = seed
        self.timeout = timeout
        self.time_grid_ = None
        self.delta_ = None
        # Diagnostics, populated by predict_tau.
        self.n_failed_horizons_ = None
        self.frac_out_of_range_ = None

    def fit(self, X, W, Z, Delta, t_end, nuisances=None, verbose=0):
        """Record the training sample; the forests are fit in ``predict_tau``.

        Nothing is fit here because keeping this to a single R process means the
        test covariates have to travel in the same call, and they are not known
        until predict time. This does not distort the reported runtime:
        ``run_single`` in examples/plot_simulation.py stops its clock after
        ``predict_tau``, so the fitting cost is timed either way.

        ``nuisances`` and ``verbose`` are accepted for interface compatibility;
        grf estimates its own nuisances.
        """
        self.time_grid_, self.delta_ = build_time_grid(t_end, self.M)
        self._X = np.asarray(X, dtype=np.float64)
        self._W = np.asarray(W, dtype=np.float64).ravel()
        self._Z = np.asarray(Z, dtype=np.float64).ravel()
        self._Delta = np.asarray(Delta, dtype=np.float64).ravel()
        self._verbose = verbose
        # Drawn from the global numpy RNG, which the simulation harness has already
        # seeded via keras.utils.set_random_seed. That makes grf inherit the
        # replication's seed without make_estimator having to pass one in -- it
        # receives only the config, which is why the RSF baselines run unseeded.
        self._r_seed = (
            int(np.random.randint(0, _R_INT_MAX - self.M - 1))
            if self.seed is None else int(self.seed)
        )
        return self

    def predict_tau(self, X):
        """Return tau_hat on the time grid, shape (n, M+1)."""
        if self.time_grid_ is None:
            raise RuntimeError("fit must be called before predict_tau")

        X_test = np.asarray(X, dtype=np.float64)
        if X_test.shape[1] != self._X.shape[1]:
            raise ValueError(
                f"test covariates have {X_test.shape[1]} columns, "
                f"training had {self._X.shape[1]}"
            )

        t_grid = np.asarray(self.time_grid_, dtype=np.float64)
        # t_0 = 0 is never fit: S_1(0) = S_0(0) = 1 makes tau(0|x) = 0 analytically,
        # and grf rejects horizon = 0.
        horizons = t_grid[1:]

        with tempfile.TemporaryDirectory(prefix="air_csf_") as workdir:
            self._write_inputs(workdir, X_test, horizons)
            self._run_r(workdir)
            raw = self._read_predictions(workdir, X_test.shape[0], horizons.size)

        # Whole columns fail together, so counting one row's NaNs counts horizons.
        self.n_failed_horizons_ = int(np.isnan(raw[0]).sum())

        tau = np.zeros((X_test.shape[0], self.M + 1), dtype=np.float32)
        tau[:, 1:] = self._hold_last_fitted(raw)
        self.frac_out_of_range_ = float(np.mean(np.abs(tau) > 1.0))
        return tau

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _hold_last_fitted(preds):
        """Repair horizons grf could not fit by holding the last one it could.

        grf refuses a horizon beyond the last observed failure time, which happens
        in the tail of small samples. Holding the last successful estimate is the
        same flat-tail convention the RSF baselines already use past the end of
        their step functions (see _predict_survival_on_grid in baselines.py). A
        leading run of failures has nothing to hold and falls back to 0, which is
        the correct limit as t -> 0.
        """
        out = np.array(preds, dtype=np.float64, copy=True)
        for j in range(out.shape[1]):
            bad = np.isnan(out[:, j])
            if bad.any():
                out[bad, j] = out[bad, j - 1] if j > 0 else 0.0
        return out

    def _write_inputs(self, workdir, X_test, horizons):
        p = self._X.shape[1]
        x_names = [f"x{i + 1}" for i in range(p)]
        train = np.column_stack([self._X, self._W, self._Z, self._Delta])
        np.savetxt(
            os.path.join(workdir, "train.csv"), train, delimiter=",",
            header=",".join(x_names + ["W", "Z", "Delta"]), comments="",
        )
        np.savetxt(
            os.path.join(workdir, "test.csv"), X_test, delimiter=",",
            header=",".join(x_names), comments="",
        )
        np.savetxt(
            os.path.join(workdir, "horizons.csv"), horizons, delimiter=",",
            header="horizon", comments="",
        )

    def _run_r(self, workdir):
        cmd = [
            self.rscript, _R_SCRIPT, workdir,
            str(int(self.num_trees)), str(int(self.num_threads)), str(self._r_seed),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=self.timeout)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"{self.rscript!r} not found on PATH. The CSF baseline needs R with "
                'the grf package: install.packages("grf")'
            ) from exc
        # R's message() writes to stderr; that is where the per-horizon failure
        # notes and the fitted-count summary arrive.
        if self._verbose and proc.stderr:
            print(proc.stderr, end="")
        if proc.returncode != 0:
            raise RuntimeError(
                f"grf causal_survival_forest failed (exit {proc.returncode}):\n"
                f"{proc.stderr[-4000:]}"
            )

    @staticmethod
    def _read_predictions(workdir, n_test, n_horizons):
        path = os.path.join(workdir, "predictions.csv")
        if not os.path.exists(path):
            raise RuntimeError(f"R produced no predictions at {path}")
        raw = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        if raw.shape != (n_test, n_horizons):
            raise RuntimeError(
                f"expected predictions of shape {(n_test, n_horizons)}, "
                f"got {raw.shape}"
            )
        return raw
