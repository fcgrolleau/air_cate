"""Quickstart example: estimate time-varying CATE on synthetic Weibull data.

Usage:
  uv run python examples/quickstart.py                  # fit and print
  uv run --extra plots python examples/quickstart.py    # also save the convergence plot
"""

import os

import numpy as np

from air_cate import AIREstimator

# Imported after air_cate, which sets KERAS_BACKEND=jax before Keras loads.
import keras  # noqa: E402

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")


def generate_data(n=1000, p=5, rng_seed=0):
    """Synthetic Weibull DGP with heterogeneous treatment effects."""
    rng = np.random.default_rng(rng_seed)
    X = rng.uniform(-1, 1, (n, p)).astype(np.float32)

    # Propensity: e(x) = sigmoid(0.5 * x_0)
    prob = 1.0 / (1.0 + np.exp(-0.5 * X[:, 0]))
    W = rng.binomial(1, prob).astype(np.float32)

    # Weibull event times: shape=1.5, scale depends on treatment and x_1
    shape = 1.5
    scale = np.exp(1.0 + 0.3 * X[:, 0] + 0.5 * W * X[:, 1])
    D = scale * (-np.log(rng.uniform(0, 1, n))) ** (1.0 / shape)

    # Independent censoring
    C = 5.0 * (-np.log(rng.uniform(0, 1, n)))
    Z = np.minimum(D, C).astype(np.float32)
    Delta = (D <= C).astype(np.float32)

    return X, W, Z, Delta


def true_cate(t, X):
    """Analytic true CATE for the Weibull DGP."""
    shape = 1.5
    s1 = np.exp(1.0 + 0.3 * X[:, 0] + 0.5 * X[:, 1])[:, None]
    s0 = np.exp(1.0 + 0.3 * X[:, 0])[:, None]
    return np.exp(-((t[None, :] / s1) ** shape)) - np.exp(-((t[None, :] / s0) ** shape))


def save_convergence_figure(est):
    """Save the CATE stage's training curve, if matplotlib is installed.

    Worth a look on every fit: the curve is what says whether `cate_epochs` was
    enough. AIREstimator early-stops on the training loss, which here descends
    essentially monotonically, so patience never fires and the run uses its full
    budget -- meaning a curve still sloping down at the right edge is a signal to
    raise the budget, not evidence that the optimizer had finished.
    """
    try:
        import matplotlib
    except ImportError:
        print("\nmatplotlib not installed; skipping the convergence figure.")
        print("Install with: uv sync --extra plots")
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, "quickstart_convergence.png")

    fig, _ = est.plot_convergence()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    n_epochs = len(est.history_["train_loss"])
    best = est.history_["best_epoch"] + 1  # history is 0-indexed
    print(f"\nRan {n_epochs} CATE epochs; best training loss at epoch {best}.")
    print(f"Saved: {path}")


def main():
    # Seed network initialization and Keras-side shuffling; `rng_seed` below seeds
    # the CATE minibatch order. Without both, the printed iMSE and the saved
    # convergence figure move from run to run.
    keras.utils.set_random_seed(0)

    print("Generating synthetic data...")
    X_train, W, Z, Delta = generate_data(n=1500, rng_seed=42)
    t_end = 5.0

    print("Fitting AIREstimator...")
    est = AIREstimator(
        K=4,
        M=50,
        n_folds=3,
        encoder_hidden_sizes=(32, 16),
        nuisance_training_config={"epochs": 50, "verbose": 0},
        cate_epochs=150,
        cate_lr=1e-3,
        rng_seed=0,
    )
    est.fit(X_train, W, Z, Delta, t_end, verbose=1)

    # Predict on test data
    rng = np.random.default_rng(99)
    X_test = rng.uniform(-1, 1, (200, X_train.shape[1])).astype(np.float32)

    tau_hat = est.predict_tau(X_test)
    gamma_hat = est.predict_gamma(X_test)
    t_grid = np.array(est.time_grid_)

    # Compare with truth
    tau_true = true_cate(t_grid, X_test)
    imse = np.mean(np.sum(est.delta_ * (tau_hat - tau_true) ** 2, axis=1))
    print(f"\nIntegrated MSE: {imse:.4f}")
    print(f"tau_hat shape: {tau_hat.shape}")
    print(f"gamma_hat shape: {gamma_hat.shape}")

    # Show CATE for a sample subject
    idx = 0
    print(f"\nSubject {idx} (x_1 = {X_test[idx, 1]:.2f}):")
    for m in [0, 10, 25, 40, 50]:
        print(f"  t={t_grid[m]:.1f}: tau_hat={tau_hat[idx, m]:.4f}, tau_true={tau_true[idx, m]:.4f}")

    save_convergence_figure(est)


if __name__ == "__main__":
    main()
