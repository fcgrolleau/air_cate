"""Plot how iMSE against the true CATE evolves during CATE model training.

Runs stages 1-2 (nuisance + pseudo-outcomes) once, then trains the CATE model
for many epochs, evaluating iMSE on a held-out covariate grid at each epoch.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import jax
import jax.numpy as jnp
import keras

from air_cate._types import SurvivalData
from air_cate.cate_model.mixture import CATEMixture
from air_cate.nuisance.cross_fit import cross_fit_nuisances
from air_cate.pseudo_outcome import compute_pseudo_outcomes
from air_cate.r_loss import r_loss as r_loss_fn
from air_cate.utils import build_time_grid
from conftest import WeibullMixtureDGP

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# --- Hyperparameters ---
N_TRAIN = 5000
K = 3
M = 50
N_FOLDS = 3
EPS = 0.01
ENCODER_HIDDEN = (64, 32)
NUISANCE_EPOCHS = 80
CATE_EPOCHS = 300
CATE_LR = 5e-4
BATCH_SIZE = 256
WEIGHT_DECAY = 1e-4

# Evaluation grid: 25x25 = 625 points in (x_0, x_1), rest = 0
EVAL_GRID_N = 25


def build_eval_grid(time_grid):
    """Build a (x_0, x_1) meshgrid for iMSE evaluation."""
    x0 = np.linspace(-0.9, 0.9, EVAL_GRID_N)
    x1 = np.linspace(-0.9, 0.9, EVAL_GRID_N)
    X0, X1 = np.meshgrid(x0, x1, indexing="ij")
    n_grid = EVAL_GRID_N * EVAL_GRID_N
    X_grid = np.zeros((n_grid, 5), dtype=np.float32)
    X_grid[:, 0] = X0.ravel()
    X_grid[:, 1] = X1.ravel()
    tau_true = WeibullMixtureDGP.true_tau(np.array(time_grid), X_grid)
    return X_grid, tau_true


def compute_imse(cate_model, X_grid, tau_true, time_grid, delta):
    """Evaluate mean integrated MSE on the grid."""
    X_jax = jnp.array(X_grid, dtype=jnp.float32)
    tau_hat = np.asarray(cate_model({"X": X_jax, "time_grid": time_grid}))
    pointwise_se = (tau_hat - tau_true) ** 2
    imse_per_subject = np.sum(delta * pointwise_se, axis=1)
    return float(np.mean(imse_per_subject))


def main():
    np.random.seed(0)

    # --- Stage 1-2: Nuisance estimation + pseudo-outcomes ---
    print(f"Generating training data (n={N_TRAIN}, WeibullMixtureDGP)...")
    X_train, W, Z, Delta, t_end = WeibullMixtureDGP.generate(n=N_TRAIN, rng_seed=42)

    time_grid, delta = build_time_grid(t_end, M)
    data = SurvivalData(
        X=jnp.array(X_train, dtype=jnp.float32),
        W=jnp.array(W, dtype=jnp.float32),
        Z=jnp.array(Z, dtype=jnp.float32),
        Delta=jnp.array(Delta, dtype=jnp.float32),
    )

    print("Cross-fitting nuisance models...")
    nuisances = cross_fit_nuisances(
        data, time_grid, n_folds=N_FOLDS,
        training_config={"epochs": NUISANCE_EPOCHS, "verbose": 0},
    )

    print("Computing pseudo-outcomes...")
    Y_tilde, S_marg, e_clipped = compute_pseudo_outcomes(
        data, nuisances, time_grid, delta, eps=EPS,
    )

    # --- Prepare training data ---
    n = X_train.shape[0]
    X_jax = jnp.array(X_train)
    time_grid_jax = jnp.array(time_grid)
    Y_tilde_jax = jnp.array(Y_tilde)
    S_marg_jax = jnp.array(S_marg)
    W_jax = jnp.array(W)
    e_clipped_jax = jnp.array(e_clipped)

    # --- Prepare evaluation grid ---
    X_eval, tau_true_eval = build_eval_grid(time_grid)

    # --- Build CATE model ---
    print(f"Building CATEMixture (K={K})...")
    cate_model = CATEMixture(K=K, encoder_hidden_sizes=ENCODER_HIDDEN)
    cate_model({"X": X_jax[:1], "time_grid": time_grid_jax})  # build

    optimizer = keras.optimizers.AdamW(
        learning_rate=CATE_LR, weight_decay=WEIGHT_DECAY,
    )
    optimizer.build(cate_model.trainable_variables)

    def _extract_vars(keras_vars):
        return [jnp.array(v.value) for v in keras_vars]

    def _assign_vars(keras_vars, values):
        for var, val in zip(keras_vars, values):
            var.assign(val)

    @jax.jit
    def train_step(trainable_vars, non_trainable_vars, opt_vars,
                   X_batch, Y_tilde_batch, S_marg_batch, W_batch, e_batch):
        def loss_fn(trainable_vars):
            tau_pred, _ = cate_model.stateless_call(
                trainable_vars, non_trainable_vars,
                {"X": X_batch, "time_grid": time_grid_jax},
            )
            return r_loss_fn(tau_pred, Y_tilde_batch, S_marg_batch,
                             W_batch, e_batch, delta)
        loss, grads = jax.value_and_grad(loss_fn)(trainable_vars)
        trainable_vars, opt_vars = optimizer.stateless_apply(
            opt_vars, grads, trainable_vars,
        )
        return loss, trainable_vars, opt_vars

    # --- Training loop with iMSE tracking ---
    print(f"Training for {CATE_EPOCHS} epochs (no early stopping)...\n")

    epochs_list = []
    losses_list = []
    imse_list = []

    # Evaluate at epoch 0 (random initialization)
    imse_0 = compute_imse(cate_model, X_eval, tau_true_eval, time_grid_jax, delta)
    epochs_list.append(0)
    losses_list.append(float("nan"))
    imse_list.append(imse_0)
    print(f"Epoch   0 | iMSE = {imse_0:.6f}")

    for epoch in range(1, CATE_EPOCHS + 1):
        perm = np.random.permutation(n)
        epoch_losses = []

        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]

            tv = _extract_vars(cate_model.trainable_variables)
            ntv = _extract_vars(cate_model.non_trainable_variables)
            ov = _extract_vars(optimizer.variables)

            loss, new_tv, new_ov = train_step(
                tv, ntv, ov,
                X_jax[idx], Y_tilde_jax[idx], S_marg_jax[idx],
                W_jax[idx], e_clipped_jax[idx],
            )

            _assign_vars(cate_model.trainable_variables, new_tv)
            _assign_vars(optimizer.variables, new_ov)
            epoch_losses.append(float(loss))

        mean_loss = np.mean(epoch_losses)
        imse = compute_imse(cate_model, X_eval, tau_true_eval, time_grid_jax, delta)

        epochs_list.append(epoch)
        losses_list.append(mean_loss)
        imse_list.append(imse)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | R-loss = {mean_loss:.6f} | iMSE = {imse:.6f}")

    # --- Plot ---
    print("\nGenerating learning curve plot...")
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_imse = "C0"
    color_loss = "C1"

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Integrated MSE against true CATE", fontsize=12, color=color_imse)
    ax1.plot(epochs_list, imse_list, color=color_imse, linewidth=2, label="iMSE")
    ax1.tick_params(axis="y", labelcolor=color_imse)
    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Training R-loss", fontsize=12, color=color_loss)
    ax2.plot(epochs_list[1:], losses_list[1:], color=color_loss, linewidth=1.5,
             alpha=0.7, label="R-loss")
    ax2.tick_params(axis="y", labelcolor=color_loss)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    fig.suptitle(
        "Learning Curve: iMSE and R-loss During Training\n"
        f"WeibullMixtureDGP, n={N_TRAIN}, K={K}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig11_learning_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
