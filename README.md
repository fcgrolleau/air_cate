# AIR-CATE

**Augmented and Integrated R-loss (AIR)** estimator for learning time-varying Conditional Average Treatment Effects (CATE) from right-censored survival data.

This package implements the method described in:

> **"Learning Time-Varying Conditional Average Treatment Effects from Right-Censored Data"**
> François Grolleau (Stanford) and Alex Fernandes (Université Paris Cité)

## Installation

Requires Python >= 3.10. Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras
```

Or with pip:

```bash
pip install -e ".[dev,plots]"
```

Plotting lives in the optional `plots` extra (matplotlib). The estimator itself does not need it, and `air_cate.plotting` imports matplotlib lazily inside each function, so the package imports fine without it.

## Quickstart

```python
import numpy as np
from air_cate import AIREstimator

# X: covariates (n, p), W: binary treatment (n,)
# Z: observed times (n,), Delta: event indicators (n,)
est = AIREstimator(K=4, M=50, n_folds=3, rng_seed=0)
est.fit(X, W, Z, Delta, t_end=5.0)

tau_hat = est.predict_tau(X_test)      # CATE curves (n_test, M+1)
gamma_hat = est.predict_gamma(X_test)  # CRMST differences (n_test, M+1)

fig, ax = est.plot_convergence()       # training AIR CATE loss vs epoch
```

See [`examples/quickstart.py`](examples/quickstart.py) for a complete example on synthetic Weibull data.

### Choosing `K`

`K` — the number of clusters in the mixture of gamma differences — is a fixed constructor argument, so `AIREstimatorCV` screens a grid of candidates by cross-validating the AIR R-loss:

```python
from air_cate import AIREstimatorCV

cv = AIREstimatorCV(K_grid=(1, 2, 3, 4), n_cv_folds=3, n_folds=2, val_n_folds=2)
cv.fit(X, W, Z, Delta, t_end=5.0, verbose=1)

print(cv.summary())              # per-K loss, paired differences, mean best epoch
cv.plot_k_selection()            # CV loss vs K: absolute and paired panels
cv.plot_convergence_grid()       # columns = K, rows = CV fold; train vs held-out curves
```

It is **diagnostic only** — no `predict_*` methods and no final refit. Read `K` off the plots, then construct `AIREstimator(K=...)` and fit it on the full training set. Within each outer fold the nuisances are cross-fit independently in the training block and re-cross-fit in the held-out block, so no model fitted on training rows ever touches validation rows; the pseudo-outcomes are computed once per fold and reused across `K`, which makes the comparison paired. See [`examples/quickstart_cv.py`](examples/quickstart_cv.py).

Because the held-out nuisances are estimated from roughly `1/n_cv_folds` as many rows, the held-out curve is *expected* to sit above the training curve even for a well-specified model. Judge overfitting by whether it turns upward, never by the gap.

### Reproducibility

Four independent random streams have to be pinned to make a fit reproducible:

```python
import keras
keras.utils.set_random_seed(0)                    # weight init (also Python + NumPy globals)
est = AIREstimator(K=4, rng_seed=0)               # CATE minibatch order
SurvivalForestTLearner(random_state=0)            # baselines, if used
WeibullDGP.generate(n=2000, rng_seed=0)           # the data itself
```

`set_random_seed` does not cover the minibatch order — `train_cate` draws from its own generator — so `rng_seed` must be passed explicitly. Nuisance cross-fitting splits are already deterministic.

## Method

The estimator follows a three-stage pipeline:

1. **Nuisance estimation** -- Cross-fit propensity scores (MLP) and survival functions (Weibull AFT) for event and censoring processes under each treatment arm.

2. **Pseudo-outcomes** -- Compute doubly-robust AIPCW pseudo-outcomes on a uniform time grid using efficient O(n*M) prefix sums.

3. **CATE model** -- Minimize the empirical R-loss over a `CATEMixture` model: a covariate-gated Mixture of Gamma Differences (MGD) with structural guarantees (tau(0|x) = 0, tau in [-1, 1], tau -> 0 as t -> infinity). An alternative unconstrained `CATETensorProduct` model is also available.

## Examples

| Script | Description |
|---|---|
| [`quickstart.py`](examples/quickstart.py) | End-to-end demo on synthetic data, plus the CATE training curve |
| [`quickstart_cv.py`](examples/quickstart_cv.py) | Pick `K` by cross-validation on a DGP whose true `K` is 3 |
| [`plot_k_selection.py`](examples/plot_k_selection.py) | Larger `K` screen: CV curve + convergence grid (`--large`, `--plot-only`) |
| [`plot_simulation.py`](examples/plot_simulation.py) | iMSE convergence across 6 methods, 2 DGPs (`--large`, `--plot-only`) |
| [`plot_convergence.py`](examples/plot_convergence.py) | AIR-only iMSE vs sample size — the narrower precursor to `plot_simulation.py` |
| [`plot_cate_curves.py`](examples/plot_cate_curves.py) | Predicted tv-CATE curves per test subject (`--seed`, `--obs`, `--preset`) |

Run any example with:

```bash
uv run python examples/<script>.py

# Scripts that save figures need the plots extra:
uv run --extra plots python examples/<script>.py
```

Figures are saved to `examples/figures/`. The longer scripts cache their fitted results (`.json` or `.npz`) and take `--plot-only` to restyle without refitting; `plot_cate_curves.py` stores a fingerprint of the configuration that produced its cache and refuses to plot a cache written under a different one.

## API

### `AIREstimator`

| Parameter | Default | Description |
|---|---|---|
| `K` | 4 | Number of mixture components |
| `M` | 100 | Number of time grid intervals (M+1 points) |
| `n_folds` | 5 | Cross-fitting folds |
| `eps` | 0.01 | Clipping bound for nuisance estimates |
| `encoder_hidden_sizes` | (64, 32) | Feature encoder MLP architecture |
| `cate_epochs` | 200 | Max training epochs |
| `cate_lr` | 1e-3 | Learning rate (AdamW) |
| `cate_batch_size` | 256 | Batch size |
| `weight_decay` | 1e-4 | AdamW weight decay |
| `patience` | 10 | Epochs without training-loss improvement before stopping (`None` = full budget) |
| `rng_seed` | `None` | Seed for minibatch shuffling |
| `cate_model_class` | `None` | Alternative CATE model, e.g. `CATETensorProduct` |
| `cate_model_kwargs` | `None` | Extra kwargs for `cate_model_class` |

**Methods:**
- `fit(X, W, Z, Delta, t_end, nuisances=None, verbose=0)` -- Fit the full pipeline. Pass `nuisances` to skip cross-fitting (used by the oracle simulations).
- `predict_tau(X)` -- Predict CATE curves tau(t|x) on the fitted time grid.
- `predict_gamma(X)` -- Predict cumulative RMST differences Gamma(t|x).
- `plot_convergence(**kwargs)` -- Training AIR CATE loss vs epoch.

**Fitted attributes:** `cate_model_`, `time_grid_`, `delta_`, `history_` (per-epoch loss trajectory).

### `AIREstimatorCV`

Mirrors the `AIREstimator` arguments above so the screen matches the final fit, plus:

| Parameter | Default | Description |
|---|---|---|
| `K_grid` | (1, 2, 3, 4, 6, 8) | Candidate cluster counts to screen |
| `n_cv_folds` | 5 | Outer folds (model selection over `K`) |
| `n_folds` | 5 | Inner nuisance folds for the training block |
| `val_n_folds` | `None` | Inner nuisance folds for the held-out block (`None` = `n_folds`) |
| `cv_rng_seed` | 42 | Seed for the outer split, nuisance splits, and weight init |
| `cv_patience` | `None` | Early stopping on held-out loss; `None` gives complete, equal-length curves |
| `k_param_name` | `"K"` | Constructor argument that sets `K`, so custom models can be screened too |

**Methods:** `fit`, `summary`, `to_dict` / `from_dict` (JSON-safe, for `--plot-only`), `plot_cv_curve`, `plot_cv_paired`, `plot_k_selection`, `plot_convergence_grid`.

**Fitted attributes:** `cv_results_` (`K_grid`, `fold_val_loss`, `mean_val_loss`, `se_val_loss`, `fold_best_epoch`), `histories_` keyed by `(K, fold)`, `best_K_`, `time_grid_`, `delta_`.

Cost: `n_cv_folds * (n_folds + val_n_folds) * 5` nuisance models plus `n_cv_folds * len(K_grid)` CATE fits — 250 nuisance models at the defaults. Use `n_folds=2, val_n_folds=2` for a first screen. Each held-out nuisance model sees about `n / n_cv_folds * (1 - 1/val_n_folds)` rows before splitting per arm; keep half of that comfortably above 20.

### `air_cate.plotting`

Standalone functions behind the estimator methods, plus the shared style. All return `(fig, ax)` and accept an existing `ax`.

| Function | Figure |
|---|---|
| `plot_convergence` | Training (and optional held-out) AIR loss vs epoch, best epoch marked |
| `plot_cv_curve` | CV loss vs `K` on the absolute scale, with per-fold lines |
| `plot_cv_paired` | CV loss differenced within folds against a reference `K` — the panel to decide from |
| `plot_k_selection` | The two panels above, side by side |
| `plot_convergence_grid` | Columns = `K`, rows = CV fold; training vs held-out curves |
| `apply_style(width_scale=1.0)` | Shared rcParams; `width_scale` thickens spines for presentation-size figures |
| `PALETTE` | The JAMA palette dict, so scripts don't redeclare hex codes |

## Project Structure

```
src/air_cate/
├── __init__.py              # Sets KERAS_BACKEND=jax; lazy-exports AIREstimator(CV)
├── _types.py                # SurvivalData, NuisanceEstimates
├── utils.py                 # Time grid, clipping utilities
├── nuisance/
│   ├── propensity.py        # PropensityMLP
│   ├── weibull_aft.py       # WeibullAFT (parametric survival)
│   ├── cross_fit.py         # Q-fold cross-fitting
│   └── oracle.py            # Oracle nuisances for simulation studies
├── pseudo_outcome.py        # AIPCW pseudo-outcomes (pure JAX)
├── cate_model/
│   ├── gamma_survival.py    # Gamma survival function
│   ├── encoder.py           # Feature encoder MLP
│   ├── mixture.py           # CATEMixture Keras model (MGD)
│   ├── tensor_product.py    # CATETensorProduct (unconstrained baseline)
│   └── train.py             # train_cate: shared JIT training loop + loss history
├── r_loss.py                # R-loss function
├── baselines.py             # Survival Forest T-learner / S-learner
├── plotting.py              # Convergence, CV, and K-selection figures
├── estimator.py             # AIREstimator (top-level API)
└── estimator_cv.py          # AIREstimatorCV (cross-validated K screen)

tests/                       # pytest suite
├── conftest.py              # Synthetic DGPs: WeibullDGP, WeibullMixtureDGP, NullDGP
└── test_*.py                # Unit + integration tests

examples/                    # Demo scripts + figures
latex/                       # Paper and simulation study writeup
```

## Testing

```bash
# All tests
uv run python -m pytest tests/ -v

# Fast unit tests only
uv run python -m pytest tests/ -v -m "not slow"

# Slow integration tests only
uv run python -m pytest tests/ -v -m slow

# Lint
uv run ruff check src/ tests/
```

Invoke pytest as `python -m pytest`, not as bare `uv run pytest`. The module form puts the working directory on `sys.path`; the console script does not, so collection dies on the first local import with a misleading `ModuleNotFoundError` (`air_cate`, or `jax`, depending on which test is collected first).

## Tech Stack

- **JAX** -- Numerical computation, JIT-compiled training loop
- **Keras 3** (JAX backend) -- Neural network layers and model abstraction
- **scikit-survival** -- Random Survival Forest baselines
- **NumPy / scikit-learn** -- Data handling and utilities
- **matplotlib** (optional `plots` extra) -- Figures
