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

## Quickstart

```python
import numpy as np
from air_cate import AIREstimator

# X: covariates (n, p), W: binary treatment (n,)
# Z: observed times (n,), Delta: event indicators (n,)
est = AIREstimator(K=4, M=50, n_folds=3)
est.fit(X, W, Z, Delta, t_end=5.0)

tau_hat = est.predict_tau(X_test)      # CATE curves (n_test, M+1)
gamma_hat = est.predict_gamma(X_test)  # CRMST differences (n_test, M+1)
```

See [`examples/quickstart.py`](examples/quickstart.py) for a complete example on synthetic Weibull data.

## Method

The estimator follows a three-stage pipeline:

1. **Nuisance estimation** -- Cross-fit propensity scores (MLP) and survival functions (Weibull AFT) for event and censoring processes under each treatment arm.

2. **Pseudo-outcomes** -- Compute doubly-robust AIPCW pseudo-outcomes on a uniform time grid using efficient O(n*M) prefix sums.

3. **CATE model** -- Minimize the empirical R-loss over a `CATEMixture` model: a covariate-gated mixture of K Gamma survival differences with structural guarantees (tau(0|x) = 0, tau in [-1, 1], tau -> 0 as t -> infinity).

## Examples

| Script | Description |
|---|---|
| [`quickstart.py`](examples/quickstart.py) | End-to-end demo on synthetic data |
| [`plot_convergence.py`](examples/plot_convergence.py) | iMSE vs sample size convergence curves with error bars |

Run any example with:

```bash
uv run python examples/<script>.py
```

Figures are saved to `examples/figures/`.

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

**Methods:**
- `fit(X, W, Z, Delta, t_end, verbose=0)` -- Fit the full pipeline.
- `predict_tau(X)` -- Predict CATE curves tau(t|x) on the fitted time grid.
- `predict_gamma(X)` -- Predict cumulative RMST differences Gamma(t|x).

## Project Structure

```
src/air_cate/
├── __init__.py              # Lazy-exports AIREstimator
├── _types.py                # SurvivalData, NuisanceEstimates
├── utils.py                 # Time grid, clipping utilities
├── nuisance/
│   ├── propensity.py        # PropensityMLP
│   ├── weibull_aft.py       # WeibullAFT (parametric survival)
│   └── cross_fit.py         # Q-fold cross-fitting
├── pseudo_outcome.py        # AIPCW pseudo-outcomes (pure JAX)
├── cate_model/
│   ├── gamma_survival.py    # Gamma survival function
│   ├── encoder.py           # Feature encoder MLP
│   └── mixture.py           # CATEMixture Keras model
├── r_loss.py                # R-loss function
└── estimator.py             # AIREstimator (top-level API)

tests/                       # pytest suite (29 tests)
├── conftest.py              # Synthetic DGPs: WeibullDGP, WeibullMixtureDGP, NullDGP
└── test_*.py                # Unit + integration tests

examples/                    # Demo scripts + figures
latex/                       # Paper source
```

## Testing

```bash
# All tests
uv run pytest tests/ -v

# Fast unit tests only
uv run pytest tests/ -v -m "not slow"

# Slow integration tests only
uv run pytest tests/ -v -m slow
```

## Tech Stack

- **JAX** -- Numerical computation, JIT-compiled training loop
- **Keras 3** (JAX backend) -- Neural network layers and model abstraction
- **NumPy / scikit-learn** -- Data handling and utilities
