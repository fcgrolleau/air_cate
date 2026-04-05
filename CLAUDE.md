# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research paper and implementation: **"Learning Time-Varying Conditional Average Treatment Effects from Right-Censored Data"** by François Grolleau (Stanford) and Alex Fernandes (Université Paris Cité). The method is called **Surv-CATE** / **AIR** (Augmented and Integrated R-loss).

## Build & Test Commands

```bash
# Install all dependencies (including dev)
uv sync --all-extras

# Run all tests
uv run pytest tests/ -v

# Run only fast unit tests (skip integration)
uv run pytest tests/ -v -m "not slow"

# Run only slow integration tests
uv run pytest tests/ -v -m slow

# Run a single test file
uv run pytest tests/test_pseudo_outcome.py -v

# Run the quickstart example
uv run python examples/quickstart.py

# Lint
uv run ruff check src/ tests/
```

## Repository Structure

```
src/air_cate/                    # Python package (JAX + Keras 3)
├── __init__.py                  # Sets KERAS_BACKEND=jax, lazy-exports AIREstimator
├── _types.py                    # SurvivalData, NuisanceEstimates NamedTuples
├── utils.py                     # build_time_grid, clip_propensity, clip_survival
├── nuisance/
│   ├── propensity.py            # PropensityMLP (binary classifier)
│   ├── weibull_aft.py           # WeibullAFT (parametric survival, custom NLL)
│   └── cross_fit.py             # Q-fold cross-fitting orchestrator
├── pseudo_outcome.py            # AIPCW pseudo-outcome (pure JAX, O(n*M) prefix sums)
├── cate_model/
│   ├── gamma_survival.py        # G(t; alpha, lambda) via jax.scipy.special.gammaincc
│   ├── encoder.py               # FeatureEncoder MLP (swappable)
│   ├── mixture.py               # CATEMixture Keras model (MGD)
│   └── tensor_product.py        # CATETensorProduct (unconstrained baseline)
├── r_loss.py                    # Empirical R-loss function
├── baselines.py                 # SurvivalForestTLearner, SurvivalForestSLearner
├── nuisance/oracle.py           # Oracle nuisance builder for simulation studies
└── estimator.py                 # AIREstimator: top-level fit/predict API

tests/                           # pytest suite
├── conftest.py                  # Synthetic DGPs: WeibullDGP, WeibullMixtureDGP, NullDGP
├── test_estimator.py            # Integration tests (@pytest.mark.slow)
└── test_*.py                    # Unit tests for each module

latex/                           # Paper & simulation writeup
├── main.tex                     # Main paper
├── main.bib                     # Bibliography
├── related_work_frazier.tex     # Related work section
└── simulation_study.tex         # Simulation study (DGPs, design, results)

examples/
├── quickstart.py                # End-to-end demo on synthetic data
├── plot_simulation.py           # iMSE convergence (6 methods × 2 DGPs, --large/--plot-only)
└── plot_cate_curves.py          # Predicted tv-CATE curves for a single test observation
```

## Architecture: 3-Stage Pipeline

The AIREstimator follows Algorithm 1 from the paper:

1. **Nuisance estimation** (`nuisance/`): Cross-fit propensity (MLP) + 4 Weibull AFT models (event/censoring × arm 0/1). Each fold trains 5 models on the complement, predicts on the held-out fold.

2. **Pseudo-outcomes** (`pseudo_outcome.py`): Compute AIPCW pseudo-outcomes on a uniform time grid. Has 3 terms: IPCW, censored augmentation, and integral augmentation (via efficient prefix sums). Pure JAX, no Keras.

3. **CATE model** (`cate_model/`, `r_loss.py`): Minimize the R-loss over a CATEMixture model — a covariate-gated Mixture of Gamma Differences (MGD). Custom JAX training loop with `stateless_call` for JIT compatibility. An alternative `CATETensorProduct` model (unconstrained tensor-product basis) is available via `cate_model_class` parameter.

## Key Technical Details

- **Keras 3 + JAX backend**: Set via `os.environ["KERAS_BACKEND"] = "jax"` in `__init__.py`. Must be set before any Keras import.
- **WeibullAFT** uses a `_WeibullTrainModel` wrapper with `compute_loss(self, x, y, y_pred, sample_weight, training)` signature (Keras 3.3+ requirement).
- **CATEMixture training** uses `model.stateless_call(trainable_vars, non_trainable_vars, inputs)` inside a `jax.jit`-compiled training step. Keras variables must be extracted via `.value` before passing to JIT.
- **Positivity constraints**: Gamma shape/scale parameters enforced via `softplus(raw) + 1e-4`.
- **Structural constraints by construction**: `tau(0|x)=0`, `tau(t|x) in [-1,1]`, `tau -> 0` as `t -> inf`.

## LaTeX Conventions

- Bibliography style: `unsrtnat` (numbered, unsorted)
- Custom math operators: `\EX` (expectation), `\PR` (probability), `\IDF` (indicator), `\eqdef` (definition equality), `\indep` (independence)
- Custom commands: `\tend` (end time), `\ms` (mathscr), `\abs`, `\norm`, `\R`, `\Rp`, `\Rpstar`
- Theorem environments: theorem, proposition, lemma, definition, corollary, assumption, remark
