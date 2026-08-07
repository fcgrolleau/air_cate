"""Tests for AIREstimatorCV: cross-validated screening of K."""

import json

import numpy as np
import pytest

from conftest import WeibullMixtureDGP


def _tiny_data(n=300, p=5, rng_seed=0):
    """Small synthetic survival dataset for fast shape/plumbing tests."""
    rng = np.random.default_rng(rng_seed)
    X = rng.uniform(-1, 1, (n, p)).astype(np.float32)
    W = rng.binomial(1, 0.5, n).astype(np.float32)
    Z = rng.exponential(2.0, n).astype(np.float32) + 0.01
    Delta = rng.binomial(1, 0.7, n).astype(np.float32)
    return X, W, Z, Delta, 5.0


TINY_CONFIG = dict(
    K_grid=(1, 2),
    n_cv_folds=2,
    n_folds=2,
    val_n_folds=2,
    M=10,
    encoder_hidden_sizes=(16,),
    nuisance_training_config={"epochs": 3, "verbose": 0},
    cate_epochs=5,
)


def test_cv_shapes_and_histories():
    """Smoke test: result arrays, history lengths, and best_K_ are well formed."""
    from air_cate.estimator_cv import AIREstimatorCV

    X, W, Z, Delta, t_end = _tiny_data()

    cv = AIREstimatorCV(**TINY_CONFIG).fit(X, W, Z, Delta, t_end, verbose=0)

    n_K, n_folds = 2, 2
    res = cv.cv_results_
    assert res["mean_val_loss"].shape == (n_K,)
    assert res["se_val_loss"].shape == (n_K,)
    assert res["fold_val_loss"].shape == (n_K, n_folds)
    assert res["fold_best_epoch"].shape == (n_K, n_folds)

    assert np.all(np.isfinite(res["fold_val_loss"])), "non-finite CV loss"
    assert np.all(res["fold_val_loss"] >= 0), "R-loss must be non-negative"

    # One history per (K, fold)
    assert len(cv.histories_) == n_K * n_folds
    for K in cv.K_grid:
        for j in range(n_folds):
            h = cv.histories_[(K, j)]
            assert len(h["train_loss"]) == TINY_CONFIG["cate_epochs"]
            assert len(h["val_loss"]) == TINY_CONFIG["cate_epochs"]
            assert 0 <= h["best_epoch"] < TINY_CONFIG["cate_epochs"]
            assert h["monitor"] == "val"
            # best_val_loss really is the minimum of the recorded curve
            assert h["best_val_loss"] == pytest.approx(min(h["val_loss"]))

    assert cv.best_K_ in cv.K_grid
    assert isinstance(cv.summary(), str)


def test_cv_no_early_stopping_by_default():
    """cv_patience=None trains the full cate_epochs, giving equal-length curves."""
    from air_cate.estimator_cv import AIREstimatorCV

    X, W, Z, Delta, t_end = _tiny_data()
    cv = AIREstimatorCV(**TINY_CONFIG).fit(X, W, Z, Delta, t_end, verbose=0)

    lengths = {len(h["train_loss"]) for h in cv.histories_.values()}
    assert lengths == {TINY_CONFIG["cate_epochs"]}, (
        "with cv_patience=None every run should train the full cate_epochs"
    )


def test_cv_determinism():
    """The same cv_rng_seed reproduces the same fold-level scores.

    Two fits run back-to-back in one process, so this also guards against RNG
    state leaking in from whatever ran before: the nuisance models take their
    weight init from Keras's global RNG, which AIREstimatorCV must seed per
    block. Without that seeding only the *later* folds come out reproducible,
    because the K loop's set_random_seed happens to pin them.
    """
    from air_cate.estimator_cv import AIREstimatorCV

    X, W, Z, Delta, t_end = _tiny_data()

    a = AIREstimatorCV(cv_rng_seed=7, **TINY_CONFIG).fit(X, W, Z, Delta, t_end)
    b = AIREstimatorCV(cv_rng_seed=7, **TINY_CONFIG).fit(X, W, Z, Delta, t_end)

    np.testing.assert_allclose(
        a.cv_results_["fold_val_loss"],
        b.cv_results_["fold_val_loss"],
        rtol=1e-5,
        err_msg="same seed should give identical CV scores",
    )

    # Compare the whole trajectories too: at tiny n the R-loss is dominated by a
    # model-independent term, so the final scores alone are a coarse probe.
    for key, h_a in a.histories_.items():
        for curve in ("train_loss", "val_loss"):
            np.testing.assert_allclose(
                h_a[curve],
                b.histories_[key][curve],
                rtol=1e-5,
                err_msg=f"{curve} trajectory for (K, fold)={key} is not reproducible",
            )


def test_cv_json_round_trip():
    """to_dict/from_dict survives a JSON round trip (used by --plot-only)."""
    from air_cate.estimator_cv import AIREstimatorCV

    X, W, Z, Delta, t_end = _tiny_data()
    cv = AIREstimatorCV(**TINY_CONFIG).fit(X, W, Z, Delta, t_end)

    restored = AIREstimatorCV.from_dict(json.loads(json.dumps(cv.to_dict())))

    assert restored.best_K_ == cv.best_K_
    assert restored.histories_.keys() == cv.histories_.keys()
    for key in ("fold_val_loss", "mean_val_loss", "se_val_loss", "fold_best_epoch"):
        np.testing.assert_allclose(
            restored.cv_results_[key], cv.cv_results_[key], rtol=1e-6
        )
    for key, h in cv.histories_.items():
        np.testing.assert_allclose(
            restored.histories_[key]["val_loss"], h["val_loss"], rtol=1e-6
        )


def test_cv_blocks_are_independent(monkeypatch):
    """No training row ever informs a held-out row's nuisances.

    This is the property the whole design turns on: within each outer fold the two
    cross_fit_nuisances calls must see disjoint row blocks that together partition
    the data.
    """
    import air_cate.estimator_cv as cv_module
    from air_cate.estimator_cv import AIREstimatorCV

    X, W, Z, Delta, t_end = _tiny_data(n=300)

    seen_blocks = []
    real_cross_fit = cv_module.cross_fit_nuisances

    def recording_cross_fit(data, *args, **kwargs):
        # Identify the block by its Z values, which are distinct here.
        seen_blocks.append(np.asarray(data.Z).copy())
        return real_cross_fit(data, *args, **kwargs)

    monkeypatch.setattr(cv_module, "cross_fit_nuisances", recording_cross_fit)

    n_cv_folds = 2
    config = dict(TINY_CONFIG)
    config["n_cv_folds"] = n_cv_folds
    AIREstimatorCV(**config).fit(X, W, Z, Delta, t_end)

    # Two calls per outer fold: training block then held-out block.
    assert len(seen_blocks) == 2 * n_cv_folds

    all_Z = np.asarray(Z, dtype=np.float32)
    for j in range(n_cv_folds):
        block_A, block_B = seen_blocks[2 * j], seen_blocks[2 * j + 1]

        set_A, set_B = set(block_A.tolist()), set(block_B.tolist())
        assert not (set_A & set_B), f"fold {j}: blocks overlap"
        assert set_A | set_B == set(all_Z.tolist()), f"fold {j}: blocks miss rows"
        assert len(block_A) + len(block_B) == len(all_Z)


def test_cv_rejects_undersized_val_block():
    """A held-out block too small to cross-fit raises a clear error."""
    from air_cate.estimator_cv import AIREstimatorCV

    X, W, Z, Delta, t_end = _tiny_data(n=12)

    cv = AIREstimatorCV(
        K_grid=(1,), n_cv_folds=6, n_folds=2, val_n_folds=5,
        M=10, encoder_hidden_sizes=(8,),
        nuisance_training_config={"epochs": 2, "verbose": 0},
        cate_epochs=2,
    )
    with pytest.raises(ValueError, match="cannot be cross-fit"):
        cv.fit(X, W, Z, Delta, t_end)


def test_cv_rejects_single_fold():
    """n_cv_folds < 2 is meaningless for cross-validation."""
    from air_cate.estimator_cv import AIREstimatorCV

    X, W, Z, Delta, t_end = _tiny_data(n=100)
    with pytest.raises(ValueError, match="n_cv_folds must be >= 2"):
        AIREstimatorCV(K_grid=(1,), n_cv_folds=1).fit(X, W, Z, Delta, t_end)


@pytest.mark.slow
def test_cv_prefers_heterogeneity_on_mixture_dgp():
    """On a K=3 DGP the screen should prefer K>1 over the no-heterogeneity model.

    With K=1 the gate is a softmax over one logit, so rho == 1 and tau(t|x) has no
    covariate dependence at all. A working screen should beat that.
    """
    from air_cate.estimator_cv import AIREstimatorCV

    X, W, Z, Delta, t_end = WeibullMixtureDGP.generate(n=1500, rng_seed=0)

    cv = AIREstimatorCV(
        K_grid=(1, 2, 3, 5),
        n_cv_folds=3,
        n_folds=2,
        val_n_folds=2,
        M=50,
        encoder_hidden_sizes=(32, 16),
        nuisance_training_config={"epochs": 50, "verbose": 0},
        cate_epochs=120,
    ).fit(X, W, Z, Delta, t_end, verbose=0)

    print("\n" + cv.summary())
    print(f"Selected K = {cv.best_K_} (true K = 3)")

    assert cv.best_K_ >= 2, (
        f"screen chose the no-heterogeneity model K={cv.best_K_} on a K=3 DGP"
    )
