"""AIR estimator for time-varying CATE from right-censored survival data."""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")


def __getattr__(name):
    if name == "AIREstimator":
        from air_cate.estimator import AIREstimator

        return AIREstimator
    if name == "AIREstimatorCV":
        from air_cate.estimator_cv import AIREstimatorCV

        return AIREstimatorCV
    raise AttributeError(f"module 'air_cate' has no attribute {name!r}")


__all__ = ["AIREstimator", "AIREstimatorCV"]
