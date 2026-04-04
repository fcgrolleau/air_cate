"""Nuisance function estimation: propensity, survival, and cross-fitting."""

__all__ = ["PropensityMLP", "WeibullAFT", "cross_fit_nuisances"]


def __getattr__(name):
    if name == "PropensityMLP":
        from air_cate.nuisance.propensity import PropensityMLP
        return PropensityMLP
    if name == "WeibullAFT":
        from air_cate.nuisance.weibull_aft import WeibullAFT
        return WeibullAFT
    if name == "cross_fit_nuisances":
        from air_cate.nuisance.cross_fit import cross_fit_nuisances
        return cross_fit_nuisances
    raise AttributeError(f"module 'air_cate.nuisance' has no attribute {name!r}")
