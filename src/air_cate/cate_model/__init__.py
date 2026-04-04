"""CATE model: gamma survival, encoder, and mixture."""

__all__ = ["gamma_survival", "FeatureEncoder", "CATEMixture"]


def __getattr__(name):
    if name == "gamma_survival":
        from air_cate.cate_model.gamma_survival import gamma_survival
        return gamma_survival
    if name == "FeatureEncoder":
        from air_cate.cate_model.encoder import FeatureEncoder
        return FeatureEncoder
    if name == "CATEMixture":
        from air_cate.cate_model.mixture import CATEMixture
        return CATEMixture
    raise AttributeError(f"module 'air_cate.cate_model' has no attribute {name!r}")
