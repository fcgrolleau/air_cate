"""CATE model: gamma survival, encoder, mixture, and tensor-product."""

__all__ = ["gamma_survival", "FeatureEncoder", "CATEMixture", "CATETensorProduct"]


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
    if name == "CATETensorProduct":
        from air_cate.cate_model.tensor_product import CATETensorProduct
        return CATETensorProduct
    raise AttributeError(f"module 'air_cate.cate_model' has no attribute {name!r}")
