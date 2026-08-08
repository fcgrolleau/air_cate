"""Column mapping and feature encoding for IST-3 / TVAE data."""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Canonical feature order (after encoding) ────────────────────────────
# Binary features (6): sex_male, rvisinf, ratrial, rdef1, rdef2, rdef3
# Continuous features (4): age, rsbp, rdelay, gcs_score
# Total: 10 features

BINARY_FEATURES = [
    "sex_male", "rvisinf", "ratrial",
    "rdef1", "rdef2", "rdef3",
]
CONTINUOUS_FEATURES = ["age", "rsbp", "rdelay", "gcs_score"]

# ── IST-3 column mappings ───────────────────────────────────────────────


def _encode_binary_12(series):
    """Recode 1=yes→1, 2=no→0."""
    return (series == 1).astype(np.float32)


def _encode_rdef_12(series):
    """Recode 1=yes→1, 2=no→0, other (3,4)→0.5."""
    out = np.full(len(series), 0.5, dtype=np.float32)
    out[series.values == 1] = 1.0
    out[series.values == 2] = 0.0
    return out


def encode_ist3(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode ist3_imp.csv into (X, W, T, Delta).

    Returns:
        X: (n, p) float32 feature matrix (unstandardized continuous features)
        W: (n,) float32 treatment indicator
        T: (n,) float32 observed time (surv6)
        Delta: (n,) float32 event indicator (1=death, 0=censored)
    """
    n = len(df)
    features = {}

    # Sex: 1=male→1, 2=female→0
    features["sex_male"] = (df["gender"].values == 1).astype(np.float32)

    # RVISINF: R_isch_change (1=yes→1, 2=no→0, 3→0.5)
    ric = df["R_isch_change"].values
    feat = np.full(n, 0.5, dtype=np.float32)
    feat[ric == 1] = 1.0
    feat[ric == 2] = 0.0
    features["rvisinf"] = feat

    # Atrial fibrillation
    features["ratrial"] = _encode_binary_12(df["atrialfib_rand"])

    # Motor deficits (RDEF 1-3)
    for ist3_col, canon in [("weakface_rand", "rdef1"),
                            ("weakarm_rand", "rdef2"),
                            ("weakleg_rand", "rdef3")]:
        features[canon] = _encode_rdef_12(df[ist3_col])

    # Continuous
    features["age"] = df["ageimp"].values.astype(np.float32)
    features["rsbp"] = df["sbprand"].values.astype(np.float32)
    features["rdelay"] = df["treatdelay"].values.astype(np.float32)
    features["gcs_score"] = df["gcs_score_rand"].values.astype(np.float32)

    # Assemble feature matrix in canonical order
    X = np.column_stack([features[c] for c in BINARY_FEATURES + CONTINUOUS_FEATURES])

    W = df["tmt"].values.astype(np.float32)
    T = df["surv6"].values.astype(np.float32)
    Delta = (1 - df["censor6"].values).astype(np.float32)  # censor6=0 → event

    return X, W, T, Delta


# ── TVAE encoding ───────────────────────────────────────────────────────
_TVAE_YN = {"Y": 1.0, "N": 0.0}
_TVAE_YNC = {"Y": 1.0, "N": 0.0, "C": 0.5}
_TVAE_RCONSC = {"F": 15.0, "D": 11.0, "U": 5.0}  # approximate GCS mapping


def encode_tvae(df: pd.DataFrame) -> np.ndarray:
    """Encode TVAE sampled_data.csv into (n, p) feature matrix.

    Same feature space as encode_ist3, but from TVAE column names/codings.
    Continuous features are NOT standardized (do that separately).
    """
    features = {}

    # Binary
    features["sex_male"] = (df["SEX"] == "M").astype(np.float32).values
    features["rvisinf"] = df["RVISINF"].map(_TVAE_YN).astype(np.float32).values
    features["ratrial"] = df["RATRIAL"].map(_TVAE_YN).astype(np.float32).values

    # RDEF 1-3
    for i in range(1, 4):
        features[f"rdef{i}"] = df[f"RDEF{i}"].map(_TVAE_YNC).astype(np.float32).values

    # Continuous
    features["age"] = df["AGE"].values.astype(np.float32)
    features["rsbp"] = df["RSBP"].values.astype(np.float32)
    features["rdelay"] = df["RDELAY"].values.astype(np.float32)
    features["gcs_score"] = df["RCONSC"].map(_TVAE_RCONSC).astype(np.float32).values

    return np.column_stack([features[c] for c in BINARY_FEATURES + CONTINUOUS_FEATURES])


def feature_names():
    """Return canonical feature names in order."""
    return BINARY_FEATURES + CONTINUOUS_FEATURES


class FeatureScaler:
    """Z-score standardization for continuous features only."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self._cont_idx = None

    def fit(self, X: np.ndarray):
        n_feat = len(BINARY_FEATURES)
        n_cont = len(CONTINUOUS_FEATURES)
        self._cont_idx = list(range(n_feat, n_feat + n_cont))
        self.mean_ = X[:, self._cont_idx].mean(axis=0)
        self.std_ = X[:, self._cont_idx].std(axis=0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        X[:, self._cont_idx] = (X[:, self._cont_idx] - self.mean_) / self.std_
        return X.astype(np.float32)

    def save(self, path: str | Path):
        d = {"mean": self.mean_.tolist(), "std": self.std_.tolist(),
             "cont_idx": self._cont_idx}
        Path(path).write_text(json.dumps(d))

    @classmethod
    def load(cls, path: str | Path):
        d = json.loads(Path(path).read_text())
        s = cls()
        s.mean_ = np.array(d["mean"], dtype=np.float32)
        s.std_ = np.array(d["std"], dtype=np.float32)
        s._cont_idx = d["cont_idx"]
        return s
