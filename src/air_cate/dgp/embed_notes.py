"""Embed clinical notes with Bio_ClinicalBERT and reduce via PCA.

Usage:
    uv run python -m air_cate.dgp.embed_notes

Requires: pip install transformers torch (or uv sync --extra text)

Reads:  artifacts_text/clinical_notes.json.gz
Writes: artifacts_text/embeddings.npz
"""

import gzip
import json
import time
from pathlib import Path

import numpy as np

ARTIFACT_DIR = Path(__file__).parent / "artifacts_text"
NOTES_PATH = ARTIFACT_DIR / "clinical_notes.json.gz"
OUTPUT_PATH = ARTIFACT_DIR / "embeddings.npz"

# Use local copy if available (avoids SSL issues behind proxies),
# otherwise fall back to HuggingFace Hub.
_LOCAL_BERT = ARTIFACT_DIR / "Bio_ClinicalBERT"
BERT_MODEL = str(_LOCAL_BERT) if _LOCAL_BERT.exists() else "emilyalsentzer/Bio_ClinicalBERT"
BATCH_SIZE = 64
PCA_VARIANCE = 0.95  # retain 95% of variance


def load_notes() -> list[str]:
    """Load clinical notes from compressed JSON."""
    with gzip.open(NOTES_PATH, "rt") as f:
        notes = json.load(f)
    print(f"Loaded {len(notes)} notes")
    return notes


def embed_notes(notes: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Embed notes with Bio_ClinicalBERT (mean-pool + L2-normalise).

    Returns (n, 768) float32 array.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    print(f"Loading {BERT_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModel.from_pretrained(BERT_MODEL)
    model.eval()

    embeddings = []
    n = len(notes)
    t0 = time.time()

    for start in range(0, n, batch_size):
        batch = notes[start : start + batch_size]
        tokens = tokenizer(
            batch, padding=True, truncation=True, max_length=128,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**tokens)

        # Mean-pool over non-padding tokens
        mask = tokens["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
        hidden = outputs.last_hidden_state * mask  # (B, L, 768)
        pooled = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, 768)

        # L2-normalise
        pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-8)
        embeddings.append(pooled.numpy())

        if (start // batch_size) % 50 == 0:
            elapsed = time.time() - t0
            done = min(start + batch_size, n)
            print(f"  [{done}/{n}] {elapsed:.1f}s")

    result = np.concatenate(embeddings, axis=0).astype(np.float32)
    print(f"Raw embeddings shape: {result.shape}")
    return result


def pca_reduce(E: np.ndarray, variance: float = PCA_VARIANCE):
    """PCA-reduce embeddings retaining `variance` fraction of total variance.

    Returns (E_pca, pca_mean, pca_components).
    """
    from sklearn.decomposition import PCA

    print(f"Fitting PCA (retaining {variance*100:.0f}% variance)...")
    pca = PCA(n_components=variance, svd_solver="full")
    E_pca = pca.fit_transform(E).astype(np.float32)
    print(f"  Reduced: {E.shape[1]} → {E_pca.shape[1]} components "
          f"({pca.explained_variance_ratio_.sum()*100:.1f}% variance)")

    return E_pca, pca.mean_.astype(np.float32), pca.components_.astype(np.float32)


def validate_signal(E_pca: np.ndarray):
    """Quick logistic regression probe: E → deathdep."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    tvae_csv = Path(__file__).parents[3] / "IST_VAE" / "sampled_data.csv"
    df = pd.read_csv(tvae_csv)
    y = df["deathdep"].values[:len(E_pca)]

    lr = LogisticRegression(max_iter=200, C=1.0)
    scores = cross_val_score(lr, E_pca, y, cv=5, scoring="roc_auc")
    print(f"Validation: E → deathdep AUC = {scores.mean():.3f} ± {scores.std():.3f}")
    return scores.mean()


def main():
    notes = load_notes()

    # Embed
    E_raw = embed_notes(notes)

    # PCA
    E_pca, pca_mean, pca_components = pca_reduce(E_raw)

    # Save
    np.savez_compressed(
        str(OUTPUT_PATH),
        E_pca=E_pca,
        pca_mean=pca_mean,
        pca_components=pca_components,
    )
    print(f"Saved embeddings to {OUTPUT_PATH} "
          f"({OUTPUT_PATH.stat().st_size / 1e6:.1f} MB)")

    # Validate
    auc = validate_signal(E_pca)
    if auc < 0.55:
        print("WARNING: Low AUC — embeddings may not carry enough signal.")


if __name__ == "__main__":
    main()
