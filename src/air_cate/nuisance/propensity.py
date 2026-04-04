"""Propensity score estimation via MLP binary classifier."""

import keras
import numpy as np


class PropensityMLP:
    """MLP-based propensity score model P(W=1|X)."""

    def __init__(self, hidden_sizes: list[int] = (64, 32), dropout: float = 0.0):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.model_ = None

    def _build_model(self, input_dim: int) -> keras.Model:
        inputs = keras.Input(shape=(input_dim,))
        x = inputs
        for units in self.hidden_sizes:
            x = keras.layers.Dense(units, activation="relu")(x)
            if self.dropout > 0:
                x = keras.layers.Dropout(self.dropout)(x)
        outputs = keras.layers.Dense(1, activation="sigmoid")(x)
        return keras.Model(inputs, outputs)

    def fit(
        self,
        X: np.ndarray,
        W: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        verbose: int = 0,
    ) -> "PropensityMLP":
        self.model_ = self._build_model(X.shape[1])
        self.model_.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="binary_crossentropy",
        )
        self.model_.fit(
            X,
            W,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            validation_split=0.1,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return P(W=1|X) as a 1-d array."""
        return self.model_.predict(X, verbose=0).ravel()
