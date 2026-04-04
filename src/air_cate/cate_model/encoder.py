"""Feature encoder phi(x) for the CATE mixture model."""

import keras


class FeatureEncoder(keras.Layer):
    """MLP feature encoder: X (n, p) -> phi(X) (n, d).

    Swappable module — replace with CNN, Transformer, etc. for non-tabular data.
    """

    def __init__(self, hidden_sizes: tuple[int, ...] = (64, 32), **kwargs):
        super().__init__(**kwargs)
        self.hidden_layers = [
            keras.layers.Dense(units, activation="relu") for units in hidden_sizes
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return x
