"""A from-scratch K-nearest neighbors classifier using Euclidean distance."""

from __future__ import annotations

from collections import Counter

import numpy as np


def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Calculate Euclidean distance between two numeric vectors."""
    return float(np.sqrt(np.sum((x1 - x2) ** 2)))


class KNNClassifier:
    """Simple KNN classifier implemented from scratch.

    The model stores the training set during `fit` and performs all distance
    calculations at prediction time.
    """

    def __init__(self, k: int = 5):
        if k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k
        self.x_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """Store training features and labels."""
        if len(x) != len(y):
            raise ValueError("x and y must contain the same number of rows")
        self.x_train = np.asarray(x, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def predict(self, x_new: np.ndarray) -> np.ndarray:
        """Predict labels for one or more observations."""
        self._check_is_fit()
        x_new = np.asarray(x_new, dtype=float)
        if x_new.ndim == 1:
            x_new = x_new.reshape(1, -1)
        return np.array([self._predict_one(row) for row in x_new])

    def _predict_one(self, row: np.ndarray) -> str:
        distances = np.array([
            euclidean_distance(train_row, row) for train_row in self.x_train
        ])
        neighbor_count = min(self.k, len(self.x_train))
        nearest_indices = np.argsort(distances)[:neighbor_count]
        nearest_labels = self.y_train[nearest_indices]
        return Counter(nearest_labels).most_common(1)[0][0]

    def _check_is_fit(self) -> None:
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Call fit before predict")
