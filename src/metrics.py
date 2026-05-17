"""Evaluation helpers for KNN classification experiments."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the share of predictions that exactly match true labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(y_true == y_pred))


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, positive_label: str = "+") -> dict[str, float]:
    """Calculate binary classification metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    tp = int(np.sum((y_true == positive_label) & (y_pred == positive_label)))
    tn = int(np.sum((y_true != positive_label) & (y_pred != positive_label)))
    fp = int(np.sum((y_true != positive_label) & (y_pred == positive_label)))
    fn = int(np.sum((y_true == positive_label) & (y_pred != positive_label)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Iterable[str]) -> np.ndarray:
    """Create a confusion matrix using the supplied label order."""
    labels = list(labels)
    label_index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for true_label, predicted_label in zip(y_true, y_pred):
        matrix[label_index[true_label], label_index[predicted_label]] += 1
    return matrix
