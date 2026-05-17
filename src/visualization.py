"""Plotting helpers for classification results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(matrix: np.ndarray, labels: list[str], title: str = "Confusion Matrix"):
    """Plot a confusion matrix with matplotlib."""
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix)
    ax.figure.colorbar(image, ax=ax)

    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, matrix[i, j], ha="center", va="center")

    fig.tight_layout()
    return fig, ax


def plot_accuracy_by_k(k_values: list[int], train_accuracy: list[float], test_accuracy: list[float]):
    """Plot train/test accuracy across candidate k values."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_values, train_accuracy, marker="o", label="Training accuracy")
    ax.plot(k_values, test_accuracy, marker="o", label="Test accuracy")
    ax.set_xlabel("k")
    ax.set_ylabel("Accuracy")
    ax.set_title("KNN Accuracy by k")
    ax.legend()
    fig.tight_layout()
    return fig, ax
