"""Data loading and validation utilities for the KNN classifier project."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def read_xy(file_path: str | Path, header: int | None = None) -> np.ndarray:
    """Read a CSV or Excel file where the final column contains class labels.

    Parameters
    ----------
    file_path:
        Path to a `.csv`, `.xlsx`, or `.xls` file.
    header:
        Header row passed through to pandas. Use `None` for files without headers
        and `0` for files where the first row contains column names.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        data_frame = pd.read_csv(file_path, header=header)
    elif suffix in {".xlsx", ".xls"}:
        data_frame = pd.read_excel(file_path, header=header)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return data_frame.to_numpy()


def validate_xy(xy: np.ndarray, allowed_labels: Iterable[str]) -> np.ndarray:
    """Remove rows with non-finite numeric features or invalid class labels."""
    allowed_labels = set(allowed_labels)
    x = np.asarray(xy[:, :-1], dtype=float)
    y = xy[:, -1]

    valid_x = np.all(np.isfinite(x), axis=1)
    valid_y = np.isin(y, list(allowed_labels))
    return xy[valid_x & valid_y]


def train_test_split(
    xy: np.ndarray,
    train_ratio: float = 0.8,
    random_state: int = 713,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle and split an XY array into train and test arrays."""
    if not 0 <= train_ratio <= 1:
        raise ValueError("train_ratio must be between 0 and 1")

    rng = np.random.default_rng(random_state)
    shuffled = xy.copy()
    rng.shuffle(shuffled)

    x = np.asarray(shuffled[:, :-1], dtype=float)
    y = shuffled[:, -1]
    split_idx = int(len(x) * train_ratio)
    return x[:split_idx], x[split_idx:], y[:split_idx], y[split_idx:]


def class_distribution(y: np.ndarray, labels: Iterable[str]) -> dict[str, dict[str, float]]:
    """Return counts and percentages for each class label."""
    result = {}
    for label in labels:
        count = int(np.sum(y == label))
        result[str(label)] = {"count": count, "percent": count / len(y)}
    return result
