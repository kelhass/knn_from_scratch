"""Run the project analysis from the command line."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data import read_xy, validate_xy, train_test_split, class_distribution
from knn import KNNClassifier
from metrics import accuracy_score, binary_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def evaluate_binary() -> pd.DataFrame:
    xy = read_xy(DATA_DIR / "Prog1data.xlsx", header=None)
    xy = validate_xy(xy, {"+", "-"})
    x_train, x_test, y_train, y_test = train_test_split(xy, train_ratio=0.8)

    rows = []
    for k in [5, 15, 25, 35]:
        model = KNNClassifier(k=k).fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = binary_metrics(y_test, predictions)
        rows.append({"k": k, **{m: metrics[m] for m in ["accuracy", "precision", "recall", "f1"]}})
    return pd.DataFrame(rows)


def evaluate_iris() -> pd.DataFrame:
    xy = read_xy(DATA_DIR / "Iris.xlsx", header=0)[:, 1:]
    labels = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"}
    xy = validate_xy(xy, labels)
    x_train, x_test, y_train, y_test = train_test_split(xy, train_ratio=0.8)

    rows = []
    for k in range(5, 100, 10):
        model = KNNClassifier(k=k).fit(x_train, y_train)
        rows.append({
            "k": k,
            "train_accuracy": accuracy_score(y_train, model.predict(x_train)),
            "test_accuracy": accuracy_score(y_test, model.predict(x_test)),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Binary classifier metrics")
    print(evaluate_binary().round(5).to_string(index=False))
    print("\nIris classifier accuracy by k")
    print(evaluate_iris().round(5).to_string(index=False))
