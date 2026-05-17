import numpy as np

from src.knn import KNNClassifier, euclidean_distance
from src.metrics import accuracy_score, binary_metrics


def test_euclidean_distance():
    assert euclidean_distance(np.array([0, 0]), np.array([3, 4])) == 5.0


def test_knn_predicts_simple_clusters():
    x = np.array([[0, 0], [0, 1], [5, 5], [6, 5]])
    y = np.array(["A", "A", "B", "B"])
    model = KNNClassifier(k=1).fit(x, y)
    assert model.predict(np.array([[0.2, 0.1], [5.2, 5.1]])).tolist() == ["A", "B"]


def test_binary_metrics():
    metrics = binary_metrics(np.array(["+", "+", "-", "-"]), np.array(["+", "-", "-", "+"]))
    assert metrics["accuracy"] == 0.5
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
