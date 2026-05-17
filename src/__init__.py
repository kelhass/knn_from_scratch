"""Utilities for the KNN classifier project."""

from .knn import KNNClassifier, euclidean_distance
from .data import read_xy, validate_xy, train_test_split
from .metrics import binary_metrics, confusion_matrix, accuracy_score
