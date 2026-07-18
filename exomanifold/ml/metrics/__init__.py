"""
Distance and similarity metrics.
"""

from .distances import (
    cosine_distance,
    euclidean_distance,
    manhattan_distance,
    pairwise_distances,
)

from .classification import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
__all__ = [
    "euclidean_distance",
    "manhattan_distance",
    "cosine_distance",
    "pairwise_distances",
    "accuracy_score",
    "confusion_matrix",
    "precision_score",
    "recall_score",
    "f1_score",
    "classification_report"
]




