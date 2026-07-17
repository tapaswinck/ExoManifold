
"""
Distance metrics for machine learning algorithms.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from exomanifold.utils.validation import (
    check_array,
    check_vector,
)


__all__ = [
    "euclidean_distance",
    "manhattan_distance",
    "cosine_distance",
    "pairwise_distances",
]

def euclidean_distance(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> float:
    """
    Compute the Euclidean distance between two vectors.
    """

    x = check_vector(x, name="x")
    y = check_vector(y, name="y")

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    return float(np.linalg.norm(x - y))

def manhattan_distance(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> float:
    """
    Compute the Manhattan (L1) distance.
    """

    x = check_vector(x, name="x")
    y = check_vector(y, name="y")

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    return float(np.sum(np.abs(x - y)))

def cosine_distance(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> float:
    """
    Compute cosine distance.

    distance = 1 - cosine similarity
    """

    x = check_vector(x, name="x")
    y = check_vector(y, name="y")

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)

    if norm_x == 0 or norm_y == 0:
        raise ValueError("Vectors must not have zero norm.")

    similarity = np.dot(x, y) / (norm_x * norm_y)

    return float(1.0 - similarity)

def pairwise_distances(
    X: NDArray[np.float64],
    metric: str = "euclidean",
) -> NDArray[np.float64]:
    """
    Compute pairwise distances.

    Parameters
    ----------
    X : ndarray
        Shape (n_samples, n_features)

    metric : str
        "euclidean", "manhattan", or "cosine".
    """

    X = check_array(X)

    n_samples = X.shape[0]

    D = np.zeros((n_samples, n_samples), dtype=float)

    if metric == "euclidean":
        distance = euclidean_distance
    elif metric == "manhattan":
        distance = manhattan_distance
    elif metric == "cosine":
        distance = cosine_distance
    else:
        raise ValueError(f"Unknown metric '{metric}'.")

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            d = distance(X[i], X[j])
            D[i, j] = d
            D[j, i] = d

    return D