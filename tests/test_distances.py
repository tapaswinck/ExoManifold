import numpy as np
import pytest

from exomanifold.ml.metrics.distances import (
    euclidean_distance,
    manhattan_distance,
    cosine_distance,
    pairwise_distances,
)


# ---------------------------------------------------------------------
# Euclidean distance
# ---------------------------------------------------------------------


def test_euclidean_distance():
    x = np.array([0.0, 0.0])
    y = np.array([3.0, 4.0])

    assert euclidean_distance(x, y) == pytest.approx(5.0)


def test_euclidean_identical_vectors():
    x = np.array([1.0, 2.0, 3.0])

    assert euclidean_distance(x, x) == pytest.approx(0.0)


# ---------------------------------------------------------------------
# Manhattan distance
# ---------------------------------------------------------------------


def test_manhattan_distance():
    x = np.array([1.0, 2.0])
    y = np.array([4.0, 6.0])

    assert manhattan_distance(x, y) == pytest.approx(7.0)


def test_manhattan_identical_vectors():
    x = np.array([5.0, 8.0])

    assert manhattan_distance(x, x) == pytest.approx(0.0)


# ---------------------------------------------------------------------
# Cosine distance
# ---------------------------------------------------------------------


def test_cosine_distance_identical():
    x = np.array([1.0, 2.0, 3.0])

    assert cosine_distance(x, x) == pytest.approx(0.0)


def test_cosine_distance_orthogonal():
    x = np.array([1.0, 0.0])
    y = np.array([0.0, 1.0])

    assert cosine_distance(x, y) == pytest.approx(1.0)


def test_cosine_distance_opposite():
    x = np.array([1.0, 0.0])
    y = np.array([-1.0, 0.0])

    assert cosine_distance(x, y) == pytest.approx(2.0)


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


def test_shape_mismatch_euclidean():
    with pytest.raises(ValueError):
        euclidean_distance(
            np.array([1.0, 2.0]),
            np.array([1.0]),
        )


def test_shape_mismatch_manhattan():
    with pytest.raises(ValueError):
        manhattan_distance(
            np.array([1.0]),
            np.array([1.0, 2.0]),
        )


def test_shape_mismatch_cosine():
    with pytest.raises(ValueError):
        cosine_distance(
            np.array([1.0]),
            np.array([1.0, 2.0]),
        )


def test_cosine_zero_vector():
    with pytest.raises(ValueError):
        cosine_distance(
            np.zeros(3),
            np.ones(3),
        )


# ---------------------------------------------------------------------
# Pairwise distances
# ---------------------------------------------------------------------


def test_pairwise_shape():
    X = np.random.rand(10, 4)

    D = pairwise_distances(X)

    assert D.shape == (10, 10)


def test_pairwise_diagonal_zero():
    X = np.random.rand(15, 3)

    D = pairwise_distances(X)

    np.testing.assert_allclose(
        np.diag(D),
        np.zeros(15),
    )


def test_pairwise_is_symmetric():
    X = np.random.rand(20, 5)

    D = pairwise_distances(X)

    np.testing.assert_allclose(
        D,
        D.T,
        atol=1e-12,
    )


def test_pairwise_single_sample():
    X = np.array([[1.0, 2.0]])

    D = pairwise_distances(X)

    np.testing.assert_allclose(
        D,
        np.array([[0.0]]),
    )


def test_pairwise_manhattan():
    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
    ])

    D = pairwise_distances(
        X,
        metric="manhattan",
    )

    expected = np.array([
        [0.0, 2.0],
        [2.0, 0.0],
    ])

    np.testing.assert_allclose(D, expected)


def test_pairwise_cosine():
    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    D = pairwise_distances(
        X,
        metric="cosine",
    )

    expected = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    np.testing.assert_allclose(D, expected)


def test_invalid_metric():
    X = np.random.rand(5, 2)

    with pytest.raises(ValueError):
        pairwise_distances(
            X,
            metric="unknown",
        )


def test_pairwise_integer_input():
    X = np.array([
        [0, 0],
        [3, 4],
    ])

    D = pairwise_distances(X)

    assert D[0, 1] == pytest.approx(5.0)


def test_pairwise_matches_manual():
    X = np.random.rand(8, 3)

    D = pairwise_distances(X)

    for i in range(len(X)):
        for j in range(len(X)):
            assert D[i, j] == pytest.approx(
                euclidean_distance(X[i], X[j])
            )