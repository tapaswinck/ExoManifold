import numpy as np
import pytest

from exomanifold.utils.exceptions import NotFittedError
from exomanifold.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
    check_X_y,
)


# ---------------------------------------------------------------------
# check_array
# ---------------------------------------------------------------------

def test_check_array_accepts_valid_array():
    X = np.array([[1, 2], [3, 4]])

    result = check_array(X)

    assert np.array_equal(result, X)


def test_check_array_rejects_none():
    with pytest.raises(ValueError):
        check_array(None)


def test_check_array_rejects_one_dimensional_array():
    X = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        check_array(X)


def test_check_array_rejects_empty_array():
    X = np.empty((0, 2))

    with pytest.raises(ValueError):
        check_array(X)


# ---------------------------------------------------------------------
# check_X_y
# ---------------------------------------------------------------------

def test_check_X_y_accepts_valid_input():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])

    X_checked, y_checked = check_X_y(X, y)

    assert np.array_equal(X_checked, X)
    assert np.array_equal(y_checked, y)


def test_check_X_y_rejects_different_lengths():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0])

    with pytest.raises(ValueError):
        check_X_y(X, y)


def test_check_X_y_rejects_two_dimensional_y():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([[0], [1]])

    with pytest.raises(ValueError):
        check_X_y(X, y)


def test_check_X_y_rejects_none_y():
    X = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        check_X_y(X, None)


# ---------------------------------------------------------------------
# check_is_fitted
# ---------------------------------------------------------------------

class DummyEstimator:
    weights: np.ndarray | None

    def __init__(self):
        self.weights = None


def test_check_is_fitted_accepts_fitted_estimator():
    estimator = DummyEstimator()
    estimator.weights = np.array([1.0, 2.0, 3.0])

    check_is_fitted(estimator, "weights")


def test_check_is_fitted_rejects_missing_attribute():
    estimator = DummyEstimator()

    with pytest.raises(AttributeError):
        check_is_fitted(estimator, "bias")


def test_check_is_fitted_rejects_none_attribute():
    estimator = DummyEstimator()

    with pytest.raises(NotFittedError):
        check_is_fitted(estimator, "weights")


# ---------------------------------------------------------------------
# check_random_state
# ---------------------------------------------------------------------

def test_check_random_state_with_none():
    rng = check_random_state(None)

    assert isinstance(rng, np.random.Generator)


def test_check_random_state_with_integer():
    rng1 = check_random_state(42)
    rng2 = check_random_state(42)

    assert np.array_equal(
        rng1.random(5),
        rng2.random(5),
    )


def test_check_random_state_with_generator():
    generator = np.random.default_rng(42)

    assert check_random_state(generator) is generator


def test_check_random_state_rejects_invalid_seed():
    with pytest.raises(TypeError):
        check_random_state("forty-two")  # type: ignore[arg-type]