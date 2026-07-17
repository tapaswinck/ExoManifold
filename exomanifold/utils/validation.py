"""
Validation utilities used throughout ExoManifold.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from exomanifold.utils.exceptions import NotFittedError

__all__ = [
    "check_array",
    "check_X_y",
    "check_vector",
    "check_is_fitted",
    "check_random_state",
]


def check_array(
    X: NDArray[Any] | None,
) -> NDArray[Any]:
    """
    Validate a two-dimensional feature matrix.

    Parameters
    ----------
    X : ndarray | None
        Input feature matrix.

    Returns
    -------
    ndarray
        Validated feature matrix.

    Raises
    ------
    ValueError
        If the input is None, empty, or not two-dimensional.
    """

    if X is None:
        raise ValueError("Input array cannot be None.")

    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("Input array must be two-dimensional.")

    if X.shape[0] == 0:
        raise ValueError("Input array cannot be empty.")

    return X


def check_X_y(
    X: NDArray[Any] | None,
    y: NDArray[Any] | None,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """
    Validate a feature matrix and target vector.

    Parameters
    ----------
    X : ndarray | None
        Feature matrix.

    y : ndarray | None
        Target vector.

    Returns
    -------
    tuple of ndarray
        Validated feature matrix and target vector.

    Raises
    ------
    ValueError
        If the inputs are invalid.
    """

    X = check_array(X)

    if y is None:
        raise ValueError("Target vector cannot be None.")

    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError("Target vector must be one-dimensional.")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "Feature matrix and target vector must contain the same number of samples."
        )

    return X, y


def check_vector(
    x: NDArray[Any] | None,
    *,
    name: str = "array",
) -> NDArray[Any]:
    """
    Validate a one-dimensional vector.

    Parameters
    ----------
    x : ndarray | None
        Input vector.

    name : str, default="array"
        Name used in error messages.

    Returns
    -------
    ndarray
        Validated vector.

    Raises
    ------
    ValueError
        If the vector is invalid.
    """

    if x is None:
        raise ValueError(f"{name} cannot be None.")

    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if x.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    return x


def check_is_fitted(
    estimator: object,
    attributes: str | Iterable[str],
) -> None:
    """
    Check whether an estimator has been fitted.

    Parameters
    ----------
    estimator : object
        Estimator instance.

    attributes : str or iterable of str
        Attribute name(s) expected after fitting.

    Raises
    ------
    AttributeError
        If a required attribute does not exist.

    NotFittedError
        If a required attribute exists but has value ``None``.
    """

    if isinstance(attributes, str):
        attributes = [attributes]

    for attribute in attributes:
        if not hasattr(estimator, attribute):
            raise AttributeError(
                f"{estimator.__class__.__name__} has no attribute "
                f"'{attribute}'."
            )

        if getattr(estimator, attribute) is None:
            raise NotFittedError(
                f"{estimator.__class__.__name__} has not been fitted."
            )


def check_random_state(
    seed: int | np.random.Generator | None,
) -> np.random.Generator:
    """
    Return a NumPy random number generator.

    Parameters
    ----------
    seed : int, Generator, or None
        Seed or existing random number generator.

    Returns
    -------
    numpy.random.Generator
        Random number generator.

    Raises
    ------
    TypeError
        If the seed has an invalid type.
    """

    if seed is None:
        return np.random.default_rng()

    if isinstance(seed, np.random.Generator):
        return seed

    if isinstance(seed, int):
        return np.random.default_rng(seed)

    raise TypeError(
        "seed must be an int, numpy.random.Generator, or None."
    )