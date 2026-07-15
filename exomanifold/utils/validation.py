"""
Validation utilities used thruoghout ExoManifold.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray
from collections.abc import Iterable


def check_array(X: NDArray[Any] | None)-> NDArray[Any]:
    """
    Validate a feature matrix.

    Parameters
    ----------
    X: numpy.ndarray
        Input feature matrix.

    Returns
    -------
    numpy.ndarray
        Validated array
    """

    if X is None:
        raise ValueError("Input array cannot be None.")
    
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("Input array must be 2-dimensional.")
    
    if X.shape[0] == 0:
        raise ValueError("Input array cannot be empty.")
    
    return X

def check_X_y(
        X: NDArray[Any] | None,
        y: NDArray[Any] | None
)->tuple[NDArray[Any], NDArray[Any]]:
    """
    Validatefeature matrix X and target vactor y.
    """

    X = check_array(X)

    if y is None:
        raise ValueError("Target vector cannot be None.")
    
    y = np.array(y)

    if y.ndim != 1:
        raise ValueError("Target vector must be one-dimensional.")
    
    if X.shape[0] != y.shape[0]:
        raise ValueError("Feature matrix and target vector must contain the same number of samples.")
    

    return X, y

def check_is_fitted(
        estimator: object,
        attributes: str | Iterable[str]
)-> None:
    """
    Check whether an estimator has been fitted.

    Parameters
    ----------
    estimator: object
        Estimator instance.

    attributes: str | Iterable[str]
        Attributes name or iterable of attribute names
        expected after fittting.
    """

    if isinstance(attributes, str):
        attributes = [attributes]

        for attribute in attributes:

            if not hasattr(estimator, attribute):
                raise AttributeError(
                    f"{estimator.__class__.__name__} has not attribute '{attribute}'."
                )
            
            if getattr(estimator, attribute) is None:
                raise ValueError(
                    f"{estimator.__class__.__name__} is not fitted yet."
                )
            


def check_random_state(
        seed: int | np.random.Generator | None
)->np.random.Generator:
    """
    Validate and return a NumPy random number generator.
    """

    if seed is None:
        return np.random.default_rng()
    
    if isinstance(seed, np.random.Generator):
        return seed
    
    if isinstance(seed, int):
        return np.random.default_rng(seed)
    
    raise TypeError(
        "seed must be an int, Generator or None."
    )

