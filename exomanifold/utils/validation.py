"""
Validation utilities used in ExoManifold.
"""


from __future__ import annotations

__all__ = [
    "check_array",
    "check_X_y",
    "check_is_fitted",
    "check_random_state"
]


from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from exomanifold.utils.exceptions import NotFittedError

def check_array(
        X: NDArray[Any] | None,
)-> NDArray[Any]:
    """
    Validate a feature matrix.

    Parameters
    ----------
    X: NDArray | None
        Input feature matrix.

    Returns
    -------
    NDArray
        Validated feature matrix.

     Raises
     ------
     ValueError
        If the input is None, empty, or not two-dimensional.   
    """

    if X is None:
        raise ValueError("Input array cannot be empty.")
    
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("Input array must be 2-dimenaional.")
    
    if X.shape[0] == 0:
        raise ValueError("Inpur array cannot be empty.")
    
    return X

def check_X_y(
        X: NDArray[Any] | None,
        y: NDArray[Any] | None
)-> tuple[NDArray[Any], NDArray[Any]]:
    """
    Validate a feature matrix and target vector.

    Parameters
    ----------

    X: NDArray | None
        Feature matrix.

    y: NDArray | None
        Target vector.

    Returns
    -------
    tuple[NDarray, NDArray]
        Validated feature matrix and target vector.

    Raises
    ------
    ValueError
        If y is None, not one-dimensional, or has the wrong number of samples.
    """

    X = check_array(X)

    if y is None:
        raise ValueError("Target vector cannot be None.")
    
    y = np.asarray(y)

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
        Attribute name(s) expected after fitting.

    Raises
    ------
    
    AttributeError
        If a required attribute does not exist.

    NotFittedError
        If a required attribute exists but is None.
    """

    if isinstance(attributes, str):
        attributes = [attributes]

        for attribute in attributes:
            if not hasattr(estimator, attribute):
                raise AttributeError(
                    f"{estimator.__class__.__name__} has no attribute '{attribute}'."
                )


            if getattr(estimator, attribute) is None:
                raise NotFittedError(
                    f"{estimator.__class__.__name__} has not been fitted."
                )
            

def check_random_state(
        seed: int | np.random.Generator | None
)-> np.random.Generator:
    """
    Return a NumPy random number generator.

    Parameters
    ----------
    seed: int | Generator | None
        Seed or existing Generator

    
    Returns
    -------
    numpy.random.Generator
        Random number generator.
    
    Raises
    ------
    TypeError
        If seed has an invalid type.
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

