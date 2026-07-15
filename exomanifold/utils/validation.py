"""
Validation utilities used thruoghout ExoManifold.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray

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