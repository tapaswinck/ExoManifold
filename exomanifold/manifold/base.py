"""
base classes for manifold learning algorithms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from exomanifold.utils.validation import (
    check_array,
    check_is_fitted
    )

__all__ = [
    "BaseEmbedding"
]

class BaseEmbedding(ABC):
    """
    Abstract base class for embedding algorithms.

    This class tries to follow the scikit-learn estimator API.
    """

    def __init__(
            self,
            n_components: int = 2
    )-> None:
        
        if n_components < 1:
            raise ValueError("n_components must be positive.")
        
        self.n_components = int(n_components)

        self.embedding_: NDArray[np.float64] | None = None
    
    @abstractmethod
    def fit(
        self,
        X: NDArray[np.float64]
    )->"BaseEmbeding":
        """
        Learn an embedding from X.
        """
    
    @abstractmethod
    def transform(
        self,
        X: NDArray[np.float64]
    )->NDArray[np.float64]:
        """
        Transform X into an embedding space.
        """

    def fit_transform(
            self,
            X: NDArray[np.float64]
    )->NDArray[np.float64]:
        """
        Fit the model and return the embedding.
        """
        self.fit(X)

        check_is_fitted(self, "embedding_")

        return self.embedding_
    
    @property
    def is_fitted(self)-> bool:
        """
        True if the embedding has been learned.
        """

        return self.embedding_ is not None

    def _validate_data(
            self,
            X: NDArray[np.float64]
    )-> NDArray[np.float64]:
        """
        Validate an input feature matrix.
        """
        return check_array(X)


    