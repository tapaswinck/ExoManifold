"""
Principal Component Analysis (PCA).

This implementation follows the scikit-learn estimator API while
remaining completely NumPy-based.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from exomanifold.manifold.base import BaseEmbedding
from exomanifold.utils.validation import check_array, check_is_fitted

__all__ = ["PCA"]


class PCA(BaseEmbedding):
    """
    Principal Component Analysis using Singular Value Decomposition.

    Parameters
    ----------
    n_components : int, default=2
        Number of principal components.
    """

    def __init__(
        self,
        n_components: int = 2,
    ) -> None:
        super().__init__(n_components=n_components)

        self.mean_: NDArray[np.float64] | None = None
        self.components_: NDArray[np.float64] | None = None
        self.explained_variance_: NDArray[np.float64] | None = None
        self.explained_variance_ratio_: NDArray[np.float64] | None = None
        self.singular_values_: NDArray[np.float64] | None = None
        self.n_features_in_: int | None = None

    def fit(
        self,
        X: NDArray[np.float64],
    ) -> "PCA":
        """
        Fit PCA using Singular Value Decomposition.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        PCA
            Fitted estimator.
        """

        X = self._validate_data(X)

        n_samples, n_features = X.shape

        if self.n_components > n_features:
            raise ValueError(
                "n_components cannot exceed the number of features."
            )

        self.n_features_in_ = n_features

        mean = X.mean(axis=0)
        X_centered = X - mean

        _, singular_values, Vt = np.linalg.svd(
            X_centered,
            full_matrices=False,
        )

        explained_variance = (singular_values ** 2) / (n_samples - 1)

        total_variance = explained_variance.sum()

        if total_variance == 0:
            explained_variance_ratio = np.zeros_like(explained_variance)
        else:
            explained_variance_ratio = (
                explained_variance / total_variance
            )

        components = Vt[: self.n_components]

        embedding = X_centered @ components.T

        self.mean_ = mean
        self.components_ = components
        self.embedding_ = embedding

        self.singular_values_ = singular_values[: self.n_components]
        self.explained_variance_ = explained_variance[: self.n_components]
        self.explained_variance_ratio_ = (
            explained_variance_ratio[: self.n_components]
        )

        return self

    def transform(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Project data into principal component space.
        """

        check_is_fitted(
            self,
            [
                "mean_",
                "components_",
            ],
        )

        X = self._validate_data(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Input has a different number of features than "
                "the data used during fitting."
            )

        assert self.mean_ is not None
        assert self.components_ is not None

        X_centered = X - self.mean_

        return X_centered @ self.components_.T

    def inverse_transform(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Transform data back into the original feature space.
        """

        check_is_fitted(
            self,
            [
                "mean_",
                "components_",
            ],
        )

        X = check_array(X)

        if X.shape[1] != self.n_components:
            raise ValueError(
                "Input has incorrect number of principal components."
            )

        assert self.mean_ is not None
        assert self.components_ is not None

        return X @ self.components_ + self.mean_

    def fit_transform(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Fit the model and return the embedding.
        """

        self.fit(X)

        assert self.embedding_ is not None

        return self.embedding_

    @property
    def explained_variance(self) -> NDArray[np.float64]:
        """
        Explained variance of each selected component.
        """

        check_is_fitted(self, "explained_variance_")

        assert self.explained_variance_ is not None

        return self.explained_variance_

    @property
    def explained_variance_ratio(
        self,
    ) -> NDArray[np.float64]:
        """
        Fraction of variance explained by each component.
        """

        check_is_fitted(self, "explained_variance_ratio_")

        assert self.explained_variance_ratio_ is not None

        return self.explained_variance_ratio_

    def __repr__(self) -> str:
        return (
            f"PCA("
            f"n_components={self.n_components}"
            f")"
        )
    
    