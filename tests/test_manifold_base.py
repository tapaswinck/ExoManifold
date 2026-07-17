import numpy as np
import pytest

from exomanifold.manifold.base import BaseEmbedding


class DummyEmbedding(BaseEmbedding):

    def fit(self, X):

        X = self._validate_data(X)

        self.embedding_ = X[:, : self.n_components]

        return self

    def transform(self, X):

        return self._validate_data(X)[:, : self.n_components]


def test_fit_transform():

    X = np.random.rand(20, 5)

    model = DummyEmbedding(n_components=2)

    Y = model.fit_transform(X)

    assert Y.shape == (20, 2)


def test_is_fitted():

    model = DummyEmbedding()

    assert not model.is_fitted

    model.fit(np.random.rand(10, 4))

    assert model.is_fitted


def test_invalid_components():

    with pytest.raises(ValueError):
        DummyEmbedding(n_components=0)