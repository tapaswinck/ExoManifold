"""
Tests for Principal Component Analysis.
"""

from __future__ import annotations

import numpy as np
import pytest

from exomanifold.utils.exceptions import NotFittedError
from exomanifold.manifold.pca import PCA


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_data(rng):
    return rng.normal(size=(100, 5))


# ---------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------

def test_invalid_n_components():

    with pytest.raises(ValueError):
        PCA(0)


# ---------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------

def test_fit(sample_data):

    model = PCA(n_components=2)
    model.fit(sample_data)

    assert model.is_fitted
    assert model.components_.shape == (2, 5)
    assert model.mean_.shape == (5,)
    assert model.embedding_.shape == (100, 2)


def test_too_many_components(sample_data):

    model = PCA(n_components=10)

    with pytest.raises(ValueError):
        model.fit(sample_data)


def test_invalid_input_dimension():

    model = PCA()

    with pytest.raises(ValueError):
        model.fit(np.arange(20))


# ---------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------

def test_fit_transform(sample_data):

    model = PCA(2)

    Y = model.fit_transform(sample_data)

    assert Y.shape == (100, 2)


def test_transform_matches_embedding(sample_data):

    model = PCA(3)
    model.fit(sample_data)

    np.testing.assert_allclose(
        model.transform(sample_data),
        model.embedding_,
    )


def test_transform_before_fit():

    model = PCA()

    with pytest.raises(NotFittedError):
        model.transform(np.random.rand(10, 5))


# ---------------------------------------------------------------------
# Inverse Transform
# ---------------------------------------------------------------------

def test_inverse_transform_shape(sample_data):

    model = PCA(2)

    Y = model.fit_transform(sample_data)

    reconstructed = model.inverse_transform(Y)

    assert reconstructed.shape == sample_data.shape


def test_inverse_before_fit():

    model = PCA()

    with pytest.raises(NotFittedError):
        model.inverse_transform(np.random.rand(20, 2))


def test_full_rank_reconstruction(sample_data):

    model = PCA(n_components=5)

    Y = model.fit_transform(sample_data)

    reconstructed = model.inverse_transform(Y)

    np.testing.assert_allclose(
        reconstructed,
        sample_data,
        atol=1e-10,
    )


# ---------------------------------------------------------------------
# Explained Variance
# ---------------------------------------------------------------------

def test_variance_shapes(sample_data):

    model = PCA(3)
    model.fit(sample_data)

    assert model.explained_variance_.shape == (3,)
    assert model.explained_variance_ratio_.shape == (3,)
    assert model.singular_values_.shape == (3,)


def test_variance_ratio_sums_to_one(sample_data):

    model = PCA(5)
    model.fit(sample_data)

    np.testing.assert_allclose(
        model.explained_variance_ratio_.sum(),
        1.0,
        atol=1e-12,
    )


def test_singular_values_sorted(sample_data):

    model = PCA(5)
    model.fit(sample_data)

    assert np.all(np.diff(model.singular_values_) <= 0)


# ---------------------------------------------------------------------
# Linear Algebra Properties
# ---------------------------------------------------------------------

def test_components_are_orthonormal(sample_data):

    model = PCA(5)
    model.fit(sample_data)

    identity = model.components_ @ model.components_.T

    np.testing.assert_allclose(
        identity,
        np.eye(5),
        atol=1e-10,
    )


def test_mean_centering(sample_data):

    model = PCA(2)
    model.fit(sample_data)

    centered = sample_data - model.mean_

    np.testing.assert_allclose(
        centered.mean(axis=0),
        np.zeros(sample_data.shape[1]),
        atol=1e-12,
    )


# ---------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------

def test_single_feature():

    rng = np.random.default_rng(123)

    X = rng.normal(size=(50, 1))

    model = PCA(1)

    Y = model.fit_transform(X)

    assert Y.shape == (50, 1)


def test_constant_feature():

    X = np.ones((40, 3))

    model = PCA(3)

    model.fit(X)

    np.testing.assert_allclose(
        model.embedding_,
        np.zeros((40, 3)),
        atol=1e-12,
    )