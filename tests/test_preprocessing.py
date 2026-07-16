import numpy as np
import pytest

from exomanifold.astronomy import ExoLightCurve
from exomanifold.astronomy.preprocessing import ExoProcessor


@pytest.fixture
def sample_curve():

    return ExoLightCurve(
        time=np.array([0., 1., 2., 3., 4.]),
        flux=np.array([1.0, np.nan, 0.99, 1.01, 1.02]),
        flux_err=np.array([0.01]*5),
        quality=np.array([0, 0, 1, 0, 0]),
        target="Kepler-10",
        mission="Kepler",
    )


def test_preprocessor_creation():

    preprocessor = ExoProcessor()

    assert isinstance(preprocessor, ExoProcessor)

def test_remove_nans(sample_curve):

    preprocessor = ExoProcessor()

    cleaned = preprocessor.remove_nans(sample_curve)

    assert len(cleaned) == 4

    assert not np.isnan(cleaned.flux).any()

def test_remove_quality_flags(sample_curve):

    preprocessor = ExoProcessor()

    cleaned = preprocessor.remove_quality_flags(sample_curve)

    assert np.all(cleaned.quality == 0)

    assert len(cleaned) == 4


def test_normalize_flux():

    curve = ExoLightCurve(
        time=np.arange(5.0),
        flux=np.array([10., 10., 10., 10., 10.]),
        target="Test",
        mission="Kepler",
    )

    preprocessor = ExoProcessor()

    normalized = preprocessor.normalize_flux(curve)

    assert np.allclose(normalized.flux, 1.0)

def test_preprocessing_does_not_modify_original(sample_curve):

    preprocessor = ExoProcessor()

    original = sample_curve.flux.copy()

    _ = preprocessor.remove_nans(sample_curve)

    assert np.array_equal(
        sample_curve.flux,
        original,
        equal_nan=True,
    )

def test_sigma_clip_removes_outlier():

    curve = ExoLightCurve(
        time=np.arange(6.0),
        flux=np.array([1.0, 1.0, 1.0, 100.0, 1.0, 1.0]),
        target="Test",
        mission="Kepler",
    )

    preprocessor = ExoProcessor()

    clipped = preprocessor.sigma_clip(
        curve,
        sigma=2.0,
    )

    assert len(clipped) == 5

    assert np.max(clipped.flux) < 100.0


def test_sigma_clip_keeps_clean_curve():

    curve = ExoLightCurve(
        time=np.arange(5.0),
        flux=np.ones(5),
        target="Test",
        mission="Kepler",
    )

    preprocessor = ExoProcessor()

    clipped = preprocessor.sigma_clip(curve)

    assert len(clipped) == len(curve)


def test_sigma_clip_does_not_modify_original(sample_curve):

    preprocessor = ExoProcessor()

    original = sample_curve.flux.copy()

    _ = preprocessor.sigma_clip(sample_curve)

    assert np.array_equal(
        sample_curve.flux,
        original,
        equal_nan=True,
    )