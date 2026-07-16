import numpy as np
import pytest

from exomanifold.astronomy import ExoLightCurve


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def sample_lightcurve() -> ExoLightCurve:
    """Return a simple valid light curve."""

    return ExoLightCurve(
        time=np.array([0.0, 1.0, 2.0, 3.0]),
        flux=np.array([1.0, 0.9, 1.1, 1.0]),
        flux_err=np.array([0.01, 0.01, 0.01, 0.01]),
        quality=np.array([0, 0, 0, 0], dtype=np.int32),
        target="Kepler-10",
        mission="Kepler",
    )


# ---------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------

def test_constructor_creates_lightcurve(sample_lightcurve):

    assert isinstance(sample_lightcurve, ExoLightCurve)


def test_constructor_invalid_time_dimension():

    with pytest.raises(ValueError):
        ExoLightCurve(
            time=np.array([[0.0, 1.0]]),
            flux=np.array([1.0, 2.0]),
        )


def test_constructor_invalid_flux_dimension():

    with pytest.raises(ValueError):
        ExoLightCurve(
            time=np.array([0.0, 1.0]),
            flux=np.array([[1.0, 2.0]]),
        )


def test_constructor_mismatched_lengths():

    with pytest.raises(ValueError):
        ExoLightCurve(
            time=np.array([0.0, 1.0]),
            flux=np.array([1.0]),
        )


def test_constructor_invalid_flux_error_length():

    with pytest.raises(ValueError):
        ExoLightCurve(
            time=np.array([0.0, 1.0]),
            flux=np.array([1.0, 2.0]),
            flux_err=np.array([0.01]),
        )


def test_constructor_invalid_quality_length():

    with pytest.raises(ValueError):
        ExoLightCurve(
            time=np.array([0.0, 1.0]),
            flux=np.array([1.0, 2.0]),
            quality=np.array([0]),
        )


# ---------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------

def test_length(sample_lightcurve):

    assert len(sample_lightcurve) == 4


# ---------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------

def test_repr_contains_target(sample_lightcurve):

    representation = repr(sample_lightcurve)

    assert "Kepler-10" in representation
    assert "Kepler" in representation


# ---------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------

def test_copy_returns_new_object(sample_lightcurve):

    copied = sample_lightcurve.copy()

    assert copied is not sample_lightcurve

    assert np.array_equal(
        copied.time,
        sample_lightcurve.time,
    )

    assert np.array_equal(
        copied.flux,
        sample_lightcurve.flux,
    )


# ---------------------------------------------------------------------
# to_numpy
# ---------------------------------------------------------------------

def test_to_numpy_returns_flux(sample_lightcurve):

    array = sample_lightcurve.to_numpy()

    assert np.array_equal(
        array,
        sample_lightcurve.flux,
    )

def test_lightcurve_is_immutable(sample_lightcurve):

    with pytest.raises(AttributeError):
        sample_lightcurve.target = "Kepler-20"