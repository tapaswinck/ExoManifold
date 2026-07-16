import numpy as np
import pytest

from exomanifold.astronomy import (
    PhaseFoldedLightCurve,
    PhaseFolder
)



def test_constructor_valid():

    curve = PhaseFoldedLightCurve(
        phase=np.array([0.1, 0.2, 0.3]),
        flux=np.array([1.0, 0.9, 1.1]),
        period=10.0,
        epoch=0.0,
    )

    assert len(curve) == 3
    assert curve.n_samples == 3
    assert not curve.has_flux_errors


def test_constructor_invalid_phase_dimension():

    with pytest.raises(ValueError):

        PhaseFoldedLightCurve(
            phase=np.array([[0.1, 0.2]]),
            flux=np.array([1.0, 1.0]),
            period=10.0,
            epoch=0.0,
        )


def test_constructor_invalid_flux_dimension():

    with pytest.raises(ValueError):

        PhaseFoldedLightCurve(
            phase=np.array([0.1, 0.2]),
            flux=np.array([[1.0, 1.0]]),
            period=10.0,
            epoch=0.0,
        )


def test_constructor_mismatched_lengths():

    with pytest.raises(ValueError):

        PhaseFoldedLightCurve(
            phase=np.array([0.1, 0.2]),
            flux=np.array([1.0]),
            period=10.0,
            epoch=0.0,
        )


def test_constructor_invalid_flux_error_length():

    with pytest.raises(ValueError):

        PhaseFoldedLightCurve(
            phase=np.array([0.1, 0.2]),
            flux=np.array([1.0, 1.1]),
            flux_err=np.array([0.1]),
            period=10.0,
            epoch=0.0,
        )


def test_constructor_invalid_period():

    with pytest.raises(ValueError):

        PhaseFoldedLightCurve(
            phase=np.array([0.1, 0.2]),
            flux=np.array([1.0, 1.0]),
            period=0.0,
            epoch=0.0,
        )


def test_constructor_negative_period():

    with pytest.raises(ValueError):

        PhaseFoldedLightCurve(
            phase=np.array([0.1, 0.2]),
            flux=np.array([1.0, 1.0]),
            period=-5.0,
            epoch=0.0,
        )


def test_constructor_invalid_phase_range():

    with pytest.raises(ValueError):

        PhaseFoldedLightCurve(
            phase=np.array([-0.1, 0.4]),
            flux=np.array([1.0, 1.0]),
            period=5.0,
            epoch=0.0,
        )


def test_constructor_invalid_phase_range_above_one():

    with pytest.raises(ValueError):

        PhaseFoldedLightCurve(
            phase=np.array([0.1, 1.2]),
            flux=np.array([1.0, 1.0]),
            period=5.0,
            epoch=0.0,
        )


def test_has_flux_errors():

    curve = PhaseFoldedLightCurve(
        phase=np.array([0.1, 0.2]),
        flux=np.array([1.0, 1.0]),
        flux_err=np.array([0.01, 0.01]),
        period=5.0,
        epoch=0.0,
    )

    assert curve.has_flux_errors


def test_sort():

    curve = PhaseFoldedLightCurve(
        phase=np.array([0.8, 0.2, 0.5]),
        flux=np.array([8.0, 2.0, 5.0]),
        period=10.0,
        epoch=0.0,
    )

    sorted_curve = curve.sort()

    assert np.allclose(
        sorted_curve.phase,
        [0.2, 0.5, 0.8],
    )

    assert np.allclose(
        sorted_curve.flux,
        [2.0, 5.0, 8.0],
    )


def test_copy():

    curve = PhaseFoldedLightCurve(
        phase=np.array([0.1, 0.2]),
        flux=np.array([1.0, 1.1]),
        metadata={"planet": "Kepler-10 b"},
        period=10.0,
        epoch=0.0,
    )

    copied = curve.copy()

    assert copied is not curve

    assert np.array_equal(
        copied.phase,
        curve.phase,
    )

    assert copied.metadata == curve.metadata

    copied.metadata["planet"] = "Changed"

    assert curve.metadata["planet"] == "Kepler-10 b"



def test_phasefolder_constructor():

    folder = PhaseFolder(
        period=10.0,
        epoch=0.5,
        bins=200,
    )

    assert folder.period == 10.0
    assert folder.epoch == 0.5
    assert folder.bins == 200

def test_phasefolder_invalid_period():

    with pytest.raises(ValueError):
        PhaseFolder(
            period=0.0,
            epoch=0.0,
        )

def test_phasefolder_invalid_bins():

    with pytest.raises(ValueError):
        PhaseFolder(
            period=10.0,
            epoch=0.0,
            bins=1,
        )
