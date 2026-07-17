import numpy as np
import pytest

from exomanifold.astronomy import (
    PhaseFoldedLightCurve,
    PhaseFolder
)

from exomanifold.astronomy.lightcurve import ExoLightCurve

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

def test_compute_phase():

    curve = ExoLightCurve(
        time=np.array([0.0, 2.5, 5.0, 7.5]),
        flux=np.ones(4),
    )

    folder = PhaseFolder(
        period=5.0,
        epoch=0.0,
    )

    phase = folder.compute_phase(curve)

    expected = np.array([0.0, 0.5, 0.0, 0.5])

    np.testing.assert_allclose(phase, expected)

def test_compute_phase_shifted_epoch():

    curve = ExoLightCurve(
        time=np.array([1.0, 3.5, 6.0]),
        flux=np.ones(3),
    )

    folder = PhaseFolder(
        period=5.0,
        epoch=1.0,
    )

    phase = folder.compute_phase(curve)

    expected = np.array([0.0, 0.5, 0.0])

    np.testing.assert_allclose(phase, expected)

def test_sort_phase_already_sorted():
    folder = PhaseFolder(period = 10.0, epoch=0.0)

    phase = np.array([0.1, 0.2, 0.3])
    flux = np.array([10.0,20.0,30.0])

    sorted_phase, sorted_flux, sorted_err = folder.sort_phase(phase, flux)

    np.testing.assert_array_equal(sorted_phase, phase)
    np.testing.assert_array_equal(sorted_flux, flux)
    
    assert sorted_err is None

def test_sorted_phase_reverse():

    folder = PhaseFolder(period = 10.0, epoch = 0.0)

    phase = np.array([0.9,0.5,0.1])
    flux = np.array([9.0,5.0,1.0])
    
    sorted_phase, sorted_flux,_ = folder.sort_phase(phase, flux)


    np.testing.assert_array_equal(sorted_phase, np.array([0.1,0.5,0.9]))

    np.testing.assert_array_equal(sorted_flux, np.array([1.0,5.0,9.0]))

def test_sort_phase_with_flux_errors():
    folder = PhaseFolder(period = 10.0, epoch = 0.0)

    phase = np.array([0.5,0.1,0.3])
    flux = np.array([5.0,1.0,3.0])
    flux_err = np.array([0.5,0.1,0.3])

    p, f, e = folder.sort_phase(phase, flux, flux_err)

    np.testing.assert_array_equal(p, np.array([0.1,0.3,0.5]))

    np.testing.assert_array_equal(f, np.array([1.0,3.0,5.0]))

    np.testing.assert_array_equal(e, np.array([0.1,0.3,0.5]))


def test_sort_phase_mismatched_lengths():
    folder = PhaseFolder(period = 10.0, epoch = 0.0)

    with pytest.raises(ValueError):
        folder.sort_phase(
            np.array([0.1,0.3]),
            np.array([1.0])      
        )


def test_bin_phase():

    folder = PhaseFolder(
        period=10.0,
        epoch=0.0,
        bins=2,
    )

    curve = folder.bin_phase(
        phase=np.array([0.1, 0.2, 0.7, 0.8]),
        flux=np.array([1.0, 2.0, 3.0, 4.0]),
    )

    np.testing.assert_allclose(
        curve.phase,
        np.array([0.15, 0.75]),
    )

    np.testing.assert_allclose(
        curve.flux,
        np.array([1.5, 3.5]),
    )

def test_bin_phase_skips_empty_bins():

    folder = PhaseFolder(
        period=10.0,
        epoch=0.0,
        bins=5,
    )

    curve = folder.bin_phase(
        phase=np.array([0.10, 0.15]),
        flux=np.array([1.0, 2.0]),
    )

    assert len(curve) == 1

def test_bin_phase_flux_errors():

    folder = PhaseFolder(
        period=10.0,
        epoch=0.0,
        bins=2,
    )

    curve = folder.bin_phase(
        phase=np.array([0.1, 0.2, 0.7, 0.8]),
        flux=np.array([1, 2, 3, 4]),
        flux_err=np.array([0.1, 0.2, 0.3, 0.4]),
    )

    assert curve.flux_err is not None

    np.testing.assert_allclose(
        curve.flux_err,
        np.array([0.15, 0.35])
    )

    