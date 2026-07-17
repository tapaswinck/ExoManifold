"""
Tests for transit detection.
"""

from __future__ import annotations

import numpy as np
import pytest

from exomanifold.astronomy import (
    ExoLightCurve,
    TransitDetector,
    TransitEvent,
    TransitSearchResult,
)




# ------------------------------------------------------------
# TransitEvent
# ------------------------------------------------------------

def test_transit_event_constructor():

    event = TransitEvent(
        period=10.0,
        epoch=2.5,
        duration=0.25,
        depth=0.01,
        snr=15.0,
        power=24.3,
    )

    assert event.period == 10.0
    assert event.epoch == 2.5
    assert event.duration == 0.25
    assert event.depth == 0.01
    assert event.snr == 15.0
    assert event.power == 24.3


def test_transit_event_negative_period():

    with pytest.raises(ValueError):
        TransitEvent(
            period=-1.0,
            epoch=0.0,
            duration=0.2,
            depth=0.01,
            snr=5.0,
            power=1.0,
        )


def test_transit_event_negative_duration():

    with pytest.raises(ValueError):
        TransitEvent(
            period=10.0,
            epoch=0.0,
            duration=-0.5,
            depth=0.01,
            snr=5.0,
            power=1.0,
        )


def test_transit_event_negative_depth():

    with pytest.raises(ValueError):
        TransitEvent(
            period=10.0,
            epoch=0.0,
            duration=0.1,
            depth=-0.01,
            snr=5.0,
            power=1.0,
        )


# ------------------------------------------------------------
# TransitSearchResult
# ------------------------------------------------------------

def test_search_result_constructor():

    event = TransitEvent(
        period=10.0,
        epoch=0.0,
        duration=0.1,
        depth=0.01,
        snr=12.0,
        power=25.0,
    )

    result = TransitSearchResult(
        best_event=event,
        events=[event],
        period_grid=np.linspace(1, 20, 100),
        power=np.random.random(100),
        algorithm="BLS",
    )

    assert result.best_event is event
    assert len(result.events) == 1
    assert result.algorithm == "BLS"


def test_search_result_length():

    event = TransitEvent(
        period=5,
        epoch=0,
        duration=0.1,
        depth=0.01,
        snr=5,
        power=10,
    )

    result = TransitSearchResult(
        best_event=event,
        events=[event],
        period_grid=np.arange(5),
        power=np.arange(5),
        algorithm="BLS",
    )

    assert len(result) == 5


# ------------------------------------------------------------
# TransitDetector
# ------------------------------------------------------------

def test_detector_constructor():

    detector = TransitDetector(
        minimum_period=1.0,
        maximum_period=20.0,
        n_periods=500,
    )

    assert detector.minimum_period == 1.0
    assert detector.maximum_period == 20.0
    assert detector.n_periods == 500


def test_detector_invalid_minimum_period():

    with pytest.raises(ValueError):
        TransitDetector(
            minimum_period=0.0,
            maximum_period=10.0,
        )


def test_detector_invalid_maximum_period():

    with pytest.raises(ValueError):
        TransitDetector(
            minimum_period=5.0,
            maximum_period=2.0,
        )


def test_detector_invalid_number_periods():

    with pytest.raises(ValueError):
        TransitDetector(
            minimum_period=1.0,
            maximum_period=20.0,
            n_periods=1,
        )


def test_period_grid():

    detector = TransitDetector(
        minimum_period=1,
        maximum_period=5,
        n_periods=5,
    )

    grid = detector.compute_period_grid()

    np.testing.assert_allclose(
        grid,
        np.array([1., 2., 3., 4., 5.]),
    )


def test_period_grid_length():

    detector = TransitDetector(
        minimum_period=1,
        maximum_period=50,
        n_periods=500,
    )

    assert len(detector.compute_period_grid()) == 500


def test_search_runs():

    detector = TransitDetector()

    curve = ExoLightCurve(
        time=np.arange(100, dtype=float),
        flux=np.ones(100),
    )

    result = detector.search(curve)

    assert isinstance(result, TransitSearchResult)


def test_detector_repr():

    detector = TransitDetector()

    assert "TransitDetector" in repr(detector)


def test_search_result_repr():

    event = TransitEvent(
        period=2,
        epoch=0,
        duration=0.1,
        depth=0.01,
        snr=10,
        power=20,
    )

    result = TransitSearchResult(
        best_event=event,
        events=[event],
        period_grid=np.arange(3),
        power=np.arange(3),
        algorithm="BLS",
    )

    assert "TransitSearchResult" in repr(result)

def test_bls_periodogram_runs():

    detector = TransitDetector()

    time = np.linspace(0.0, 30.0, 1000)
    flux = np.ones_like(time)

    curve = ExoLightCurve(
        time=time,
        flux=flux,
    )

    result = detector.search(curve)

    assert len(result.period_grid) == detector.n_periods
    assert len(result.power) == detector.n_periods

        
def test_validate_curve_too_short():

    detector = TransitDetector()

    curve = ExoLightCurve(
        time=np.arange(5, dtype=float),
        flux=np.ones(5),
    )

    with pytest.raises(ValueError):
        detector._validate_curve(curve)

def test_validate_curve_nan_time():

    detector = TransitDetector()

    time = np.arange(20, dtype=float)
    time[5] = np.nan

    curve = ExoLightCurve(
        time=time,
        flux=np.ones(20),
    )

    with pytest.raises(ValueError):
        detector._validate_curve(curve)

def test_validate_curve_nan_flux():

    detector = TransitDetector()

    flux = np.ones(20)
    flux[3] = np.nan

    curve = ExoLightCurve(
        time=np.arange(20, dtype=float),
        flux=flux,
    )

    with pytest.raises(ValueError):
        detector._validate_curve(curve)

def test_validate_curve_nan_flux_error():

    detector = TransitDetector()

    flux_err = np.ones(20)
    flux_err[7] = np.nan

    curve = ExoLightCurve(
        time=np.arange(20, dtype=float),
        flux=np.ones(20),
        flux_err=flux_err,
    )

    with pytest.raises(ValueError):
        detector._validate_curve(curve)

def test_validate_curve_unsorted_time():

    detector = TransitDetector()

    curve = ExoLightCurve(
        time=np.array([0.0, 2.0, 1.0, 3.0]),
        flux=np.ones(4),
    )

    with pytest.raises(ValueError):
        detector._validate_curve(curve)

def test_search_returns_event():

    detector = TransitDetector()

    time = np.linspace(0.0, 30.0, 1000)
    flux = np.ones_like(time)

    curve = ExoLightCurve(
        time=time,
        flux=flux,
    )

    result = detector.search(curve)

    assert result.best_event.period > 0
    assert result.best_event.duration > 0
    assert result.algorithm == "BLS"