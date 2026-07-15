import pytest

from exomanifold.utils.exceptions import (
    DatasetError,
    ExoManifoldError,
    InvalidLightCurveError,
    NotFittedError
)

def test_base_exception_is_exception():
    assert issubclass(ExoManifoldError, Exception)


def test_not_fitted_error_is_exomanifold_error():
    assert issubclass(NotFittedError, ExoManifoldError)

def test_invalid_lightcurve_error_is_exomanifold_error():
    assert issubclass(InvalidLightCurveError, ExoManifoldError)

def test_dataset_error_is_exomanifold_error():
    assert issubclass(DatasetError, ExoManifoldError)

def test_not_fitted_error_can_be_raised():
    with pytest.raises(NotFittedError):
        raise NotFittedError("Estimator is not fitted.")
    
    