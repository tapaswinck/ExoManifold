import numpy as np
import pytest

from exomanifold.utils.validation import (
    check_array, check_X_y
    )

def test_check_array_accepts_valid_array():
    X = np.array([[1,2],[3,4]])

    result = check_array(X)

    assert np.array_equal(result, X)



def test_check_array_rejects_none():
    with pytest.raises(ValueError):
        check_array(None)

def test_check_array_rejects_one_dimensional_array():
    X = np.array([1,2,3])

    with pytest.raises(ValueError):
        check_array(X)

def test_check_array_rejects_empty_array():
    X = np.empty((0,2))

    with pytest.raises(ValueError):
        check_array(X)


    

def test_check_X_y_accepts_valid_input():
    X = np.array([[1,2],[3,4],[5,6]])
    y = np.array([0,1,0])

    X_checked, y_checked = check_X_y(X,y)
    
    assert np.array_equal(X_checked, X)
    assert np.array_equal(y_checked, y)


def test_check_X_y_rejects_different_lenghts():
    X = np.array([[1,2],[3,4]])
    y = np.array([0])

    with pytest.raises(ValueError):
        check_X_y(X,y)


def test_check_X_y_rejects_two_dimensional_y():
    X = np.array([[1, 2],[3,4]])
    y = np.array([[0], [1]])

    with pytest.raises(ValueError):
        check_X_y(X,y)

def test_check_X_y_rejects_none_y():
    X = np.array([[1,2], [3,4]])

    with pytest.raises(ValueError):
        check_X_y(X,None)





