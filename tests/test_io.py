import numpy as np
import pytest

from exomanifold.utils.io import *



# ---------------------------------------------------------
# JSON
# ---------------------------------------------------------

def test_save_and_load_json(tmp_path):

    data = {
        "mission": "Kepler",
        "neighbors": 15,
        "metric": "euclidean",
    }

    filename = tmp_path / "config.json"

    save_json(data, filename)

    loaded = load_json(filename)

    assert loaded == data


# ---------------------------------------------------------
# Pickle
# ---------------------------------------------------------

def test_save_and_load_pickle(tmp_path):

    data = {
        "planet": "Kepler-10b",
        "period": 0.837,
        "confirmed": True,
    }

    filename = tmp_path / "planet.pkl"

    save_pickle(data, filename)

    loaded = load_pickle(filename)

    assert loaded == data


# ---------------------------------------------------------
# NumPy
# ---------------------------------------------------------

def test_save_and_load_numpy(tmp_path):

    array = np.random.rand(20, 5)

    filename = tmp_path / "array.npy"

    save_numpy(array, filename)

    loaded = load_numpy(filename)

    assert np.array_equal(array, loaded)


# ---------------------------------------------------------
# Error handling
# ---------------------------------------------------------

def test_load_json_missing_file(tmp_path):

    filename = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        load_json(filename)


def test_load_pickle_missing_file(tmp_path):

    filename = tmp_path / "missing.pkl"

    with pytest.raises(FileNotFoundError):
        load_pickle(filename)


def test_load_numpy_missing_file(tmp_path):

    filename = tmp_path / "missing.npy"

    with pytest.raises(FileNotFoundError):
        load_numpy(filename)