"""
Iput/Output utilities for ExoManifold.
"""


from __future__ import annotations


__all__ = [
    "save_json",
    "load_json",
    "save_pickle",
    "load_pickle",
    "save_numpy",
    "load_numpy",
]

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

def save_json(
        data: dict[str, Any],
        filename: Path
)-> None:
    """
    Save a dictionary to a JSON file.
    """
    with filename.open("w", encoding = "utf-8") as file:
        json.dump(data, file, indent=4)


def load_json(
    filename: Path,
) -> dict[str, Any]:
    """
    Load a dictionary from a JSON file.
    """

    with filename.open("r", encoding="utf-8") as file:
        data = json.load(file)
        return data
    

def save_pickle(
        obj: Any,
        filename: Path
)-> None:
    """
    Save a Python object using pickle.
    """

    with filename.open("wb") as file:
        pickle.dump(obj, file)


def load_pickle(
        filename: Path,
)-> Any:
    """
    Load a pickle object.
    """
    with filename.open("rb") as file:
        return pickle.load(file)
    

def save_numpy(
        array: NDArray[Any],
        filename: Path
)-> None:
    """
    Save a NumPy array.
    """

    np.save(filename, array)

def load_numpy(
        filename: Path
)-> NDArray[Any]:
    """
    Load a NumPy array.
    """

    return np.load(filename)

