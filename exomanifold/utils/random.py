"""
Utilities for reproducible random number generation.
"""
from __future__ import annotations

import os
import random
import numpy as np

def set_random_seed(seed: int = 42)-> None:
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed: int, default=42
        seed value
    """

    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    