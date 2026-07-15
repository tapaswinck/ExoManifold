import random
import numpy as np

from exomanifold.utils.random import set_random_seed

def test_python_random_seed():
    set_random_seed(42)

    x = random.random()

    set_random_seed(42)
    
    y = random.random()

    assert y == x

    def test_numpy_random_seed():
        set_random_seed(42)

        x = np.random.rand(5)

        set_random_seed(42)

        y = np.random.rand(5)

        assert np.array_equal(x, y)

        