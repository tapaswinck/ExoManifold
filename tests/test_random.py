import random
import numpy as np

from exomanifold.utils.random import seed_everything
def test_python_random_seed():
    seed_everything(42)

    x = random.random()

    seed_everything(42)
    
    y = random.random()

    assert y == x

    def test_numpy_random_seed():
        seed_everything(42)

        x = np.random.rand(5)

        seed_everything(42)

        y = np.random.rand(5)

        assert np.array_equal(x, y)


    def test_different_seeds_produce_different_results():
        seed_everything(42)
        first = random.random()

        seed_everything(123)
        second = random.random()

        assert first != second

