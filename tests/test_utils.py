import numpy as np

from epymetheus.utils import train_test_roll


def test_train_test_roll():
    # TODO(simaki): test for list
    # TODO(simaki): test for pandas dataframe

    array = np.arange(500)
    roll = train_test_roll(array, 200, 100)

    assert len(roll) == 3
    assert np.array_equal(roll[0][0], np.arange(200))
    assert np.array_equal(roll[0][1], np.arange(200, 300))
    assert np.array_equal(roll[1][0], np.arange(100, 300))
    assert np.array_equal(roll[1][1], np.arange(300, 400))
    assert np.array_equal(roll[2][0], np.arange(200, 400))
    assert np.array_equal(roll[2][1], np.arange(400, 500))
