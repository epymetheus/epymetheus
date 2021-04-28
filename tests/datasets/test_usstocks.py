import numpy as np
import pytest  # noqa: F401
from pandas_datareader._utils import RemoteDataError

from epymetheus.datasets import fetch_usstocks

# --------------------------------------------------------------------------------


def test_toomanyasset():
    """
    Test if fetch_usstocks raises ValueError
    when n_asset is too many.
    """
    with pytest.raises(ValueError):
        fetch_usstocks(n_assets=1000)


def test_usstocks():
    try:
        universe = fetch_usstocks(n_assets=2)
        assert not np.isnan(universe.values).any(axis=None)
    except RemoteDataError as e:
        print("Skip", e)
