import pandas as pd
import pytest

from epymetheus.universe import Universe


def test_universe_deprecated():
    with pytest.raises(DeprecationWarning):
        u = Universe(pd.DataFrame({"A": [1]}))
