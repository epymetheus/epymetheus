import pandas as pd
import pytest  # noqa: F401

from epymetheus.datasets._utils import fill_and_cut


def test_fill_and_cut():
    from pandas.testing import assert_series_equal

    series = pd.Series(
        {
            pd.Timestamp("1999-12-30"): 0,
            pd.Timestamp("2000-01-02"): 1,
            pd.Timestamp("2000-01-03"): 2,
            pd.Timestamp("2000-01-05"): 3,
        }
    )
    series_expected = pd.Series(
        [0, 1, 2, 2, 3], index=pd.date_range("2000-01-01", "2000-01-05")
    )

    assert_series_equal(fill_and_cut(series, "2000-01-01"), series_expected)
