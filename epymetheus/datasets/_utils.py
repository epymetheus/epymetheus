import pandas as pd


def fill_and_cut(series, begin_date=None, method="ffill"):
    """
    Examples
    --------
    >>> index = pd.DatetimeIndex([
    ...     "1999-12-30",
    ...     "2000-01-02",
    ...     "2000-01-03",
    ...     "2000-01-05",
    ... ])
    >>> s = pd.Series([0,1,2,3], index=index)
    >>> s
    1999-12-30    0
    2000-01-02    1
    2000-01-03    2
    2000-01-05    3
    dtype: int64
    >>> fill_and_cut(s, begin_date='2000-01-01')
    2000-01-01    0
    2000-01-02    1
    2000-01-03    2
    2000-01-04    2
    2000-01-05    3
    Freq: D, dtype: int64
    """
    f_index = pd.date_range(series.index[0], series.index[-1])
    n_index = pd.date_range(begin_date or series.index[0], series.index[-1])
    return series.reindex(f_index, method=method).reindex(n_index)
