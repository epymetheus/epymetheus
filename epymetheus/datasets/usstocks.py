from os.path import dirname
from pathlib import Path

import pandas as pd
from pandas_datareader import DataReader

from ._utils import fill_and_cut

module_path = Path(dirname(__file__))


def fetch_usstocks(
    begin_date="2000-01-01",
    end_date="2019-12-31",
    n_assets=10,
    column="Adj Close",
    verbose=True,
    fill=True,
) -> pd.DataFrame:
    """
    Return `pandas.DataFrame` of historical prices of US stocks.

    Parameters
    ----------
    begin_date : str
    end_date : str
    n_assets : int

    Returns
    -------
    universe : pandas.DataFrame
        Historical prices of US stocks.
    """
    begin_date = pd.Timestamp(begin_date)
    end_date = pd.Timestamp(end_date)

    with open(module_path / "usstocks.txt") as f:
        tickers = [ticker.strip() for ticker in f.readlines()]

    if n_assets > len(tickers):
        raise ValueError("n_assets should be <=", len(tickers))

    prices_dict = {
        ticker: DataReader(
            name=ticker,
            data_source="yahoo",
            start=begin_date - pd.Timedelta(days=10),
            end=end_date,
        )[column]
        for ticker in tickers[:n_assets]
    }

    if fill:
        prices_dict = {
            k: fill_and_cut(price, begin_date=begin_date)
            for k, price in prices_dict.items()
        }

    return pd.DataFrame(prices_dict)
