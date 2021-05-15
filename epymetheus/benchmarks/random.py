import numpy as np

from epymetheus import Strategy
from epymetheus import trade


def random_uniform(min_value=0, max_value=1, size=1):
    """
    Return random floats in range [min_value, max_value].

    Examples
    --------
    >>> np.random.seed(42)
    >>> random_uniform(1, 2, 3)
    array([1.37454012, 1.95071431, 1.73199394])
    """
    return (max_value - min_value) * np.random.random(size) + min_value


class RandomStrategy(Strategy):
    """
    Randomly yield trades.

    Parameters
    ----------
    n_trades : int, default 10
        Number of trades to yield.
    max_n_assets : int, default 1
        Maximum number of assets in a single trade.
    max_lot : 1
        Maximum value of lots.
    min_lot : 1
        Minimum value of lots.

    Examples
    --------
    >>> np.random.seed(42)
    >>> from epymetheus.datasets import make_randomwalk
    >>> strategy = RandomStrategy(n_trades=2)
    >>> universe = make_randomwalk(10, 3)
    >>> strategy(universe)
    [trade(['2'], lot=[1.], entry=1, exit=8), trade(['1'], lot=[1.], entry=1, exit=4)]
    """

    def __init__(self, n_trades=10, max_n_assets=1, max_lot=1.0, min_lot=1.0):
        self._n_trades = n_trades
        self.max_n_assets = max_n_assets
        self.max_lot = max_lot
        self.min_lot = min_lot

    def logic(self, universe):
        for _ in range(self._n_trades):
            n_assets = np.random.randint(1, self.max_n_assets + 1, size=1)[0]
            asset = list(np.random.choice(universe.columns, n_assets))
            lot = list(random_uniform(self.min_lot, self.max_lot, size=n_assets))
            entry, exit = sorted(np.random.choice(universe.index, 2))

            yield (lot * trade(asset, entry=entry, exit=exit))
