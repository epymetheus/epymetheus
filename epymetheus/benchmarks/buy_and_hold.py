import numpy as np

from epymetheus import Strategy
from epymetheus import trade


class BuyAndHold(Strategy):
    """
    Buy-and-Hold Strategy.

    Parameters
    ----------
    weight : dict[str, float]
        Keys are assets to trade.
        Values are value-based weights.

    Examples
    --------
    >>> from epymetheus.datasets import make_randomwalk
    >>> strategy = BuyAndHold({"0": 0.3, "1": 0.7})
    >>> universe = make_randomwalk(5, 3)
    >>> strategy(universe)
    [trade(['0' '1'], lot=[0.3 0.7], entry=0)]
    """

    def __init__(self, weight):
        self.weight = weight

    def logic(self, universe):
        asset = list(self.weight.keys())
        price = universe.loc[:, asset].iloc[0].values
        lot = list(np.array(list(self.weight.values())) / price)
        yield lot * trade(asset, entry=universe.index[0])
