from epymetheus import Strategy


class DeterminedStrategy(Strategy):
    """
    Yield given trades.

    Parameters
    ----------
    trade : iterable of Trade
        Trades to yield.

    Examples
    --------
    >>> import epymetheus as ep
    >>> strategy = DeterminedStrategy([ep.trade("A"), ep.trade("B")])
    >>> universe = ...
    >>> strategy(universe)
    [trade(['A'], lot=[1.]), trade(['B'], lot=[1.])]
    """

    def __init__(self, trades):
        self._trades = trades

    def logic(self, universe):
        for t in self._trades:
            yield t
