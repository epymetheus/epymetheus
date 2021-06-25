import json
from copy import deepcopy

import numpy as np
import pandas as pd

from ._utils import to_json


def trade(asset, entry=None, exit=None, take=None, stop=None, lot=1.0, **kwargs):
    """Create a :class:`Trade`.

    Args:
        asset (str or array-like): Name of assets to trade.
            If an array-like (e.g. `list` or `numpy.ndarray`) of objects is given,
            a trade transacts multiple assets at once.
        entry (object, optional): Timestamp to open a trade.
            If not provided, the first timestamp in a given `universe`.
        exit (object, optional): Timestamp to exit a trade.
            If not provided, the last timestamp in a given `universe`.
        take (float > 0, optional): Threshold of profit-take.
        stop (float < 0, optional): Threshold of stop-loss.
        lot (float or array-like, default 1.0): The amount of trade.
            If an array-like (e.g. `list` or `numpy.ndarray`) of objects is given,
            each element in an array represents the amount of each asset.

    Returns:
        :class:`Trade`

    Examples:

        Buy a single unit of "AAPL".

        >>> trade("AAPL")
        trade(['AAPL'], lot=[1.])

        Buy single units of "AAPL" and "AMZN".

        >>> trade(["AAPL", "AMZN"])
        trade(['AAPL' 'AMZN'], lot=[1. 1.])

        Buy a single unit of "AAPL" and sell two units of "AMZN".

        >>> [1.0, -2.0] * trade(["AAPL", "AMZN"])
        trade(['AAPL' 'AMZN'], lot=[ 1. -2.])

        Entry a trade from 2020-01-01.

        >>> from datetime import date
        >>> trade("AAPL", date(2020, 1, 1))
        trade(['AAPL'], lot=[1.], entry=2020-01-01)

        Entry a trade from 2020-01-01 and exit at 2020-01-31.

        >>> trade("AAPL", date(2020, 1, 1), date(2020, 1, 31))
        trade(['AAPL'], lot=[1.], entry=2020-01-01, exit=2020-01-31)

        Configure profit-take and loss-cut (in unit of dollar).

        >>> trade("AAPL", take=200.0, stop=-100.0)
        trade(['AAPL'], lot=[1.], take=200.0, stop=-100.0)
    """
    return Trade(asset, entry=entry, exit=exit, take=take, stop=stop, lot=lot, **kwargs)


class Trade:
    """A `epymetheus.Trade` represents a single trade.

    Args:
        asset (str or array-like): Name of assets to trade.
            If an array-like (e.g. `list` or `numpy.ndarray`) of objects is given,
            a trade transacts multiple assets at once.
        entry (object, optional): Timestamp to open a trade.
            If not provided, the first timestamp in a given `universe`.
        exit (object, optional): Timestamp to exit a trade.
            If not provided, the last timestamp in a given `universe`.
        take (float > 0, optional): Threshold of profit-take.
        stop (float < 0, optional): Threshold of stop-loss.
        lot (float or array-like, default 1.0): The amount of trade.
            If an array-like (e.g. `list` or `numpy.ndarray`) of objects is given,
            each element in an array represents the amount of each asset.

    Attributes:
        close (object): Datetime to close the trade.
            It is set by the method `self.execute`.
    """

    def __init__(self, asset, entry=None, exit=None, take=None, stop=None, lot=1.0):
        # Convert to np.array
        asset = np.asarray(asset).reshape(-1)
        lot = np.broadcast_to(np.asarray(lot), asset.shape)

        self.asset = asset
        self.entry = entry
        self.exit = exit
        self.take = take
        self.stop = stop
        self.lot = lot

    def execute(self, universe):
        """Execute trade and set `self.close`.

        Args:
            universe : pandas.DataFrame

        Returns:
            self

        Examples:

            >>> import pandas as pd
            >>> import epymetheus as ep
            >>> universe = pd.DataFrame({
            ...     "A0": [1., 2., 3., 4., 5., 6., 7.],
            ...     "A1": [2., 3., 4., 5., 6., 7., 8.],
            ...     "A2": [3., 4., 5., 6., 7., 8., 9.],
            ... }, dtype=float)

            >>> t = ep.trade("A0", entry=1, exit=6)
            >>> t = t.execute(universe)
            >>> t.close
            6

            >>> t = ep.trade("A0", entry=1, exit=6, take=2)
            >>> t = t.execute(universe)
            >>> t.close
            3

            >>> t = -ep.trade(asset="A0", entry=1, exit=6, stop=-2)
            >>> t = t.execute(universe)
            >>> t.close
            3
        """
        entry = universe.index[0] if self.entry is None else self.entry
        exit = universe.index[-1] if self.exit is None else self.exit

        if (universe.columns.get_indexer(self.asset) == -1).any():
            raise KeyError(f"asset {self.asset} not in universe.columns")
        if entry not in universe.index:
            raise KeyError(f"entry {self.entry} not in universe.index")
        if exit not in universe.index:
            raise KeyError(f"exit {self.exit} not in universe.index")

        close = exit

        if (self.take is not None) or (self.stop is not None):
            i_entry = universe.index.get_indexer([entry]).item()
            i_exit = universe.index.get_indexer([exit]).item()

            # Compute pnl
            value = self.array_value(universe).sum(axis=1)
            pnl = value - value[i_entry]
            pnl[:i_entry] = 0

            # Place profit-take or loss-cut order
            take = self.take if self.take is not None else np.inf
            stop = self.stop if self.stop is not None else -np.inf
            signal = np.logical_or(pnl >= take, pnl <= stop)

            # Compute close
            i_order = np.searchsorted(signal, True)
            i_close = min(i_exit, i_order)
            close = universe.index[i_close]

        self.close = close

        return self

    def array_value(self, universe):
        """Return value of self for each asset.

        Args:
            array_value (numpy.ndarray): Array of values.
                shape (n_bars, n_orders)

        Examples:

            >>> import pandas as pd
            >>> import epymetheus as ep
            ...
            >>> universe = pd.DataFrame({
            ...     "A0": [1, 2, 3, 4, 5],
            ...     "A1": [2, 3, 4, 5, 6],
            ...     "A2": [3, 4, 5, 6, 7],
            ... })
            >>> trade = [2, -3] * ep.trade(["A0", "A2"], entry=1, exit=3)
            >>> trade.array_value(universe)
            array([[  2.,  -9.],
                   [  4., -12.],
                   [  6., -15.],
                   [  8., -18.],
                   [ 10., -21.]])
        """
        array_value = self.lot * universe.loc[:, self.asset].values
        return array_value

    def final_pnl(self, universe):
        """Return final profit-loss of self.

        Returns:
            numpy.ndarray, shape (n_orders, )

        Examples:

            >>> import pandas as pd
            >>> import epymetheus as ep
            ...
            >>> universe = pd.DataFrame({
            ...     "A0": [1, 2, 3, 4, 5],
            ...     "A1": [2, 3, 4, 5, 6],
            ...     "A2": [3, 4, 5, 6, 7],
            ... }, dtype=float)
            >>> t = ep.trade(["A0", "A2"], entry=1, exit=3)
            >>> t = t.execute(universe)
            >>> t.final_pnl(universe)
            array([2., 2.])
        """
        i_entry = universe.index.get_indexer([self.entry]).item()
        i_close = universe.index.get_indexer([self.close]).item()

        value = self.array_value(universe)
        pnl = value - value[i_entry]
        pnl[:i_entry] = 0
        pnl[i_close:] = pnl[i_close]

        final_pnl = pnl[-1]

        return final_pnl

    @classmethod
    def load_history(cls, history: pd.DataFrame):
        """Load trades from a DataFrame of history.

        Args:
            pandas.DataFrame

        Returns:
            list[Trade]
        """
        trades = []
        for trade_id in history.trade_id.unique().tolist():
            h = history[history.trade_id == trade_id]
            kwargs = {}
            kwargs["asset"] = h.asset.values
            kwargs["lot"] = h.lot.values
            kwargs["entry"] = h.entry.values[0]
            kwargs["exit"] = h.exit.values[0]
            kwargs["take"] = h.loc[:, "take"].values[0]  # pd.Series has `take` method
            kwargs["stop"] = h.stop.values[0]
            trades.append(cls(**kwargs))
        return trades

    def to_dict(self) -> dict:
        """Represents and returns `self` as `dict` object.

        Returns:
            dict

        Examples:

            >>> import epymetheus as ep

            >>> trade = ep.trade("A0", entry=1, exit=6, take=2)
            >>> trade.to_dict()
            {'asset': ['A0'], 'lot': [1.0], 'entry': 1, 'exit': 6, 'take': 2}
        """
        trade_as_dict = dict(asset=self.asset.tolist(), lot=self.lot.tolist())
        for attr in ("entry", "exit", "take", "stop", "close"):
            if getattr(self, attr, None) is not None:
                trade_as_dict[attr] = getattr(self, attr)

        if hasattr(self, "close"):
            trade_as_dict["close"] = self.close

        return trade_as_dict

    def to_json(self) -> str:
        """Represents and returns `self` as a string in JSON format.

        Returns:
            str

        Examples:

            >>> import epymetheus as ep

            >>> trade = ep.trade("A0", entry=1, exit=6, take=2)
            >>> trade.to_json()
            '{"asset": ["A0"], "lot": [1.0], "entry": 1, "exit": 6, "take": 2}'

            >>> s = trade.to_json()
            >>> ep.Trade.load_json(s)
            trade(['A0'], lot=[1.], entry=1, exit=6, take=2)
        """
        return to_json(self.to_dict())

    @classmethod
    def load_json(cls, s: str) -> "Trade":
        """Loads JSON and creates a `Trade`.

        Args:
            s (str): JSON string

        Returns:
            Trade
        """
        return cls.load_dict(json.loads(s))

    @classmethod
    def load_dict(cls, d: dict) -> "Trade":
        """Loads `dict` object and creates a `Trade`.

        Args:
            d (dict):

        Returns:
            Trade
        """
        close = None
        if "close" in d:
            # __init__ does not have a parameter "close"
            close = d["close"]
            del d["close"]
        trade = cls(**d)
        if close is not None:
            trade.close = close
        return trade

    def __eq__(self, other):
        def eq(t0, t1, attr):
            attr0 = getattr(t0, attr)
            attr1 = getattr(t1, attr)
            if isinstance(attr0, np.ndarray) and isinstance(attr1, np.ndarray):
                return np.array_equal(attr0, attr1)
            else:
                return attr0 == attr1

        # close does not have to be tested:
        # if the following attributes are identical, close will be the same too
        attrs = ("asset", "entry", "exit", "take", "stop", "lot")
        return all(eq(self, other, attr) for attr in attrs)

    def __mul__(self, num):
        return self.__rmul__(num)

    def __rmul__(self, num):
        """
        Multiply lot of self.

        Examples
        --------
        >>> trade("AMZN")
        trade(['AMZN'], lot=[1.])
        >>> (-2.0) * trade("AMZN")
        trade(['AMZN'], lot=[-2.])

        >>> trade(["AMZN", "AAPL"])
        trade(['AMZN' 'AAPL'], lot=[1. 1.])
        >>> (-2.0) * trade(["AMZN", "AAPL"])
        trade(['AMZN' 'AAPL'], lot=[-2. -2.])
        >>> [2.0, 3.0] * trade(["AMZN", "AAPL"])
        trade(['AMZN' 'AAPL'], lot=[2. 3.])
        """
        t = deepcopy(self)
        t.lot = t.lot * np.asarray(num)
        return t

    def __neg__(self):
        """
        Invert the lot of self.

        Examples
        --------
        >>> -trade("AMZN")
        trade(['AMZN'], lot=[-1.])
        """
        return (-1.0) * self

    def __truediv__(self, num):
        """
        Divide the lot of self.

        Examples
        --------
        >>> trade("AMZN", lot=2.0) / 2.0
        trade(['AMZN'], lot=[1.])

        >>> trade(["AMZN", "AAPL"], lot=[2.0, 4.0]) / 2.0
        trade(['AMZN' 'AAPL'], lot=[1. 2.])
        """
        return (1.0 / num) * self

    def __repr__(self):
        """
        >>> t = trade("AMZN", entry=1)
        >>> t
        trade(['AMZN'], lot=[1.], entry=1)

        >>> t = trade("AMZN", take=100.0)
        >>> t
        trade(['AMZN'], lot=[1.], take=100.0)
        """
        params = [f"{self.asset}", f"lot={self.lot}"]

        for attr in ("entry", "exit", "take", "stop"):
            value = getattr(self, attr)
            if value is not None:
                params.append(f"{attr}={value}")

        return f"trade({', '.join(params)})"


def check_trade(
    trade: Trade,
    universe: pd.DataFrame,
    check_asset: bool = True,
    check_index: bool = True,
    check_lot: bool = True,
    check_take: bool = True,
    check_stop: bool = True,
) -> None:
    """Validation for `Trade`.

    Args:
        trade (Trade): Trade object to validate.
        universe (pandas.DataFrame): Universe (price data) to apply `trade`.
        check_asset (bool, default=True): If `True`, ...
        check_index (bool, default=True):
        check_lot (bool, default=True):
        check_take (bool, default=True):
        check_stop (bool, default=True):

    Returns:
        None

    Raises:
        ValueError: If something is wrong

    Examples:

        >>> import epymetheus as ep
        >>> from epymetheus.trade import check_trade
        >>>
        >>> universe = pd.DataFrame({"A": [100, 101, 102]}, index=[0, 1, 2])
        >>> trade = ep.trade("A", entry=1)
        >>> check_trade(trade, universe)  # OK
    """
    if check_asset:
        for a in trade.asset:
            if a not in universe.columns:
                raise ValueError("asset is not found in index:", a)

    if check_index:
        if getattr(trade, "entry", None) is not None:
            if trade.entry not in universe.index:
                raise ValueError("entry is not found in index:", trade.entry)
        if getattr(trade, "exit", None) is not None:
            if trade.exit not in universe.index:
                raise ValueError("exit is not found in index:", trade.exit)

    if check_lot:
        if not np.isfinite(trade.lot).all():
            raise ValueError("lot is not finite:", trade.lot)

    if check_take:
        if getattr(trade, "take", None) is not None:
            if trade.take < 0:
                raise ValueError("take should be nonnegative, got:", trade.take)

    if check_stop:
        if getattr(trade, "stop", None) is not None:
            if trade.stop > 0:
                raise ValueError("stop should be nonpositive, got:", trade.stop)
