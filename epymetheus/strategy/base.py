import abc
import json
from functools import partial
from time import time
from typing import TypeVar

import numpy as np
import pandas as pd

from .. import ts
from .._utils import print_if_verbose
from .._utils import to_json
from ..exceptions import NoTradeWarning
from ..exceptions import NotRunError
from ..metrics import metric_from_name
from ..trade import Trade
from ..trade import check_trade

T = TypeVar("T", bound="Strategy")


def create_strategy(fn, **params) -> T:
    """Create a :class:`Strategy` from a function.

    A function `fn` takes more than one parameters.
    The first parameter should be `universe`, which is a `pandas.DataFrame`
    of historical prices. That is, its indices represents timestamp, columns
    stands for the names of assets, and each element in the frame stores a price.
    The following parameters are user-defined ones parameterizing the strategy.
    These parameters should be useful when one want to try various configurations
    such as trade volume, profit-taking threshold, loss-cut threshold, and so forth.
    The function `fn` is supposed to yield arbitrary numbers of :class:`Trade`
    depending on the `universe` and other parameters. These trades will be executed
    for the `universe` and evaluated accordingly.

    Args:
        fn (callable): A function that returns iterable of :class:`Trade`
            from a `universe` and parameters.
        **params: names and values of the parameters.

    Returns:
        :class:`Strategy`

    Examples:

        The following strategy trades the first asset in a given `universe`.
        The parameter `my_param` controls the volume to trade.

        >>> import epymetheus as ep
        >>>
        >>> def fn(universe, my_param):
        ...     asset = universe.columns[0]
        ...     yield my_param * ep.trade(asset)
        >>>
        >>> strategy = ep.create_strategy(fn, my_param=2.0)
        >>> universe = pd.DataFrame({"AAPL": [100, 101], "AMZN": [200, 201]})
        >>> strategy(universe)
        [trade(['AAPL'], lot=[2.])]
    """
    return Strategy._create_strategy(fn, **params)


class Strategy(abc.ABC):
    """Base class of trading strategy.

    There are two ways to create a :class:`Strategy`:

    - Use :func:`create_strategy`: This should be easier for simple strategies.
      See :func:`create_strategy`.
    - Subclass :class:`Strategy`: See below.

    One can create a strategy by subclassing :class:`Strategy` and
    overriding a method `logic`.
    The method `logic` takes arbitrary numbers of user-defined parameters
    parameterizing the strategy.
    These parameters should be useful when one want to try various configurations
    such as trade volume, profit-taking threshold, loss-cut threshold, and so forth.
    The method `logic` is supposed to yield arbitrary numbers of :class:`Trade`
    depending on the `universe` and other parameters. These trades will be executed
    for the `universe` and evaluated accordingly.


    Examples:

        The following strategy trades the first asset in a given `universe`.
        The parameter `my_param` controls the volume to trade.

        >>> import pandas as pd
        >>> import epymetheus as ep
        >>>
        >>> class MyStrategy(ep.Strategy):
        ...     def __init__(self, my_param):
        ...         self.my_param = my_param
        ...
        ...     def logic(self, universe: pd.DataFrame):
        ...         asset = universe.columns[0]
        ...         yield self.my_param * ep.trade(asset)
        ...
        >>> strategy = MyStrategy(my_param=2.0)
        >>> universe = pd.DataFrame({"AAPL": [100, 101], "AMZN": [200, 201]})
        >>> strategy(universe)
        [trade(['AAPL'], lot=[2.])]

        The method :func:`run` runs the strategy on a given universe.

        >>> strategy = MyStrategy(my_param=2.0).run(universe, verbose=False)
        >>> strategy.trades
        [trade(['AAPL'], lot=[2.])]
    """

    @classmethod
    def _create_strategy(cls, fn, **params) -> T:
        self = cls()
        self._fn = fn
        self._params = params
        return self

    def __call__(self, universe, to_list=True):
        if hasattr(self, "_fn"):
            setattr(self, "logic", partial(self._fn, **self.get_params()))
        trades = self.logic(universe)
        trades = list(trades) if to_list else trades
        return trades

    def logic(self, universe):
        """Logic to generate trades from universe.

        Override this to implement trading strategy by subclassing `Strategy`.

        Args:
            universe (pandas.DataFrame): Historical price data to apply this strategy.
                The index represents timestamps and the column is the assets.
            **params: Parameter values.

        Returns:
            iterable[Trade]
        """

    def run(self: T, universe, verbose=True, check_trades=False) -> T:
        """Run a backtesting of strategy.

        Args:
            universe (pandas.DataFrame): Historical price data to apply this strategy.
                The index represents timestamps and the column is the assets.
            verbose (bool, default=True): Verbose mode.
            check_trade (bool, default=False):
                If `True`, check that `asset`, `entry`, `exit` of trade

        Returns:
            self
        """
        _begin_time = time()

        self.universe = universe

        # Yield trades
        _begin_time_yield = time()
        trades = []
        for i, t in enumerate(self(universe, to_list=False) or []):
            print_if_verbose(
                f"\r{i + 1} trades returned: {t} ... ", end="", verbose=verbose
            )
            if check_trades:
                check_trade(t, universe)
            trades.append(t)
        if len(trades) == 0:
            raise NoTradeWarning("No trade was returned.")
        _time = time() - _begin_time_yield
        print_if_verbose(f"Done. (Runtume: {_time:.4f} sec)", verbose=verbose)

        # Execute trades
        _begin_time_execute = time()
        for i, t in enumerate(trades):
            print_if_verbose(
                f"\r{i + 1} trades executed: {t} ... ", end="", verbose=verbose
            )
            t.execute(universe)
        _time = time() - _begin_time_execute
        print_if_verbose(f"Done. (Runtime: {_time:.4f} sec)", verbose=verbose)

        self.trades = trades

        _time = time() - _begin_time
        final_wealth = self.score("final_wealth")
        print_if_verbose(
            f"Done. Final wealth: {final_wealth:.2f} (Runtime: {_time:.4f} sec)",
            verbose=verbose,
        )

        return self

    def score(self, metric_name) -> float:
        """Returns the value of a metric of self.

        Args:
            metric_name (str): Metric to evaluate.

        Returns:
            float
        """
        if not hasattr(self, "trades"):
            raise NotRunError("Strategy has not been run")

        return metric_from_name(metric_name)(self.trades, self.universe)

    def history(self) -> pd.DataFrame:
        """Return `pandas.DataFrame` of trade history.

        Returns:
            pandas.DataFrame
        """
        if not hasattr(self, "trades"):
            raise NotRunError("Strategy has not been run")

        data = {}

        n_orders = np.array([t.asset.size for t in self.trades])

        data["trade_id"] = np.repeat(np.arange(len(self.trades)), n_orders)
        data["asset"] = np.concatenate([t.asset for t in self.trades])
        data["lot"] = np.concatenate([t.lot for t in self.trades])
        data["entry"] = np.repeat([t.entry for t in self.trades], n_orders)
        data["close"] = np.repeat([t.close for t in self.trades], n_orders)
        data["exit"] = np.repeat([t.exit for t in self.trades], n_orders)
        data["take"] = np.repeat([t.take for t in self.trades], n_orders)
        data["stop"] = np.repeat([t.stop for t in self.trades], n_orders)
        data["pnl"] = np.concatenate([t.final_pnl(self.universe) for t in self.trades])

        return pd.DataFrame(data)

    def trades_to_dict(self) -> list:
        """Represents and returns `trades` as `dict` objects.

        Returns:
            list[dict]

        Examples:

            >>> import epymetheus as ep
            >>>
            >>> strategy = ep.create_strategy(
            ...     lambda universe: [ep.trade("AAPL")]
            ... ).run(pd.DataFrame({"AAPL": [100, 101]}), verbose=False)
            >>> strategy.trades_to_dict()
            [{'asset': ['AAPL'], 'lot': [1.0], 'close': 1}]
        """
        return [trade.to_dict() for trade in self.trades]

    def trades_to_json(self):
        """Represents and returns `trades` as a string in JSON format.

        Returns:
            str

        Examples:

            >>> import epymetheus as ep
            >>>
            >>> strategy = ep.create_strategy(
            ...     lambda universe: [ep.trade("AAPL")]
            ... ).run(pd.DataFrame({"AAPL": [100, 101]}), verbose=False)
            >>> strategy.trades_to_json()
            '[{"asset": ["AAPL"], "lot": [1.0], "close": 1}]'

            >>> s = '[{"asset": ["AAPL"], "lot": [1.0], "close": 1}]'
            >>> strategy = Strategy()
            >>> strategy.universe = pd.DataFrame({"AAPL": [100, 101]})
            >>> strategy.load_trades_json(s).trades
            [trade(['AAPL'], lot=[1.])]
        """
        return to_json(self.trades_to_dict())

    def load(self, history: pd.DataFrame, universe: pd.DataFrame):
        """Load trade history and universe.

        Args:
            history (pandas.DataFrame): History to load.
            universe (pandas.DataFrame): Universe to load.

        Returns:
            self
        """
        return self.load_universe(universe).load_history(history)

    def load_history(self, history: pd.DataFrame):
        self.trades = Trade.load_history(history)
        for trade in self.trades:
            # Assuming that self has loaded universe
            trade.execute(self.universe)
        return self

    def load_trades_dict(self, l: list) -> "Strategy":
        """

        Args:
            l (list[dict]):
        """
        self.trades = [Trade.load_dict(d) for d in l]
        for trade in self.trades:
            # Assuming that self has loaded universe
            trade.execute(self.universe)
        return self

    def load_trades_json(self, s: str) -> "Strategy":
        """

        Args:
            s (str):
        """
        # try:
        trades_as_dict = json.loads(s)
        # except json.JSONDecodeError:
        #     # If s cannot be interpreted as a json string,
        #     # try to interpret it as a file name of json
        #     trades_as_dict = json.load(s)

        self.load_trades_dict(trades_as_dict)

        for trade in self.trades:
            # Assuming that self has loaded universe
            trade.execute(self.universe)

        return self

    def load_universe(self, universe: pd.DataFrame):
        """Load universe.

        Args:
            universe (pandas.DataFrame): Universe to load.

        Returns:
            self
        """
        self.universe = universe
        return self

    def wealth(self) -> pd.Series:
        """Return `pandas.Series` of wealth.

        Returns:
            pandas.Series
        """
        if not hasattr(self, "trades"):
            raise NotRunError("Strategy has not been run")

        return pd.Series(
            ts.wealth(self.trades, self.universe), index=self.universe.index
        )

    def drawdown(self) -> pd.Series:
        """

        Returns:
            pandas.Series
        """
        if not hasattr(self, "trades"):
            raise NotRunError("Strategy has not been run")

        drawdown = ts.drawdown(self.trades, self.universe)

        return pd.Series(drawdown, index=self.universe.index)

    def exposure(self) -> pd.DataFrame:
        """Return exposure of self to each asset.

        Returns:
            pandas.DataFrame: DataFrame of exposure.

        Examples:

            >>> import pandas as pd
            >>> import epymetheus as ep
            ...
            >>> universe = pd.DataFrame({
            ...     "A0": [1, 2, 3, 4, 5],
            ...     "A1": [2, 3, 4, 5, 6],
            ...     "A2": [3, 4, 5, 6, 7],
            ... })
            >>> strategy = ep.create_strategy(lambda universe: [
            ...     [1, -1] * ep.trade(["A0", "A2"], entry=1, exit=3),
            ...     [-1, 2] * ep.trade(["A1", "A2"], entry=2, exit=4),
            ... ]).run(universe, verbose=False)
            >>> strategy.exposure()
                A0   A1    A2
            0  0.0  0.0   0.0
            1  0.0  0.0   0.0
            2  3.0  0.0  -5.0
            3  4.0 -5.0   6.0
            4  0.0 -6.0  14.0
        """
        if not hasattr(self, "trades"):
            raise NotRunError("Strategy has not been run")

        dict_exposure = {k: 0.0 for k in self.universe.columns}
        for d in map(lambda t: t.dict_exposure(self.universe), self.trades):
            for key in d:
                dict_exposure[key] += d[key]

        return pd.DataFrame(dict_exposure, index=self.universe.index)

    def net_exposure(self) -> pd.Series:
        """Return net exposure of the strategy.

        Returns:
            pandas.Series

        Examples:

            >>> import pandas as pd
            >>> import epymetheus as ep
            ...
            >>> universe = pd.DataFrame({
            ...     "A0": [1.0, 2.0, 3.0, 4.0, 5.0],
            ...     "A1": [2.0, 3.0, 4.0, 5.0, 6.0],
            ...     "A2": [3.0, 4.0, 5.0, 6.0, 7.0],
            ... }, dtype=float)
            >>> strategy = ep.create_strategy(lambda universe: [
            ...     [1, -1] * ep.trade(["A0", "A2"], entry=1, exit=3),
            ...     [-1, 2] * ep.trade(["A1", "A2"], entry=2, exit=4),
            ... ]).run(universe, verbose=False)
            >>> strategy.exposure()
                A0   A1    A2
            0  0.0  0.0   0.0
            1  0.0  0.0   0.0
            2  3.0  0.0  -5.0
            3  4.0 -5.0   6.0
            4  0.0 -6.0  14.0
            >>> strategy.net_exposure()
            0    0.0
            1    0.0
            2   -2.0
            3    5.0
            4    8.0
            dtype: float64
        """
        return self.exposure().sum(1)

    def abs_exposure(self) -> pd.Series:
        """Return absolute exposure of the strategy.

        Returns:
            pandas.Series

        Examples:

            >>> import pandas as pd
            >>> import epymetheus as ep
            ...
            >>> universe = pd.DataFrame({
            ...     "A0": [1.0, 2.0, 3.0, 4.0, 5.0],
            ...     "A1": [2.0, 3.0, 4.0, 5.0, 6.0],
            ...     "A2": [3.0, 4.0, 5.0, 6.0, 7.0],
            ... }, dtype=float)
            >>> strategy = ep.create_strategy(lambda universe: [
            ...     [1, -1] * ep.trade(["A0", "A2"], entry=1, exit=3),
            ...     [-1, 2] * ep.trade(["A1", "A2"], entry=2, exit=4),
            ... ]).run(universe, verbose=False)
            >>> strategy.exposure()
                A0   A1    A2
            0  0.0  0.0   0.0
            1  0.0  0.0   0.0
            2  3.0  0.0  -5.0
            3  4.0 -5.0   6.0
            4  0.0 -6.0  14.0
            >>> strategy.abs_exposure()
            0     0.0
            1     0.0
            2     8.0
            3    15.0
            4    20.0
            dtype: float64
        """
        return self.exposure().abs().sum(1)

    def get_params(self) -> dict:
        """Set the parameters of this strategy.

        Returns:
            dict[str, *]
        """
        return getattr(self, "_params", {})

    def set_params(self, **params):
        """Set the parameters of this strategy.

        Args:
            **params (dict): Strategy parameters.

        Returns:
            self
        """
        valid_keys = self.get_params().keys()

        for key, value in params.items():
            if key not in valid_keys:
                raise ValueError(f"Invalid parameter: {key}")
            else:
                self._params[key] = value

        return self

    def __repr__(self):
        """
        >>> def my_func(universe, param_1, param_2):
        ...     return ...

        >>> strategy = create_strategy(my_func, param_1=1.0, param_2=2.0)
        >>> repr(strategy)
        'strategy(my_func, param_1=1.0, param_2=2.0)'

        >>> class MyStrategy(Strategy):
        ...     pass

        >>> strategy = MyStrategy()
        >>> repr(strategy)
        'MyStrategy'
        """
        if hasattr(self, "_fn"):
            fname = self._fn.__name__
            param = ", ".join(f"{k}={v}" for k, v in self.get_params().items())
            param = f", {param}" if param != "" else ""
            return f"strategy({fname}{param})"
        else:
            return self.__class__.__name__
