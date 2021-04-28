import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal

from epymetheus import Strategy
from epymetheus import create_strategy
from epymetheus import trade
from epymetheus import ts
from epymetheus.benchmarks import DeterminedStrategy
from epymetheus.benchmarks import RandomStrategy
from epymetheus.datasets import make_randomwalk
from epymetheus.exceptions import NoTradeError
from epymetheus.exceptions import NotRunError
from epymetheus.metrics import avg_lose
from epymetheus.metrics import avg_pnl
from epymetheus.metrics import avg_win
from epymetheus.metrics import final_wealth
from epymetheus.metrics import num_lose
from epymetheus.metrics import num_win
from epymetheus.metrics import rate_lose
from epymetheus.metrics import rate_win

metrics = [
    avg_lose,
    avg_pnl,
    avg_win,
    final_wealth,
    num_lose,
    num_win,
    rate_lose,
    rate_win,
]


def my_func(universe, param_1, param_2):
    return


class MyStrategy(Strategy):
    def __init__(self, param_1, param_2):
        self.param_1 = param_1
        self.param_2 = param_2

    def logic(self, universe):
        yield (self.param_1 * trade("A"))
        yield (self.param_2 * trade("B"))


class TestStrategy:
    """
    Test `Strategy`
    """

    universe = pd.DataFrame({"A": range(10), "B": range(10)})

    @staticmethod
    def my_strategy(universe, param_1, param_2):
        """
        Example logic
        """
        yield (param_1 * trade("A"))
        yield (param_2 * trade("B"))

    # Initializing

    def test_init_from_func(self):
        strategy = create_strategy(self.my_strategy, param_1=1.0, param_2=2.0)
        universe = self.universe

        assert strategy(universe) == [1.0 * trade("A"), 2.0 * trade("B")]

    def test_init_from_init(self):
        strategy = MyStrategy(param_1=1.0, param_2=2.0)
        universe = self.universe

        assert strategy(universe) == [1.0 * trade("A"), 2.0 * trade("B")]

    @pytest.mark.parametrize("verbose", [True, False])
    def test_run_trades(self, verbose):
        """Test if trades are the same with call"""
        strategy = create_strategy(self.my_strategy, param_1=1.0, param_2=2.0)
        universe = self.universe

        result = strategy.run(universe, verbose=verbose).trades
        expected = strategy(universe)

        assert result == expected

    @pytest.mark.parametrize("verbose", [True, False])
    def test_run_notradeerror(self, verbose):
        strategy = create_strategy(lambda universe: [])
        universe = self.universe
        with pytest.raises(NoTradeError):
            strategy.run(universe, verbose=verbose)

    def test_run_execution(self):
        strategy = RandomStrategy()
        universe = make_randomwalk()

        np.random.seed(42)
        trades = strategy.run(universe)
        result = [t.close for t in strategy.trades]

        np.random.seed(42)
        expected = [t.execute(universe).close for t in strategy(universe)]

        assert result == expected

    def test_get_params(self):
        strategy = create_strategy(self.my_strategy, param_1=1.0, param_2=2.0)
        assert strategy.get_params() == {"param_1": 1.0, "param_2": 2.0}

        strategy = MyStrategy(param_1=1.0, param_2=2.0)
        assert strategy.get_params() == {}

    def test_set_params(self):
        strategy = create_strategy(self.my_strategy, param_1=1.0, param_2=2.0)
        assert strategy.get_params() == {"param_1": 1.0, "param_2": 2.0}
        strategy.set_params(param_1=3.0)
        assert strategy.get_params() == {"param_1": 3.0, "param_2": 2.0}
        with pytest.raises(ValueError):
            strategy.set_params(nonexistent_param=1.0)

    def test_wealth_sanity(self):
        """
        wealth[-1] == sum(pnl)
        """
        np.random.seed(42)
        universe = make_randomwalk()
        strategy = RandomStrategy(max_n_assets=5)

        strategy.run(universe)

        assert np.isclose(sum(strategy.history().pnl), strategy.wealth().values[-1])

    def test_history(self):
        universe = pd.DataFrame({"A": range(10), "B": range(10), "C": range(10)})
        trades = [
            trade("A", entry=1, exit=2, take=3, stop=-4),
            [2, -3] * trade(["B", "C"], entry=3, exit=9, take=5, stop=-2),
        ]
        strategy = DeterminedStrategy(trades).run(universe)
        history = strategy.history()

        expected = pd.DataFrame(
            {
                "trade_id": [0, 1, 1],
                "asset": ["A", "B", "C"],
                "lot": [1, 2, -3],
                "entry": [1, 3, 3],
                "close": [2, 5, 5],
                "exit": [2, 9, 9],
                "take": [3, 5, 5],
                "stop": [-4, -2, -2],
                "pnl": [1, 4, -6],
            }
        )
        pd.testing.assert_frame_equal(history, expected, check_dtype=False)

    def test_history_notrunerror(self):
        strategy = RandomStrategy()
        with pytest.raises(NotRunError):
            # epymetheus.exceptions.NotRunError: Strategy has not been run
            strategy.history()

    def test_wealth(self):
        # TODO test for when exit != close

        universe = pd.DataFrame({"A": range(10), "B": range(10), "C": range(10)})

        # TODO remove it when #129 will be fixed ---
        universe = universe.astype(float)
        # ---

        trades = [trade("A", entry=1, exit=3), trade("B", entry=2, exit=4)]
        strategy = DeterminedStrategy(trades).run(universe)
        wealth = strategy.wealth()

        expected = pd.Series([0, 0, 1, 3, 4, 4, 4, 4, 4, 4], index=universe.index)

        pd.testing.assert_series_equal(wealth, expected, check_dtype=False)

    def test_ts(self):
        universe = make_randomwalk()
        strategy = RandomStrategy().run(universe)
        trades = strategy.trades

        assert_equal(strategy.wealth().values, ts.wealth(trades, universe))
        assert_equal(strategy.drawdown().values, ts.drawdown(trades, universe))
        assert_equal(strategy.net_exposure().values, ts.net_exposure(trades, universe))
        assert_equal(strategy.abs_exposure().values, ts.abs_exposure(trades, universe))

    def test_notrunerror(self):
        strategy = RandomStrategy()
        with pytest.raises(NotRunError):
            strategy.wealth()
        with pytest.raises(NotRunError):
            strategy.history()
        with pytest.raises(NotRunError):
            strategy.drawdown()
        with pytest.raises(NotRunError):
            strategy.net_exposure()
        with pytest.raises(NotRunError):
            strategy.abs_exposure()

    @pytest.mark.parametrize("metric", metrics)
    def test_score(self, metric):
        np.random.seed(42)
        universe = make_randomwalk()
        strategy = RandomStrategy().run(universe)
        result = strategy.score(metric.__name__)
        expected = metric(strategy.trades, universe)

        assert result == expected

    @pytest.mark.parametrize("metric", metrics)
    def test_score_notrunerror(self, metric):
        strategy = RandomStrategy()

        with pytest.raises(NotRunError):
            strategy.score(metric)

    def test_score_deprecation_warning(self):
        """
        `strategy.evaluate` is deprecated
        """
        strategy = create_strategy(self.my_strategy, param_1=1.0, param_2=2.0)
        with pytest.raises(DeprecationWarning):
            strategy.evaluate(None)

    def test_repr(self):
        strategy = create_strategy(my_func, param_1=1.0, param_2=2.0)
        assert repr(strategy) == "strategy(my_func, param_1=1.0, param_2=2.0)"

        strategy = MyStrategy(param_1=1.0, param_2=2.0)
        assert repr(strategy) == "MyStrategy"


# import pytest

# import random
# import numpy as np

# from epymetheus import Trade, TradeStrategy
# from epymetheus.datasets import make_randomwalk
# from epymetheus.benchmarks import RandomTrader


# params_seed = [42]
# params_n_steps = [10, 1000]
# params_n_assets = [10, 100]
# params_n_trades = [10, 100]
# params_a = [1.23, -1.23]

# lots = [0.0, 1, 1.23, -1.23, 12345.678]


# class MultipleTradeStrategy(TradeStrategy):
#     """
#     Yield multiple trades.

#     Parameters
#     ----------
#     trades : iterable of Trade
#     """
#     def __init__(self, trades):
#         self.trades = trades

#     def logic(self, universe):
#         for trade in self.trades:
#             yield trade


# def make_random_trades(universe, n_trades, seed):
#     random_trader = RandomTrader(n_trades=n_trades, seed=seed)
#     trades = random_trader.run(universe).trades
#     return list(trades)  # for of array is slow


# def assert_add(history_0, history_1, history_A, attribute):
#     array_0 = getattr(history_0, attribute)
#     array_1 = getattr(history_1, attribute)
#     array_A = getattr(history_A, attribute)
#     array_01 = np.sort(np.concatenate([array_0, array_1]))
#     assert np.equal(array_01, np.sort(array_A)).all()


# def assert_mul(history_1, history_a, attribute, a=None):
#     array_1 = getattr(history_1, attribute)
#     array_a = getattr(history_a, attribute)
#     if a is not None:
#         array_1 *= float(a)

#     print(array_1, array_1.dtype)
#     print(array_a, array_a.dtype)

#     if array_1.dtype == np.float64:
#         assert np.allclose(array_1, array_a)
#     else:
#         assert (array_1 == array_a).all()


# # --------------------------------------------------------------------------------


# @pytest.mark.parametrize('seed', params_seed)
# @pytest.mark.parametrize('n_steps', params_n_steps)
# @pytest.mark.parametrize('n_assets', params_n_assets)
# @pytest.mark.parametrize('n_trades', params_n_trades)
# def test_linearity_add(seed, n_steps, n_assets, n_trades):
#     """
#     Test additivity of strategies for the following strategies:
#         - strategy_0 : yield (trade_00, trade_01, ...)
#         - strategy_1 : yield (trade_10, trade_11, ...)
#         - strategy_A : yield (trade_00, trade_01, ..., trade_10, trade_11, ...)
#     """
#     np.random.seed(seed)
#     random.seed(seed)

#     universe = make_randomwalk(n_steps, n_assets)

#     trades_0 = make_random_trades(universe, n_trades, seed + 0)
#     trades_1 = make_random_trades(universe, n_trades, seed + 1)
#     trades_A = trades_0 + trades_1

#     strategy_0 = MultipleTradeStrategy(trades=trades_0).run(universe)
#     strategy_1 = MultipleTradeStrategy(trades=trades_1).run(universe)
#     strategy_A = MultipleTradeStrategy(trades=trades_A).run(universe)

#     history_0 = strategy_0.history
#     history_1 = strategy_1.history
#     history_A = strategy_A.history

#     for attr in (
#         'asset',
#         'lot',
#         'entrys',
#         'exits',
#         'durations',
#         'open_prices',
#         'close_prices',
#         'gains',
#     ):
#         assert_add(history_0, history_1, history_A, attr)


# @pytest.mark.parametrize('seed', params_seed)
# @pytest.mark.parametrize('n_steps', params_n_steps)
# @pytest.mark.parametrize('n_assets', params_n_assets)
# @pytest.mark.parametrize('n_trades', params_n_trades)
# @pytest.mark.parametrize('a', params_a)
# def test_linearity_mul(seed, n_steps, n_assets, n_trades, a):
#     """
#     Test additivity of strategies for the following strategies:
#         - strategy_1 : yield (1 * trade_0, 1 * trade_11, ...)
#         - strategy_a : yield (a * trade_0, a * trade_01, ...)
#     """
#     np.random.seed(seed)
#     random.seed(seed)

#     universe = make_randomwalk(n_steps, n_assets)

#     trades_1 = make_random_trades(universe, n_trades, seed + 1)
#     trades_a = [
#         Trade(
#             asset=trade.asset,
#             lot=a * trade.lot,
#             entry=trade.entry,
#             exit=trade.exit
#         )
#         for trade in trades_1
#     ]

#     strategy_1 = MultipleTradeStrategy(trades=trades_1).run(universe)
#     strategy_a = MultipleTradeStrategy(trades=trades_a).run(universe)

#     history_1 = strategy_1.history
#     history_a = strategy_a.history

#     for attr in (
#         'asset',
#         'entrys',
#         'exits',
#         'durations',
#         'open_prices',
#         'close_prices',
#     ):
#         assert_mul(history_1, history_a, attr, None)

#     for attr in ('lot', 'gains'):
#         assert_mul(history_1, history_a, attr, a)
