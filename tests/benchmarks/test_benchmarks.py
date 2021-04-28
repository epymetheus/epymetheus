import pandas as pd
import pytest

from epymetheus import create_strategy
from epymetheus.benchmarks import BuyAndHold
from epymetheus.benchmarks import dumb_strategy
from epymetheus.datasets import make_randomwalk


class TestBuyAndHold:
    def test(self):
        universe = pd.DataFrame({"A": [1, 2, 3], "B": [2, 3, 4], "C": [3, 4, 5]})
        strategy = BuyAndHold({"A": 0.5, "B": 0.5}).run(universe)

        assert len(strategy.trades) == 1
        assert (strategy.trades[0].asset == ["A", "B"]).all()
        assert (strategy.trades[0].lot == [0.5 / 1, 0.5 / 2]).all()
        assert strategy.trades[0].entry == 0
        assert strategy.trades[0].close == 2


class TestDumbStrategy:
    def test(self):
        universe = make_randomwalk()
        universe.index = pd.date_range("2000-01-01", "2020-12-31")[
            : universe.index.size
        ]
        strategy = create_strategy(dumb_strategy, profit_take=2.0, stop_loss=-1.0)
        strategy.run(universe)
