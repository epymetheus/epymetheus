import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_equal

from epymetheus import trade
from epymetheus import ts
from epymetheus.benchmarks import RandomStrategy
from epymetheus.datasets import make_randomwalk
from epymetheus.strategy.container import StrategyList


class TestAbsExposure:
    @pytest.fixture(scope="function", autouse=True)
    def setup(self):
        np.random.seed(42)

    def test_zero(self):
        """
        result = 0 if trade is zero
        """
        universe = make_randomwalk()
        strategy = RandomStrategy(min_lot=0, max_lot=0).run(universe)

        result = ts.abs_exposure(strategy.trades, universe)
        expected = np.zeros_like(result)

        assert_equal(result, expected)

    def test_handmade(self):
        universe = pd.DataFrame(
            {
                "A": [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0],
                "B": [2.0, 7.0, 1.0, 8.0, 1.0, 8.0, 1.0],
            }
        )
        trades = [
            2 * trade("A", entry=1, exit=5).execute(universe),
            -3 * trade("B", entry=2, exit=4).execute(universe),
        ]
        result = ts.abs_exposure(trades, universe)
        expected = np.array([0.0, 2.0, 11.0, 26.0, 13.0, 18.0, 0.0])

        assert_equal(result, expected)


class TestDrawdown:
    @pytest.fixture(scope="function", autouse=True)
    def setup(self):
        np.random.seed(42)

    def test_zero(self):
        """
        result = 0 if trade is zero
        """
        universe = make_randomwalk()
        strategy = RandomStrategy(min_lot=0, max_lot=0).run(universe)

        result = ts.drawdown(strategy.trades, universe)
        expected = np.zeros_like(result)

        assert_equal(result, expected)

    def test_monotonous(self):
        """
        result = 0 if wealth is monotonously increasing
        """
        universe = pd.DataFrame({"A": np.linspace(1, 2, 100)})
        trades = [trade("A").execute(universe)]

        result = ts.drawdown(trades, universe)
        expected = np.zeros_like(result)

        assert_equal(result, expected)

    def test_handmade(self):
        universe = pd.DataFrame({"A": [0.0, 3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]})
        trades = [trade("A").execute(universe)]

        result = ts.drawdown(trades, universe)
        expected = np.array([0.0, 0.0, -2.0, 0.0, -3.0, 0.0, 0.0, -7.0])

        assert_equal(result, expected)


class TestNetExposure:
    @pytest.fixture(scope="function", autouse=True)
    def setup(self):
        np.random.seed(42)

    def test_zero(self):
        """
        result = 0 if trade is zero
        """
        universe = make_randomwalk()
        strategy = RandomStrategy(min_lot=0, max_lot=0).run(universe)

        result = ts.net_exposure(strategy.trades, universe)
        expected = np.zeros_like(result)

        assert_equal(result, expected)

    def test_linearity_add(self):
        universe = make_randomwalk()
        s0 = RandomStrategy().run(universe)
        s1 = RandomStrategy().run(universe)

        result = ts.net_exposure(s0.trades, universe) + ts.net_exposure(
            s1.trades, universe
        )
        expected = ts.net_exposure(s0.trades + s1.trades, universe)

        assert_allclose(result, expected)

    @pytest.mark.parametrize("a", [2.0, -1.0])
    def test_linearity_mul(self, a):
        universe = make_randomwalk()
        strategy = RandomStrategy().run(universe)

        result = a * ts.net_exposure(strategy.trades, universe)
        expected = ts.net_exposure([a * t for t in strategy.trades], universe)

        assert_allclose(result, expected)

    def test_handmade(self):
        universe = pd.DataFrame(
            {
                "A": [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0],
                "B": [2.0, 7.0, 1.0, 8.0, 1.0, 8.0, 1.0],
            }
        )
        trades = [
            2 * trade("A", entry=1, exit=5).execute(universe),
            -3 * trade("B", entry=2, exit=4).execute(universe),
        ]
        result = ts.net_exposure(trades, universe)
        expected = np.array([0.0, 2.0, 5.0, -22.0, 7.0, 18.0, 0.0])

        assert_equal(result, expected)


class TestWealth:
    @pytest.fixture(scope="function", autouse=True)
    def setup(self):
        np.random.seed(42)

    def test_zero(self):
        """
        result = 0 if trade is zero
        """
        universe = make_randomwalk()
        strategy = RandomStrategy(min_lot=0, max_lot=0).run(universe)

        result = ts.wealth(strategy.trades, universe)
        expected = np.zeros_like(result)

        assert_equal(result, expected)

    def test_linearity_add(self):
        universe = make_randomwalk()
        s0 = RandomStrategy().run(universe)
        s1 = RandomStrategy().run(universe)

        result = ts.wealth(s0.trades, universe) + ts.wealth(s1.trades, universe)
        expected = ts.wealth(s0.trades + s1.trades, universe)

        assert_allclose(result, expected)

    @pytest.mark.parametrize("a", [2.0, -1.0])
    def test_linearity_mul(self, a):
        universe = make_randomwalk()
        strategy = RandomStrategy().run(universe)

        result = a * ts.wealth(strategy.trades, universe)
        expected = ts.wealth([a * t for t in strategy.trades], universe)

        assert_allclose(result, expected)
