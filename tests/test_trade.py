import numpy as np
import pandas as pd
import pytest

from epymetheus import trade
from epymetheus.benchmarks import RandomStrategy
from epymetheus.datasets import make_randomwalk


class TestTrade:

    # handmade universe
    universe_hand = pd.DataFrame(
        {
            "A": [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0],
            "B": [2.0, 7.0, 1.0, 8.0, 2.0, 8.0, 1.0],
        },
    )

    def test_init_array(self):
        t = trade("A", lot=1.0)
        assert isinstance(t.asset, np.ndarray)
        assert isinstance(t.lot, np.ndarray)
        t = trade("A", lot=[1.0])
        assert isinstance(t.asset, np.ndarray)
        assert isinstance(t.lot, np.ndarray)
        t = trade(["A"], lot=[1.0])
        assert isinstance(t.asset, np.ndarray)
        assert isinstance(t.lot, np.ndarray)
        t = trade(["A", "B"], lot=1.0)
        t = trade(["A", "B"], lot=[1.0, 2.0])
        assert isinstance(t.asset, np.ndarray)
        assert isinstance(t.lot, np.ndarray)

    def test_init_shape(self):
        t = trade("A", lot=1.0)
        assert t.asset.shape == (1,)
        assert t.lot.shape == (1,)

        t = trade(["A"], lot=1.0)
        assert t.asset.shape == (1,)
        assert t.lot.shape == (1,)

        t = trade("A", lot=[1.0])
        assert t.asset.shape == (1,)
        assert t.lot.shape == (1,)

        t = trade(["A", "B"], lot=1.0)
        assert t.asset.shape == (2,)
        assert t.lot.shape == (2,)

        t = trade(["A", "B"], lot=[1.0])
        assert t.asset.shape == (2,)
        assert t.lot.shape == (2,)

        t = trade(["A", "B"], lot=[1.0, 2.0])
        assert t.asset.shape == (2,)
        assert t.lot.shape == (2,)

    def test_init_deprecation(self):
        with pytest.raises(DeprecationWarning):
            trade("A", open_bar=0)
        with pytest.raises(DeprecationWarning):
            trade("A", shut_bar=0)

    def test_repr(self):
        t = trade("A")
        assert repr(t) == "trade(['A'], lot=[1.])"

        t = trade("A", lot=2, take=3.0, stop=-3.0, entry="B0", exit="B1")

        assert (
            repr(t) == "trade(['A'], lot=[2], entry=B0, exit=B1, take=3.0, stop=-3.0)"
        )

    def test_array_value_value_hand(self):
        t = [2.0, -3.0] * trade(["A", "B"], entry=1, exit=3)
        result = t.array_value(self.universe_hand)
        expected = np.array(
            [
                [6.0, -6.0],
                [2.0, -21.0],
                [8.0, -3.0],
                [2.0, -24.0],
                [10.0, -6.0],
                [18.0, -24.0],
                [4.0, -3.0],
            ]
        )
        assert np.allclose(result, expected)

        t = [-3.0, 2.0] * trade(["B", "A"], entry=1, exit=3)
        result = t.array_value(universe=self.universe_hand)
        expected = expected[:, [1, 0]]
        assert np.allclose(result, expected)

    def test_array_value_value_zero(self):
        t = 0.0 * trade(["A", "B"], entry=1, exit=3)
        result = t.array_value(self.universe_hand)
        expected = np.zeros_like(self.universe_hand.iloc[:, :2])
        assert np.allclose(result, expected)

    @pytest.mark.parametrize("seed", [0])
    def test_array_value_linearity_add(self, seed):
        np.random.seed(seed)
        universe = self.universe_hand

        lot0, lot1 = np.random.randn(2), np.random.randn(2)
        t0 = list(lot0) * trade(["A", "B"], entry=1, exit=3)
        t1 = list(lot1) * trade(["A", "B"], entry=1, exit=3)
        ta = list(lot0 + lot1) * trade(["A", "B"], entry=1, exit=3)
        result = ta.array_value(universe)
        expected = t0.array_value(universe) + t1.array_value(universe)

        assert np.allclose(result, expected)

    @pytest.mark.parametrize("a", [-2.0, -1.0, 0.0, 1.0, 2.0])
    @pytest.mark.parametrize("seed", [0])
    def test_array_value_linearity_mul(self, a, seed):
        np.random.seed(seed)

        universe = self.universe_hand
        lot0 = np.random.randn(2)
        t0 = list(lot0) * trade(["A", "B"], entry=1, exit=3)
        ta = list(a * lot0) * trade(["A", "B"], entry=1, exit=3)
        result = ta.array_value(universe)
        expected = a * t0.array_value(universe)

        assert np.allclose(result, expected)

    @pytest.mark.parametrize("seed", [0])
    def test_final_pnl_lineality_add(self, seed):
        np.random.seed(seed)
        universe = self.universe_hand

        lot0, lot1 = np.random.randn(2), np.random.randn(2)
        t0 = list(lot0) * trade(["A", "B"], entry=1, exit=3)
        t1 = list(lot1) * trade(["A", "B"], entry=1, exit=3)
        ta = list(lot0 + lot1) * trade(["A", "B"], entry=1, exit=3)
        t0.execute(universe)
        t1.execute(universe)
        ta.execute(universe)
        result = ta.final_pnl(universe)
        expected = t0.final_pnl(universe) + t1.final_pnl(universe)

        assert np.allclose(result, expected)

    def test_nonexitent(self):
        """non-existent asset, entry, exit"""
        universe = pd.DataFrame({"A": range(10)})

        with pytest.raises(KeyError):
            t = trade("NONEXISTENT").execute(universe)
        with pytest.raises(KeyError):
            t = trade("A", entry=99).execute(universe)
        with pytest.raises(KeyError):
            t = trade("A", entry=0, exit=99).execute(universe)

    @pytest.mark.parametrize("a", [-2.0, -1.0, 0.0, 1.0, 2.0])
    @pytest.mark.parametrize("seed", [0])
    def test_final_pnl_linearity_mul(self, a, seed):
        np.random.seed(seed)

        universe = self.universe_hand
        lot0 = np.random.randn(2)
        t0 = list(lot0) * trade(["A", "B"], entry=1, exit=3)
        ta = list(a * lot0) * trade(["A", "B"], entry=1, exit=3)
        t0.execute(universe)
        ta.execute(universe)
        result = ta.final_pnl(universe)
        expected = a * t0.final_pnl(universe)

        assert np.allclose(result, expected)

    def test_eq(self):
        t = trade("A")
        assert t == trade("A")
        assert t == trade("A", lot=[1])
        assert t == trade("A", lot=[1.0])
        assert t != trade("A", lot=-1)
        assert t != trade("A", lot=[-1.0])

        t = trade("A", lot=2)
        assert t == trade("A", lot=2)
        assert t == trade("A", lot=2.0)
        assert t == trade("A", lot=[2])
        assert t == trade("A", lot=[2.0])
        assert t == trade("A", lot=np.array([2]))
        assert t == trade("A", lot=np.array([2.0]))
        assert t != trade("A", lot=-1)
        assert t != trade("A", lot=[-1.0])
        assert t != trade("A", lot=np.array([-1]))
        assert t != trade("A", lot=np.array([-1.0]))

        t = trade(["A", "B"], lot=[1, 2])
        assert t == trade(["A", "B"], lot=[1, 2])
        assert t == trade(["A", "B"], lot=[1.0, 2.0])
        assert t == trade(["A", "B"], lot=np.array([1, 2]))
        assert t == trade(["A", "B"], lot=np.array([1.0, 2.0]))
        assert t != trade(["A", "B"], lot=1.0)
        assert t != trade(["A", "B"], lot=[-1.0, 2.0])
        assert t != trade(["A", "B"], lot=[1.0, -1.0])

        t = trade(["A", "B"], lot=[1, 2], entry=1)
        assert t == trade(["A", "B"], lot=[1, 2], entry=1)
        assert t != trade(["A", "B"], lot=[1, 2], entry=2)

        t = trade(["A", "B"], lot=[1, 2], exit=1)
        assert t == trade(["A", "B"], lot=[1, 2], exit=1)
        assert t != trade(["A", "B"], lot=[1, 2], exit=2)

        t = trade(["A", "B"], lot=[1, 2], take=1)
        assert t == trade(["A", "B"], lot=[1, 2], take=1)
        assert t != trade(["A", "B"], lot=[1, 2], take=2)

        t = trade(["A", "B"], lot=[1, 2], stop=1)
        assert t == trade(["A", "B"], lot=[1, 2], stop=1)
        assert t != trade(["A", "B"], lot=[1, 2], stop=2)

    @pytest.mark.parametrize("a", [-2.0, -1.0, 0.0, 1.0, 2.0])
    def test_mul(self, a):
        t = trade("A", entry=0, exit=1, take=2.0, stop=-3.0)
        result = a * t
        expect = trade("A", lot=[a], entry=0, exit=1, take=2.0, stop=-3.0)
        assert result == expect
        result = t * a
        assert result == expect

        t = trade(["A", "B"], lot=[1.0, -2.0], entry=0, exit=1, take=2.0, stop=-3.0)
        result = a * t
        expect = trade(
            ["A", "B"], lot=[a, -2.0 * a], entry=0, exit=1, take=2.0, stop=-3.0
        )
        assert result == expect
        result = t * a
        assert result == expect

    def test_neg(self):
        t = trade("A", entry=0, exit=1, take=2.0, stop=-3.0)
        result = -t
        expect = trade("A", lot=[-1.0], entry=0, exit=1, take=2.0, stop=-3.0)
        assert result == expect

    @pytest.mark.parametrize("a", [-2.0, -1.0, 1.0, 2.0])
    def test_truediv(self, a):
        t = trade("A", entry=0, exit=1, take=2.0, stop=-3.0)
        result = t / a
        expect = trade("A", lot=[1 / a], entry=0, exit=1, take=2.0, stop=-3.0)
        assert result == expect

        t = trade(["A", "B"], lot=[1.0, -2.0], entry=0, exit=1, take=2.0, stop=-3.0)
        result = t / a
        expect = trade(
            ["A", "B"], lot=[1 / a, -2.0 / a], entry=0, exit=1, take=2.0, stop=-3.0
        )
        assert result == expect


# @pytest.mark.parametrize("seed", params_seed)
# def test_execute_0_0(seed):
#     """
#     Test `trade.execute` sets `trade.close_bar` correctly.

#     Setup
#     -----
#     - trade.take is None
#     - trade.stop is None
#     - trade.exit is not None

#     Expected Result
#     ---------------
#     trade.close_bar == universe.exit
#     """
#     # exit is not None
#     universe = make_randomwalk(seed=seed)
#     trade = make_random_trade(universe, seed=seed)
#     trade.execute(universe)

#     assert trade.close_bar == trade.exit


# @pytest.mark.parametrize("seed", params_seed)
# def test_execute_0_1(seed):
#     """
#     Test `trade.execute` sets `trade.close_bar` correctly.

#     Setup
#     -----
#     - trade.take is None
#     - trade.stop is None
#     - trade.exit is None

#     Expected Result
#     ---------------
#     trade.close_bar == universe.bars[-1]
#     """
#     # exit is not None
#     universe = make_randomwalk(seed=seed)
#     trade = make_random_trade(universe, seed=seed)
#     trade.exit = None
#     trade.execute(universe)

#     assert trade.close_bar == universe.bars[-1]


# @pytest.mark.parametrize("seed", params_seed)
# def test_execute(seed):
#     """
#     Test `trade.execute` sets `trade.close_bar` correctly.

#     Setup
#     -----
#     - trade.take is None
#     - trade.stop is None
#     - trade.exit is None

#     Expected Result
#     ---------------
#     trade.close_bar == universe.bars[-1]
#     """
#     # exit is not None
#     universe = make_randomwalk(seed=seed)
#     trade = make_random_trade(universe, seed=seed)
#     trade.exit = None
#     trade.execute(universe)

#     assert trade.close_bar == universe.bars[-1]


# # @pytest.mark.parametrize('seed', params_seed)
# # @pytest.mark.parametrize('n_bars', params_n_bars)
# # @pytest.mark.parametrize('const', params_const)
# # def test_execute(seed, n_bars, const):
# #     period = n_samples // 10
# #     shift = np.random.randint(period)
# #     prices = pd.DataFrame({
# #         'Asset0': const + make_sin(n_bars=n_bars, period=period, shift=shift)
# #     })
# #     universe = prices

# #     trade = ep.trade('Asset0', lot=1.0, )


# # def test_execute_take():
# #     universe = pd.DataFrame({"Asset0": np.arange(100, 200)}

# #     trade = ep.trade("Asset0", lot=1.0, take=1.9, entry=1, exit=5)
# #     trade.execute(universe)
# #     assert trade.close_bar == 3
# #     assert np.array_equal(trade.final_pnl(universe), [103 - 101])

# #     trade = ep.trade("Asset0", lot=2.0, take=3.8, entry=1, exit=5)
# #     trade.execute(universe)
# #     assert trade.close_bar == 3
# #     assert np.array_equal(trade.final_pnl(universe), [2 * (103 - 101)])

# #     trade = ep.trade("Asset0", lot=1.0, take=1000, entry=1, exit=5)
# #     trade.execute(universe)
# #     assert trade.close_bar == 5
# #     assert np.array_equal(trade.final_pnl(universe), [105 - 101])


# # def test_execute_stop():
# #     universe = prices=pd.DataFrame({"Asset0": np.arange(100, 0, -1)})

# #     trade = ep.trade("Asset0", lot=1.0, stop=-1.9, entry=1, exit=5)
# #     trade.execute(universe)
# #     assert trade.close_bar == 3
# #     assert np.array_equal(trade.final_pnl(universe), [97 - 99])

# #     trade = ep.trade("Asset0", lot=2.0, stop=-3.8, entry=1, exit=5)
# #     trade.execute(universe)
# #     assert trade.close_bar == 3
# #     assert np.array_equal(trade.final_pnl(universe), [2 * (97 - 99)])

# #     trade = ep.trade("Asset0", lot=1.0, stop=-1000, entry=1, exit=5)
# #     trade.execute(universe)
# #     assert trade.close_bar == 5
# #     assert np.array_equal(trade.final_pnl(universe), [95 - 99])


# # TODO both take and stop
# # TODO short position
# # TODO multiple orders

# # def test_execute_takestop():
# #     pass
