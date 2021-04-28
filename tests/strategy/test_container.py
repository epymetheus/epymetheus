import re
from collections import OrderedDict

import pytest

from epymetheus import create_strategy
from epymetheus import trade
from epymetheus.benchmarks import RandomStrategy
from epymetheus.strategy.container import StrategyContainer
from epymetheus.strategy.container import StrategyDict
from epymetheus.strategy.container import StrategyList


class TestStrategyContainer:
    """
    Test StrategyContainer
    """

    def test_len(self):
        s0 = create_strategy(lambda universe: [])
        s1 = create_strategy(lambda universe: [])
        container = StrategyContainer()
        assert len(container) == 0
        container.add_strategy("s0", s0)
        assert len(container) == 1
        container.add_strategy("s1", s1)
        assert len(container) == 2

    def test_iter(self):
        s0 = create_strategy(lambda universe: [])
        s1 = create_strategy(lambda universe: [])
        container = StrategyContainer(s0=s0, s1=s1)

        for i, strategy in enumerate(container):
            assert strategy == [s0, s1][i]

    def test_key(self):
        s0 = create_strategy(lambda universe: [])
        s1 = create_strategy(lambda universe: [])
        container = StrategyContainer(s0=s0, s1=s1)

        for i, name in enumerate(container.keys()):
            assert name == ["s0", "s1"][i]

    def test_values(self):
        s0 = create_strategy(lambda universe: [])
        s1 = create_strategy(lambda universe: [])
        container = StrategyContainer(s0=s0, s1=s1)

        for i, strategy in enumerate(container.values()):
            assert strategy == [s0, s1][i]

    def test_items(self):
        s0 = create_strategy(lambda universe: [])
        s1 = create_strategy(lambda universe: [])
        container = StrategyContainer(s0=s0, s1=s1)

        for (k1), (k2, _) in zip(container.keys(), container.items()):
            assert k1 == k2
        for (v1), (_, v2) in zip(container.values(), container.items()):
            assert v1 == v2

    def test_add_strategy(self):
        s0 = create_strategy(lambda universe: [])
        s1 = create_strategy(lambda universe: [])

        container = StrategyContainer()
        assert container._strategies == OrderedDict()
        container.add_strategy("s0", s0)
        assert container._strategies == OrderedDict({"s0": s0})
        container.add_strategy("s1", s1)
        assert container._strategies == OrderedDict({"s0": s0, "s1": s1})

    def test_add_strategy_typeerror(self):
        s = create_strategy(lambda universe: [])
        container = StrategyContainer()
        with pytest.raises(TypeError):
            container.add_strategy(0, s)
        with pytest.raises(TypeError):
            container.add_strategy("name", None)

    def test_logic(self):
        s0 = create_strategy(lambda universe: [trade("A")])
        s1 = create_strategy(lambda universe: [trade("B"), trade("C")])
        container = StrategyContainer(s0=s0, s1=s1)
        universe = ...

        result = container(universe)
        assert container(universe) == s0(universe) + s1(universe)


class TestStrategyList(TestStrategyContainer):
    def test_repr(self):
        s0 = create_strategy(lambda universe: [])
        s1 = create_strategy(lambda universe: [])
        container = StrategyList([s0, s1])
        assert (
            repr(container) == "StrategyList([strategy(<lambda>), strategy(<lambda>)])"
        )

    def test_getitem(self):
        s0 = create_strategy(lambda universe: [])
        s1 = create_strategy(lambda universe: [])
        container = StrategyList([s0, s1])

        assert container[0] == s0
        assert container[1] == s1

    def test_setitem(self):
        s0 = create_strategy(lambda universe: [])
        s1 = create_strategy(lambda universe: [])
        t0 = create_strategy(lambda universe: [])
        t1 = create_strategy(lambda universe: [])
        container = StrategyList([s0, s1])

        container[0] = t0
        assert container[0] == t0

        container[1] = t1
        assert container[1] == t1

    def test_list(self):
        s0 = create_strategy(lambda universe: [])
        s1 = create_strategy(lambda universe: [])
        container = StrategyList([s0, s1])

        assert container.list() == [s0, s1]

    def test_append(self):
        s0 = create_strategy(lambda universe: [])
        s1 = create_strategy(lambda universe: [])
        container = StrategyList([])
        container.append(s0)
        assert container[0] == s0
        container.append(s1)
        assert container[1] == s1


class TestStrategyDict(TestStrategyContainer):
    def test_repr(self):
        s0 = create_strategy(lambda universe: [])
        s1 = create_strategy(lambda universe: [])
        container = StrategyDict({"s0": s0, "s1": s1})
        assert (
            repr(container)
            == "StrategyDict(OrderedDict([('s0', strategy(<lambda>)), ('s1', strategy(<lambda>))]))"
        )

    def test_getitem(self):
        s0 = create_strategy(lambda universe: [])
        s1 = create_strategy(lambda universe: [])
        container = StrategyDict({"s0": s0, "s1": s1})

        assert container["s0"] == s0
        assert container["s1"] == s1

    def test_setitem(self):
        s0 = create_strategy(lambda universe: [])
        s1 = create_strategy(lambda universe: [])
        t0 = create_strategy(lambda universe: [])
        t1 = create_strategy(lambda universe: [])
        container = StrategyDict({"s0": s0, "s1": s1})

        container["s0"] = t0
        assert container["s0"] == t0

        container["s1"] = t1
        assert container["s1"] == t1

    def test_dict(self):
        s0 = create_strategy(lambda universe: [])
        s1 = create_strategy(lambda universe: [])
        container = StrategyDict({"s0": s0, "s1": s1})

        assert container.dict() == OrderedDict(s0=s0, s1=s1)
