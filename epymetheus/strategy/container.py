from collections import OrderedDict

from .base import Strategy


class StrategyContainer(Strategy):
    """
    Attributes
    ----------
    - _strategies : OrderedDict

    >>> from epymetheus import trade
    >>> from epymetheus import create_strategy

    >>> strategy1 = create_strategy(lambda universe: [trade("A")])
    >>> strategy2 = create_strategy(lambda universe: [trade("B"), trade("C")])
    >>> s = StrategyList([strategy1, strategy2])
    >>> len(s)
    2
    """

    def __init__(self, **kwargs):
        super().__init__()

        self._strategies = OrderedDict()

        for key, value in kwargs.items():
            self.add_strategy(key, value)

    def __len__(self):
        return len(self._strategies)

    def __iter__(self):
        return iter(self._strategies.values())

    def keys(self):
        return self._strategies.keys()

    def values(self):
        return self._strategies.values()

    def items(self):
        return self._strategies.items()

    def add_strategy(self, name, strategy):
        if not isinstance(name, str):
            raise TypeError
        if not isinstance(strategy, Strategy):
            raise TypeError

        self._strategies[name] = strategy

    def logic(self, universe):
        for strategy in self:
            for trade in strategy(universe, to_list=False):
                yield trade


class StrategyList(StrategyContainer):
    """
    Holds strategies in a list.

    Parameters
    ----------
    - strategies : iterable of Strategies

    Examples
    --------
    >>> from epymetheus import trade
    >>> from epymetheus import create_strategy

    >>> strategy1 = create_strategy(lambda universe: [trade("A")])
    >>> strategy2 = create_strategy(lambda universe: [trade("B"), trade("C")])
    >>> strategy_list = StrategyList([strategy1, strategy2])
    >>> universe = ...
    >>> strategy_list(universe)
    [trade(['A'], lot=[1.]), trade(['B'], lot=[1.]), trade(['C'], lot=[1.])]
    """

    def __init__(self, strategies: list):
        super().__init__(**{self.__get_key(i): s for i, s in enumerate(strategies)})

    def __get_key(self, i: int) -> str:
        return f"strategy_{i}"

    def __repr__(self):
        """
        >>> from epymetheus import trade
        >>> from epymetheus import create_strategy

        >>> strategy1 = create_strategy(lambda universe: [trade("A")])
        >>> strategy2 = create_strategy(lambda universe: [trade("B"), trade("C")])
        >>> strategy_list = StrategyList([strategy1, strategy2])
        >>> strategy_list
        StrategyList([strategy(<lambda>), strategy(<lambda>)])
        """
        return f"{self.__class__.__name__}({self.list()})"

    def __getitem__(self, i: int) -> Strategy:
        return self.list()[i]

    def __setitem__(self, i: int, strategy) -> None:
        self.add_strategy(self.__get_key(i), strategy)

    def list(self):
        return list(iter(self))

    def append(self, strategy) -> None:
        self.add_strategy(self.__get_key(len(self)), strategy)


class StrategyDict(StrategyContainer):
    """
    Holds strategies in a list.

    Parameters
    ----------
    - strategies : iterable of Strategies

    Examples
    --------
    >>> from epymetheus import trade
    >>> from epymetheus import create_strategy

    >>> strategy1 = create_strategy(lambda universe: [trade("A")])
    >>> strategy2 = create_strategy(lambda universe: [trade("B"), trade("C")])
    >>> strategy_dict = StrategyDict({"S1": strategy1, "S2": strategy2})
    >>> universe = ...
    >>> strategy_dict(universe)
    [trade(['A'], lot=[1.]), trade(['B'], lot=[1.]), trade(['C'], lot=[1.])]
    """

    def __init__(self, strategies: dict):
        super().__init__(**strategies)

    def __repr__(self):
        """
        >>> from epymetheus import trade
        >>> from epymetheus import create_strategy

        >>> strategy1 = create_strategy(lambda universe: [trade("A")])
        >>> strategy2 = create_strategy(lambda universe: [trade("B"), trade("C")])
        >>> strategy_dict = StrategyDict({"S1": strategy1, "S2": strategy2})
        >>> strategy_dict
        StrategyDict(OrderedDict([('S1', strategy(<lambda>)), \
('S2', strategy(<lambda>))]))
        """
        return f"{self.__class__.__name__}({self.dict()})"

    def __getitem__(self, key):
        return self._strategies[key]

    def __setitem__(self, key, value):
        self.add_strategy(key, value)

    def dict(self):
        return self._strategies
