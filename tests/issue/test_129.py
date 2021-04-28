import pandas as pd

from epymetheus import trade
from epymetheus.benchmarks import DeterminedStrategy


def test():
    universe = pd.DataFrame({"A": range(10)})
    trades = [trade("A", entry=1, exit=3)]
    strategy = DeterminedStrategy(trades).run(universe)
    wealth = strategy.wealth()
