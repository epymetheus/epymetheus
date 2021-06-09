import sys

import matplotlib.pyplot as plt
import pandas as pd


def print_as_comment(obj):
    print("\n".join(f"# {line}" for line in str(obj).splitlines()))


if __name__ == "__main__":
    sys.path.append("..")

    # ---

    import pandas as pd

    import epymetheus as ep
    from epymetheus.benchmarks import dumb_strategy

    # ---

    my_strategy = ep.create_strategy(dumb_strategy, profit_take=20.0, stop_loss=-10.0)

    # ---

    from epymetheus.datasets import fetch_usstocks

    universe = fetch_usstocks(begin_date="20200101", end_date="20201231")

    print(">>> my_strategy.run(universe)")
    my_strategy.run(universe)

    # ---

    with open("trades.json", "w") as f:
        f.write(my_strategy.trades_to_json())
        print(my_strategy.trades)

    with open("trades.json") as f:
        my_strategy = ep.create_strategy(None)
        my_strategy.load_universe(universe)
        my_strategy.load_trades_json(f.read())
        print(my_strategy.trades)

    print(">>> my_strategy.score(\"final_wealth\")")
    my_strategy.score("final_wealth")

    # ---

    my_strategy.history().to_csv("history.csv")

    with open("trades.json") as f:
        my_strategy = ep.create_strategy(None)
        my_strategy.load_universe(universe)
        my_strategy.load_history(pd.read_csv("history.csv"))
        print(my_strategy.trades)

    print(">>> my_strategy.score(\"final_wealth\")")
    my_strategy.score("final_wealth")