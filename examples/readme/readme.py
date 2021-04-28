import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn


def print_as_comment(obj):
    print("\n".join(f"# {line}" for line in str(obj).splitlines()))


if __name__ == "__main__":
    sys.path.append("../..")
    seaborn.set_style("whitegrid")

    # ---

    import pandas as pd

    import epymetheus as ep
    from epymetheus.benchmarks import dumb_strategy

    # ---

    my_strategy = ep.create_strategy(dumb_strategy, profit_take=20.0, stop_loss=-10.0)

    # ---

    from epymetheus.datasets import fetch_usstocks

    universe = fetch_usstocks()
    print(">>> universe.head()")
    print_as_comment(universe.head())

    print(">>> my_strategy.run(universe)")
    my_strategy.run(universe)

    # ---

    df_history = my_strategy.history()
    df_history.head()
    print(">>> df_history.head()")
    print_as_comment(df_history.head())

    # ---

    series_wealth = my_strategy.wealth()

    print(">>> series_wealth.head()")
    print_as_comment(series_wealth.head())

    plt.figure(figsize=(16, 4))
    plt.plot(series_wealth, linewidth=1)
    plt.xlabel("date")
    plt.ylabel("wealth [USD]")
    plt.title("Wealth")
    plt.savefig("wealth.png", bbox_inches="tight", pad_inches=0.1)

    # ---

    print(">>> my_strategy.score('final_wealth')")
    print_as_comment(my_strategy.score("final_wealth"))
    print(">>> my_strategy.score('max_drawdown')")
    print_as_comment(my_strategy.score("max_drawdown"))
    # my_strategy.score("sharpe_ratio")

    # ---

    drawdown = my_strategy.drawdown()
    exposure = my_strategy.net_exposure()

    plt.figure(figsize=(16, 4))
    plt.plot(pd.Series(drawdown, index=universe.index), linewidth=1)
    plt.xlabel("date")
    plt.ylabel("drawdown [USD]")
    plt.title("Drawdown")
    plt.savefig("drawdown.png", bbox_inches="tight", pad_inches=0.1)

    plt.figure(figsize=(16, 4))
    plt.plot(pd.Series(exposure, index=universe.index), linewidth=1)
    plt.xlabel("date")
    plt.ylabel("net exposure [USD]")
    plt.title("Net exposure")
    plt.savefig("net_exposure.png", bbox_inches="tight", pad_inches=0.1)

    # ---

    plt.figure(figsize=(16, 4))
    plt.hist(my_strategy.history().pnl, bins=100)
    plt.axvline(0, ls="--", color="k")
    plt.xlabel("profit and loss")
    plt.ylabel("number of trades")
    plt.title("Profit-loss distribution")
    plt.savefig("pnl.png", bbox_inches="tight", pad_inches=0.1)
