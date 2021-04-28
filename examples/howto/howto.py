import matplotlib.pyplot as plt
import pandas as pd
import seaborn
from pandas.plotting import register_matplotlib_converters
from pandas.tseries.offsets import DateOffset

from epymetheus import Trade
from epymetheus import TradeStrategy
from epymetheus.datasets import fetch_usstocks

register_matplotlib_converters()
seaborn.set_style("whitegrid")


class SimpleTrendFollower(TradeStrategy):
    """
    A simple trend-following strategy.
    Buys stocks for a month with the highest percentile of one month returns.

    Parameters
    ----------
    - percentile : float
    - bet : float
    - take : float
    - stop : float
    - loolback : DateOffset
    - hold : DateOffset
    """

    def __init__(
        self,
        percentile=0.2,
        bet=10000,
        take=5000,
        stop=-1000,
        lookback=DateOffset(months=1),
        hold=DateOffset(months=6),
    ):
        self.percentile = percentile
        self.bet = bet
        self.take = take
        self.stop = stop
        self.lookback = lookback
        self.hold = hold

    def sorted_assets(self, universe, open_date):
        """
        Return list of asset sorted according to one-month returns.
        Sort is ascending (poor-return first).

        Returns
        -------
        list
        """
        onemonth_returns = (
            universe.prices.loc[open_date]
            / universe.prices.loc[open_date - self.lookback]
        )
        return list(onemonth_returns.sort_values().index)

    def logic(self, universe):
        n_trade = int(universe.n_assets * self.percentile)
        date_range = pd.date_range(universe.bars[0], universe.bars[-1], freq="BM")

        for open_date in date_range[1:]:
            for asset in self.sorted_assets(universe, open_date)[:n_trade]:
                yield Trade(
                    asset=asset,
                    lot=self.bet / universe.prices.at[open_date, asset],
                    open_bar=open_date,
                    shut_bar=open_date + self.hold,
                    take=self.take,
                    stop=self.stop,
                )


def plot(strategy):
    plt.figure(figsize=(16, 4))
    df_wealth = strategy.wealth.to_dataframe()
    plt.plot(df_wealth, linewidth=1)
    plt.title("Wealth / USD")
    plt.savefig("wealth.png", bbox_inches="tight", pad_inches=0.1)

    plt.figure(figsize=(16, 4))
    plt.hist(strategy.history.pnl, bins=100)
    plt.axvline(0, ls="--", color="red")
    plt.title("Gains")
    plt.savefig("pnl.png", bbox_inches="tight", pad_inches=0.1)

    # df_exposure = pd.Series(strategy.net_exposure, index=strategy.universe.bars)

    # plt.figure(figsize=(16, 4))
    # plt.plot(df_exposure)
    # plt.axhline(0, ls='--', color='gray')
    # plt.title('Net exposure')
    # plt.savefig('exposure.png', bbox_inches="tight", pad_inches=0.1)

    with open("history.md", "w") as f:
        f.write(strategy.history.to_dataframe().to_markdown())


def main():
    universe = fetch_usstocks(n_assets=10)

    strategy = SimpleTrendFollower()
    strategy.run(universe)

    plot(strategy)


if __name__ == "__main__":
    main()
