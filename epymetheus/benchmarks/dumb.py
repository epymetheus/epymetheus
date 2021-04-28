import pandas as pd

from ..trade import trade


def dumb_strategy(universe: pd.DataFrame, profit_take, stop_loss):
    # I get $100 allowance on the first business day of each month
    allowance = 100

    trades = []
    for date in pd.date_range(universe.index[0], universe.index[-1], freq="BMS"):
        cheapest_stock = universe.loc[date].idxmin()

        # Find the maximum number of shares that I can buy with my allowance
        n_shares = allowance // universe.at[date, cheapest_stock]

        t = n_shares * trade(cheapest_stock, date, take=profit_take, stop=stop_loss)
        trades.append(t)

    return trades
