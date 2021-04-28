import numpy as np

from .. import ts


def _pnls(trades, universe) -> np.array:
    return np.array([np.sum(t.final_pnl(universe)) for t in trades])


def final_wealth(trades, universe) -> float:
    # maybe faster than wealth[-1]
    return np.sum(_pnls(trades, universe))


def num_win(trades, universe) -> int:
    pnls = _pnls(trades, universe)
    return np.sum(pnls > 0)


def num_lose(trades, universe) -> int:
    pnls = _pnls(trades, universe)
    return np.sum(pnls <= 0)


def rate_win(trades, universe) -> float:
    return num_win(trades, universe) / len(trades)


def rate_lose(trades, universe) -> float:
    return num_lose(trades, universe) / len(trades)


def avg_win(trades, universe) -> float:
    pnls = _pnls(trades, universe)
    return np.mean(pnls[pnls > 0])


def avg_lose(trades, universe) -> float:
    pnls = _pnls(trades, universe)
    return np.mean(pnls[pnls <= 0])


def avg_pnl(trades, universe) -> float:
    pnls = _pnls(trades, universe)
    return np.mean(pnls)


def max_drawdown(trades, universe) -> float:
    return np.min(ts.drawdown(trades, universe))


# def avg_return(trades, universe):
#     return ...


# def volatility(trades, universe):
#     wealth = ts.wealth(trades, universe)
#     return np.std(np.diff(wealth))


# def sharpe_ratio(trades, universe):
#     return ...
