import numpy as np


def wealth(trades, universe) -> np.array:
    wealth = np.zeros_like(universe.iloc[:, 0], dtype=float)
    for t in trades:
        i_entry = universe.index.get_indexer([t.entry]).item()
        i_entry = i_entry if i_entry != -1 else 0
        i_close = universe.index.get_loc(t.close)

        value = t.array_value(universe).sum(axis=1)
        pnl = value - value[i_entry]
        pnl[:i_entry] = 0
        pnl[i_close:] = pnl[i_close]

        wealth += pnl

    return wealth


def drawdown(trades, universe) -> np.array:
    # not drawdown rate
    w = wealth(trades, universe)
    return w - np.maximum.accumulate(w)


def _exposure(trades, universe, net: bool):
    exposure = np.zeros_like(universe.iloc[:, 0], dtype=float)
    for t in trades:
        i_entry = universe.index.get_indexer([t.entry]).item()
        i_close = universe.index.get_indexer([t.close]).item()
        value = t.array_value(universe).astype(exposure.dtype)
        value[:i_entry] = 0
        value[i_close + 1 :] = 0
        value = value if net else np.abs(value)
        exposure += value.sum(axis=1)

    return exposure


def net_exposure(trades, universe) -> np.array:
    return _exposure(trades, universe, net=True)


def abs_exposure(trades, universe) -> np.array:
    return _exposure(trades, universe, net=False)
