import pandas as pd

from epymetheus.stochastic import generate_geometric_brownian


def make_randomwalk(
    n_steps: int = 1000,
    n_assets: int = 10,
    volatility: float = 0.2,
    dt: float = 1 / 250,
    drift: float = 0.0,
    index=None,
    columns=None,
) -> pd.DataFrame:
    """Return `pandas.DataFrame` of random-walking prices (geometric Brownian motion).

    Random seed can be set by `np.random.seed`.

    Args:
        n_steps (int, default 1000): Number of time steps.
        n_assets (int, default 10): Number of assets.
        volatility (float, default=0.2): Volatility of the motion.
        dt (float, default=1/250): Intervals of time steps.
        drift (float, default=0.0): Drift of the motion.
        index (pandas.Index or array-like, optional): Index to use
            for resulting frame.
            Will default to `RangeIndex` if no index labels are provided.
        columns (pandas.Index or array-like, optional): Column labels to use
            for resulting frame.
            Will default to `RangeIndex` if no column labels are provided.

    Returns:
        pandas.DataFrame

    Examples:

        >>> import numpy as np
        >>> np.random.seed(42)
        >>> make_randomwalk(5, 2, columns=("A", "B"))
                  A         B
        0  1.000000  1.000000
        1  0.988262  0.999265
        2  0.965828  0.976582
        3  0.965805  0.966582
        4  0.941075  0.953967
    """
    data = generate_geometric_brownian(
        n_steps=n_steps, n_paths=n_assets, volatility=volatility, dt=dt, drift=drift
    )
    return pd.DataFrame(data, index=index, columns=columns)
