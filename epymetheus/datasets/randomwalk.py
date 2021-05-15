import pandas as pd

from epymetheus.stochastic import generate_geometric_brownian


def make_randomwalk(
    n_steps=1000,
    n_assets=10,
    volatility=0.01,
    init_value=1.0,
    dt=1.0,
    drift=0.0,
    bars=None,
    assets=None,
) -> pd.DataFrame:
    """
    Return `pandas.DataFrame` of random-walking prices (geometric Brownian motion).
    Daily returns follow log-normal distribution.
    Seed can be set by `np.random.seed(...)`.

    Parameters
    ----------
    n_steps : int, default 1000
        Number of time steps.
    n_assets : int, default 10
        Number of assets.
    volatility : float, default 0.01
        Volatility of asset prices.
    bars : list[str], optional
        Names of time steps.
    assets : list[str], optional
        Names of assets.
    seed : int
        Random seed to generate Brownian motion.

    Returns
    -------
    universe : pandas.DataFrame
        DataFrame with of asset prices that follow Brownian motions.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> make_randomwalk(10, 3)
              0         1         2
    0  1.000000  1.000000  1.000000
    1  1.004929  0.998568  1.006448
    2  1.020301  0.996183  1.004044
    3  1.036490  1.003807  0.999291
    4  1.042076  0.999116  0.994598
    5  1.044549  0.980133  0.977540
    6  1.038640  0.970208  0.980568
    7  1.029200  0.956554  0.994996
    8  1.026827  0.957152  0.980871
    9  1.021202  0.958167  0.969598
    """
    data = generate_geometric_brownian(
        n_steps=n_steps,
        n_paths=n_assets,
        volatility=volatility,
        init_value=init_value,
        dt=dt,
        drift=drift,
    )
    index = bars or list(range(n_steps))
    columns = assets or [str(i) for i in range(n_assets)]
    return pd.DataFrame(data, index=index, columns=columns)
