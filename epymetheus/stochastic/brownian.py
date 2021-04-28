import numpy as np


def generate_brownian(
    n_steps, n_paths, volatility, init_value=0.0, dt=1.0, drift=0.0
) -> np.array:
    """
    Examples
    --------
    >>> np.random.seed(42)
    >>> generate_brownian(10, 3, 0.01)
    array([[ 0.        ,  0.        ,  0.        ],
           [ 0.00496714, -0.00138264,  0.00647689],
           [ 0.02019744, -0.00372418,  0.00413552],
           [ 0.03598957,  0.00395017, -0.00055923],
           [ 0.04141517, -0.00068401, -0.00521653],
           [ 0.04383479, -0.01981681, -0.0224657 ],
           [ 0.03821192, -0.02994512, -0.01932323],
           [ 0.02913168, -0.04406816, -0.00466674],
           [ 0.02687391, -0.04339288, -0.01891422],
           [ 0.02143009, -0.04228365, -0.03042416]])
    """
    randn = np.random.randn(n_steps - 1, n_paths)
    zeros = np.zeros_like(randn[:1])
    drift = (drift * np.arange(n_steps)).reshape(-1, 1)

    w = np.concatenate((zeros, randn), 0).cumsum(0)

    return init_value + drift + volatility * np.sqrt(dt) * w


def generate_geometric_brownian(
    n_steps, n_paths, volatility, init_value=1.0, dt=1.0, drift=0.0
) -> np.array:
    """
    Examples
    --------
    >>> np.random.seed(42)
    >>> generate_geometric_brownian(10, 3, 0.01)
    array([[1.        , 1.        , 1.        ],
           [1.00492925, 0.99856838, 1.00644758],
           [1.02030075, 0.99618313, 1.00404367],
           [1.03648955, 1.0038074 , 0.99929102],
           [1.0420763 , 0.99911638, 0.99459812],
           [1.04454856, 0.98013319, 0.97754036],
           [1.03863974, 0.97020769, 0.98056805],
           [1.02919987, 0.95655388, 0.99499582],
           [1.02682746, 0.95715219, 0.9808711 ],
           [1.02120171, 0.95816656, 0.96959758]])
    """
    w = generate_brownian(
        n_steps=n_steps,
        n_paths=n_paths,
        volatility=volatility,
        init_value=0.0,
        dt=dt,
        drift=drift - (volatility ** 2 / 2),
    )

    return init_value * np.exp(w)
