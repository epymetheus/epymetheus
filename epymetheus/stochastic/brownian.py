import numpy as np


def generate_brownian(
    n_steps: int,
    n_paths: int,
    volatility: float = 0.2,
    dt: float = 1 / 250,
    drift: float = 0.0,
) -> np.ndarray:
    """Generate Brownian motion.

    Args:
        n_steps (int): Number of time steps.
        n_paths (int): Number of simulated paths.
        volatility (float, default=0.2): Volatility of the motion.
        dt (float, default=1/250): Intervals of time steps.
        drift (float, default=0.0): Drift of the motion.

    Examples:

        >>> np.random.seed(42)
        >>> generate_brownian(5, 2)
        array([[0.        , 0.        ],
               [0.00819268, 0.01926497],
               [0.00523085, 0.01630335],
               [0.02520649, 0.02601072],
               [0.01926806, 0.03287362]])
    """
    randn = np.random.randn(n_steps, n_paths)
    randn[0] = 0.0

    w = randn.cumsum(0)
    drift = (drift * np.arange(n_steps)).reshape(-1, 1)

    return drift + volatility * np.sqrt(dt) * w


def generate_geometric_brownian(
    n_steps: int,
    n_paths: int,
    volatility: float = 0.2,
    dt: float = 1 / 250,
    drift: float = 0.0,
) -> np.ndarray:
    """Generate geometric Brownian motion.

    Args:
        n_steps (int): Number of time steps.
        n_paths (int): Number of simulated paths.
        volatility (float, default=0.2): Volatility of the motion.
        dt (float, default=1/250): Intervals of time steps.
        drift (float, default=0.0): Drift of the motion.

    Examples:

        >>> np.random.seed(42)
        >>> generate_geometric_brownian(5, 2)
        array([[1.        , 1.        ],
               [0.98826212, 0.99926524],
               [0.96582835, 0.97658191],
               [0.96580482, 0.96658186],
               [0.94107547, 0.95396683]])
    """
    w = generate_brownian(
        n_steps=n_steps,
        n_paths=n_paths,
        volatility=volatility,
        dt=dt,
        drift=drift - (volatility ** 2 / 2),
    )

    return np.exp(w)
