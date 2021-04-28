from .metrics import avg_lose
from .metrics import avg_pnl
from .metrics import avg_win
from .metrics import final_wealth
from .metrics import max_drawdown
from .metrics import num_lose
from .metrics import num_win
from .metrics import rate_lose
from .metrics import rate_win


def metric_from_name(name: str):
    """
    Return metrics from name.
    """
    metrics = (
        avg_lose,
        avg_pnl,
        avg_win,
        final_wealth,
        max_drawdown,
        num_lose,
        num_win,
        rate_lose,
        rate_win,
    )
    return {m.__name__: m for m in metrics}[name]
