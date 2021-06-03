import json

import numpy as np


def print_if_verbose(*objects, verbose=True, **kwargs):
    """
    Print objects to the text stream if `verbose=True``.
    Otherwise, do nothing.
    """
    if verbose:
        print(*objects, **kwargs)


class _JSONEncoderNumpy(json.JSONEncoder):
    """
    JSON encoder that can seriealize numpy objects.
    """

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return super().default(o)


def to_json(o) -> str:
    return _JSONEncoderNumpy().encode(o)
