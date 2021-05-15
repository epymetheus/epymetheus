def print_if_verbose(*objects, verbose=True, **kwargs):
    """
    Print objects to the text stream if `verbose=True``.
    Otherwise, do nothing.
    """
    if verbose:
        print(*objects, **kwargs)
