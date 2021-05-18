def train_test_roll(array, tr_samples, te_samples, roll=None):
    """
    Split arrays or matrices into rolling train and test subsets

    Parameters
    ----------
    array : indexable
        Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
    tr_samples : int
        Number of train samples.
    te_samples : int
        Number of test samples.
    roll : int, optional
        Rolling period. If None, set to ``te_samples``.

    Returns
    -------
    splitting : list
        List containing train-test split of inputs.

    Examples
    --------
    >>> array = list(range(5))
    >>> for i_tr, i_te in train_test_roll(array, 2, 1):
    ...     print(i_tr, i_te)
    [0, 1] [2]
    [1, 2] [3]
    [2, 3] [4]
    """
    if roll is None:
        roll = te_samples

    i = 0
    splitting = []
    while i + tr_samples + te_samples <= len(array):
        index_tr = array[i : i + tr_samples]
        index_te = array[i + tr_samples : i + tr_samples + te_samples]
        splitting.append((index_tr, index_te))
        i += roll

    return splitting
