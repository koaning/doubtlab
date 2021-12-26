import numpy as np


def _parse_check_p_n_y(p, n, y):
    """Parses and checks `n`, `y` and `p`, returns (inferred) `n`."""
    if p:
        if p < 0:
            raise ValueError("Probability value `p` must be larger than 0.")
        if p > 1:
            raise ValueError("Probability value `p` must be less than 1.")
        n = round(len(y) * p)
    if not n:
        raise ValueError("Either `n` or `p` must be given to `shuffle_labels`.")
    if n <= 1:
        raise ValueError("Must shuffle at least 2 values. Increase `n` or `p`.")
    return n


def shuffle_labels(y, random_seed=42, n=None, p=None):
    """
    Shuffles subset of labels for benchmarking. Recommended for regression.

    Either `p` or `n` should be given. Returns a tuple `(y_out, indicator)`-tuple.

    Arguments:
        y: array of labels
        random_seed: random seed
        n: number of labels to flip
        p: percentage of labels to flip

    Usage:

    ```python
    import numpy as np
    from doubtlab.benchmark import shuffle_labels

    # Let's pretend these are the actual labels
    y = np.random.normal(0, 1, 10000)

    # You now have some shuffled labels and an indicator
    y_out, indicator = shuffle_labels(y, n=100)
    ```
    """
    np.random.seed(random_seed)
    y = np.array(y)
    n = _parse_check_p_n_y(p=p, n=n, y=y)

    y_out = y.copy()
    sample = np.random.choice(np.arange(y.shape[0]), size=n, replace=False)

    # Since `sample` is already randomly shuffled, we can move everything
    # over by one index to guarantee a shuffle of the values from another index.
    y_out[sample] = np.concatenate([sample[1:], sample[0:1]])
    return y_out, (y != y_out).astype(int)


def flip_labels(y, random_seed=42, n=None, p=None):
    """
    Flips subset of labels for benchmarking. Recommended for classification.

    Either `p` or `n` should be given. Returns a tuple `(y_out, indicator)`-tuple.

    Arguments:
        y: array of labels
        random_seed: random seed
        n: number of labels to flip
        p: percentage of labels to flip

    Usage:

    ```python
    import numpy as np
    from doubtlab.benchmark import flip_labels

    # Let's pretend these are the actual labels
    y = np.random.randint(0, 3, 10000)

    # You now have some shuffled labels and an indicator
    y_out, indicator = flip_labels(y, n=100)
    ```
    """
    np.random.seed(random_seed)
    y = np.array(y)
    n = _parse_check_p_n_y(p=p, n=n, y=y)

    y_out = y.copy()
    classes = np.unique(y)
    if len(classes) == 1:
        raise ValueError("Need more that 1 class in `y`.")

    # Only sample classes that didn't appear before.
    idx = np.random.choice(np.arange(y.shape[0]), size=n, replace=False)
    y_out[idx] = [np.random.choice(classes[classes != _]) for _ in y_out[idx]]
    return y_out, (y != y_out).astype(int)
