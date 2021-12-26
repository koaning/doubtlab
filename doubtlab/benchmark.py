import numpy as np


def shuffle_labels(y, random_seed=42, n=None, p=None):
    """
    Shuffles labels for benchmarking. Recommended for regression.

    Either `p` or `n` should be given. Returns a tuple `(y_out, indicator)`-tuple.

    Arguments:
        y: array of labels
        random_seed: random seed
        n: number of labels to flip
        p: percentage of labels to flip

    Usage:

    ```python
    from doubtlab.benchmark import shuffle_labels

    # Let's pretend these are the actual labels
    y = np.random.normal(0, 1, 10000)

    # You now have some shuffled labels and an indicator
    y_out, indicator = shuffle_labels(y, n=100)
    ```
    """
    np.random.seed(random_seed)
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

    y_out = y.copy()
    sample = np.random.choice(np.arange(y.shape[0]), size=n, replace=False)

    # Since `sample` is already randomly shuffled, we can move everything
    # over by one index to guarantee a shuffle of the values from another index.
    y_out[sample] = np.concatenate([sample[1:], sample[0:1]])
    return y_out, (y != y_out).astype(int)
