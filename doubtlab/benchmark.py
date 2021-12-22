import numpy as np


def shuffle_labels(y, random_seed=42, p=None, n=None):
    """Shuffles labels for benchmarking. Recommended for regression."""
    np.random.seed(random_seed)
    if p:
        flip_indicator = np.random.random(y.shape[0]) < p
    elif n:
        flip_indicator = np.zeros(y.shape[0])
        flip_indicator[np.random.choice(np.arange(y.shape[0]), n)] = 1
    else:
        raise ValueError("Either `n` or `p` must be given to `shuffle_labels`.")

    y_out = y.copy()
    to_flip = y_out[flip_indicator]
    np.random.shuffle(to_flip)
    y_out[flip_indicator] = to_flip
    return y_out, flip_indicator


def flip_labels(y, random_seed=42, p=None, n=None):
    """Flips labels for benchmarking. Recommended for classification."""
    np.random.seed(random_seed)
    if p:
        flip_indicator = np.random.random(y.shape[0]) < p
    elif n:
        flip_indicator = np.zeros(y.shape[0])
        flip_indicator[np.random.choice(np.arange(y.shape[0]))] = 1
    else:
        raise ValueError("Either `n` or `p` must be given to `flip_labels`.")

    # Generate values to be flipped
    uniq_y = np.unique(y)
    tiled = np.tile(uniq_y, (y.shape[0], 1))
    idx = np.argwhere(tiled != y.reshape(-1, 1))
    # Apply trick found in https://stackoverflow.com/questions/56534309/efficiently-apply-different-permutations-for-each-row-of-a-2d-numpy-array
    allowed = (
        idx[:, 1].copy().reshape(idx.shape[0] // (len(uniq_y) - 1), (len(uniq_y) - 1))
    )
    y_flipped = np.take_along_axis(
        allowed, np.random.randn(*allowed.shape).argsort(axis=1), axis=1
    )[:, 0]

    # Finally, flip only indicated indices
    y_out = y.copy()
    y_out[flip_indicator] = y_flipped[flip_indicator]
    return y_out, flip_indicator
