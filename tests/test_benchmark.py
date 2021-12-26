import pytest

import numpy as np
from doubtlab.benchmark import shuffle_labels


@pytest.mark.parametrize("n", [100, 200, 500, 1000])
def test_shuffle_labels_n(n):
    """Test some basic properties."""
    y = np.random.normal(0, 1, 10000)
    y_out, indicator = shuffle_labels(y, n=n)
    assert indicator.sum() == n
    flipped = (y != y_out).astype(np.int8)
    print(flipped.sum())
    assert np.all(flipped == indicator)


@pytest.mark.parametrize("p", [0.1, 0.2, 0.3, 0.5])
def test_shuffle_labels_p(p):
    """Test some basic properties."""
    y = np.random.normal(0, 1, 10000)
    y_out, indicator = shuffle_labels(y, p=p)
    flipped = (y != y_out).astype(np.int8)
    print(flipped.sum())
    assert np.all(flipped == indicator)


def test_raise_error_bad_n():
    """If n == 1 then we can't really shuffle so we raise an error."""
    with pytest.raises(ValueError):
        y = np.random.normal(0, 1, 10000)
        shuffle_labels(y, n=1)


def test_raise_error_no_n_or_p():
    """Gotta make sure one of the values is given."""
    with pytest.raises(ValueError):
        y = np.random.normal(0, 1, 10000)
        shuffle_labels(y, n=1)


def test_raise_error_bad_p_value():
    """Probability values should be 0 < p < 1."""
    y = np.random.normal(0, 1, 10000)
    with pytest.raises(ValueError):
        shuffle_labels(y, p=1.5)
    with pytest.raises(ValueError):
        shuffle_labels(y, p=0.0)
