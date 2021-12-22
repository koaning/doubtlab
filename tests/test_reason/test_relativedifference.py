import pytest

import numpy as np
from doubtlab.reason import RelativeDifferenceReason


@pytest.mark.parametrize("t, s", [(0.05, 4), (0.2, 3), (0.4, 2), (0.6, 1)])
def test_from_predict(t, s):
    """Test `from_predict` on an obvious examples"""
    y = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    preds = np.array([1.0, 1.1, 1.3, 1.5, 1.7])

    predicate = RelativeDifferenceReason.from_predict(pred=preds, y=y, threshold=t)
    assert np.sum(predicate) == s


def test_zero_error():
    """Ensure error is throw when `y=0`"""
    y = np.array([0.0])
    preds = np.array([1.0])
    with pytest.raises(ValueError):
        RelativeDifferenceReason.from_predict(pred=preds, y=y, threshold=0.1)
