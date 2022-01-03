import pytest

import numpy as np
from doubtlab.reason import AbsoluteDifferenceReason


@pytest.mark.parametrize("t, s", [(0.1, 4), (0.2, 3), (0.3, 2)])
def test_from_predict(t, s):
    """Test `from_predict` on an obvious examples"""
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    preds = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

    predicate = AbsoluteDifferenceReason.from_predict(pred=preds, y=y, threshold=t)
    assert np.sum(predicate) == s
