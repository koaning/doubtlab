import numpy as np
from doubtlab.reason import DisagreeReason


def test_short_conf_probas():
    """
    Test `from_probas` on an obvious example.
    """
    pred1 = [0, 1, 2]
    pred2 = [0, 1, 1]
    predicate = DisagreeReason.from_pred(pred1, pred2)
    assert np.all(predicate == np.array([0.0, 0.0, 1.0]))
