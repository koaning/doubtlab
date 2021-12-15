import numpy as np
from doubtlab.reason import ShortConfidenceReason


def test_short_conf_probas():
    """
    Test `from_probas` on an obvious example.
    """
    probas = np.array([[0.9, 0.1], [0.5, 0.5]])
    y = np.array([0, 1])
    classes = np.array([0, 1])
    threshold = 0.6
    predicate = ShortConfidenceReason.from_probas(probas, y, classes, threshold)
    assert np.all(predicate == np.array([0.0, 1.0]))
