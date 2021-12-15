import numpy as np
from doubtlab.reason import ShannonEntropyReason


def test_short_conf_probas():
    """
    Test `from_proba` on an obvious example.
    """
    probas = np.array([[0.9, 0.1, 0.0], [0.5, 0.4, 0.1]])
    predicate = ShannonEntropyReason.from_proba(probas, threshold=0.8)
    assert np.all(predicate == np.array([0.0, 1.0]))
