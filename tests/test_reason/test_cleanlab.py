import numpy as np
from doubtlab.reason import CleanlabReason


def test_longconf_proba():
    """Test from_probas on a obvious example."""
    probas = np.array([[0.9, 0.1], [0.5, 0.5]])
    y = np.array([0, 1])
    predicate = CleanlabReason.from_proba(proba=probas, y=y)
    assert predicate.dtype == np.float16
