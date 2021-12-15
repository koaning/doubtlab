import numpy as np
from doubtlab.reason import ProbaReason


def test_from_proba():
    """Ensure internal `from_proba` method handles obvious example"""
    probas = np.array([[0.9, 0.1], [0.5, 0.5]])
    predicate = ProbaReason.from_proba(probas, max_proba=0.5)
    assert np.all(predicate == np.array([0.0, 1.0]))


def test_from_proba_max_proba():
    """Ensure internal `from_proba` method handles another obvious example"""
    probas = np.array([[0.9, 0.1], [0.5, 0.5]])
    predicate = ProbaReason.from_proba(probas, max_proba=0.3)
    assert np.all(predicate == np.array([0.0, 0.0]))
