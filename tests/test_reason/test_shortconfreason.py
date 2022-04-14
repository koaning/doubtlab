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
    predicate = ShortConfidenceReason.from_proba(
        proba=probas, y=y, classes=classes, threshold=threshold
    )
    assert np.all(predicate == np.array([0.0, 1.0]))


def test_short_conf_probas_bigger():
    """
    Test `from_probas` on an bigger obvious example.
    """
    probas = np.array([[0.5, 0.5, 0.0], [0.3, 0.3, 0.4], [0.65, 0.15, 0.3]])
    y = np.array([1, 2, 0])
    classes = np.array([0, 1, 2])
    threshold = 0.6
    predicate = ShortConfidenceReason.from_proba(
        proba=probas, y=y, classes=classes, threshold=threshold
    )
    assert np.all(predicate == np.array([1.0, 1.0, 0.0]))


def test_short_conf_non_numeric():
    """
    Test `from_probas` on an obvious example.
    """
    probas = np.array([[0.9, 0.1], [0.5, 0.5]])
    y = np.array(["a", "b"])
    classes = np.array(["a", "b"])
    threshold = 0.6
    predicate = ShortConfidenceReason.from_proba(
        proba=probas, y=y, classes=classes, threshold=threshold
    )
    assert np.all(predicate == np.array([0.0, 1.0]))
