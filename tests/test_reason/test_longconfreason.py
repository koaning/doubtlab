import numpy as np
from doubtlab.reason import LongConfidenceReason


def test_longconf_proba():
    """Test from_probas on a obvious example."""
    probas = np.array([[0.9, 0.1], [0.5, 0.5]])
    y = np.array([0, 1])
    classes = np.array([0, 1])
    threshold = 0.4
    predicate = LongConfidenceReason.from_proba(
        proba=probas, y=y, classes=classes, threshold=threshold
    )
    assert np.all(predicate == np.array([0.0, 1.0]))


def test_longconf_proba_nonnumeric():
    """Test from_probas on a obvious example."""
    probas = np.array([[0.9, 0.1], [0.5, 0.5]])
    y = np.array(["a", "b"])
    classes = np.array(["a", "b"])
    threshold = 0.4
    predicate = LongConfidenceReason.from_proba(
        proba=probas, y=y, classes=classes, threshold=threshold
    )
    assert np.all(predicate == np.array([0.0, 1.0]))


def test_longconf_proba_bigger_nonnumeric():
    """Test from_probas on a bigger obvious example."""
    probas = np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9], [0.2, 0.8]])
    y = np.array(["a", "b", "a", "b"])
    classes = np.array(["a", "b"])
    threshold = 0.4
    predicate = LongConfidenceReason.from_proba(
        proba=probas, y=y, classes=classes, threshold=threshold
    )
    assert np.all(predicate == np.array([0.0, 1.0, 1.0, 0.0]))
