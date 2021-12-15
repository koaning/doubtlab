import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from doubtlab.reason import MarginConfidenceReason


def test_margin_confidence_margin():
    """Ensures margin is calculated correctly."""
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=1_000)
    model.fit(X, y)

    probas = np.eye(3)
    reason = MarginConfidenceReason.from_proba(proba=probas)
    assert all([r == 0.0 for r in reason])


def test_margin_simple_example():
    """Test on a obvious example."""
    probas = np.array([[0.9, 0.1, 0.0], [0.5, 0.4, 0.1]])
    predicate = MarginConfidenceReason.from_proba(proba=probas, threshold=0.3)
    assert np.all(predicate == np.array([0.0, 1.0]))
