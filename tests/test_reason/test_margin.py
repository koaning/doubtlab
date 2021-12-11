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
    reason = MarginConfidenceReason.from_probas(probas)
    assert all([r == 0.0 for r in reason])
