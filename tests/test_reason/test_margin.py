import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from doubtlab.reason import MarginConfidenceReason


def test_margin_confidence_margin():
    """Ensures margin is calculated correctly."""
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=1_000)
    model.fit(X, y)

    reason = MarginConfidenceReason(model=model)
    probas = np.eye(3)
    margin = reason._calc_margin(probas=probas)
    assert np.all(np.isclose(margin, np.ones(3)))
