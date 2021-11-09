from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from doubtlab.ensemble import DoubtEnsemble
from doubtlab.reason import ProbaReason, WrongPredictionReason


def test_smoke():
    """Very basic smoke test on the iris dataset."""
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=1_000)
    model.fit(X, y)

    reasons = {
        "proba": ProbaReason(model=model),
        "wrong_pred": WrongPredictionReason(model=model),
    }

    doubt = DoubtEnsemble(**reasons)

    predicates = doubt.get_predicates(X, y)
    assert predicates.shape[0] > 0
    indices = doubt.get_indices(X, y)
    assert indices.shape[0] > 0
