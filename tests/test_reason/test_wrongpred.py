import numpy as np
from doubtlab.reason import WrongPredictionReason


def test_from_predict():
    """Test `from_predict` on an obvious example"""
    preds = np.array(["positive", "negative"])
    y = np.array(["positive", "neutral"])
    predicate = WrongPredictionReason.from_predict(preds, y)
    assert np.all(predicate == np.array([0.0, 1.0]))
