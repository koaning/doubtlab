import numpy as np
from doubtlab.reason import StandardizedErrorReason


def test_from_predict():
    """Test `from_predict` on an obvious examples"""
    y = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    preds = np.array(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 10.0]
    ) + np.random.choice([-0.05, 0, 0.05], 10)
    predicate = StandardizedErrorReason.from_predict(pred=preds, y=y, threshold=3.0)
    assert np.all(
        predicate == np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    )


def test_from_predict_no_reason():
    """Test `from_predict` on an obvious examples"""
    y = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    preds = y + np.random.choice([-0.05, 0, 0.05], 10)
    predicate = StandardizedErrorReason.from_predict(pred=preds, y=y, threshold=3.0)
    assert np.all(
        predicate == np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
