import pytest
import numpy as np
from doubtlab.reason import WrongPredictionReason


def test_from_predict():
    """Test `from_predict` on an obvious example"""
    preds = np.array(["positive", "negative"])
    y = np.array(["positive", "neutral"])
    predicate = WrongPredictionReason.from_predict(pred=preds, y=y)
    assert np.all(predicate == np.array([0.0, 1.0]))


def test_from_predict_fp():
    """Test `from_predict` on an obvious fp example"""
    preds = np.array([0, 0, 1, 0])
    y = np.array([0, 0, 1, 1])
    predicate = WrongPredictionReason.from_predict(pred=preds, y=y, method="fp")
    assert np.all(predicate == np.array([0.0, 0.0, 0.0, 1.0]))


def test_from_predict_fn():
    """Test `from_predict` on an obvious fn example"""
    preds = np.array([0, 0, 1, 1])
    y = np.array([0, 0, 1, 0])
    predicate = WrongPredictionReason.from_predict(pred=preds, y=y, method="fn")
    assert np.all(predicate == np.array([0.0, 0.0, 0.0, 1.0]))


def test_value_error():
    """Test `from_predict` on an obvious value-error example"""
    preds = np.array([0, 1, 2])
    y = np.array([0, 1, 0])
    with pytest.raises(ValueError):
        WrongPredictionReason.from_predict(pred=preds, y=y, method="fn")
    with pytest.raises(ValueError):
        WrongPredictionReason.from_predict(pred=preds, y=y, method="fp")

    preds = np.array([0, 1, 0])
    y = np.array([0, 1, 2])
    with pytest.raises(ValueError):
        WrongPredictionReason.from_predict(pred=preds, y=y, method="fn")
    with pytest.raises(ValueError):
        WrongPredictionReason.from_predict(pred=preds, y=y, method="fp")
