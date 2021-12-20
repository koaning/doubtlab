import pytest
import itertools as it

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_iris, load_breast_cancer, load_wine

from doubtlab.reason import (
    ProbaReason,
    OutlierReason,
    DisagreeReason,
    MarginConfidenceReason,
    LongConfidenceReason,
    ShortConfidenceReason,
    WrongPredictionReason,
    AbsoluteDifferenceReason,
    RelativeDifferenceReason,
    CleanlabReason,
    StandardizedErrorReason,
)

clf_reasons = [
    ProbaReason,
    LongConfidenceReason,
    ShortConfidenceReason,
    MarginConfidenceReason,
    WrongPredictionReason,
    CleanlabReason,
]

regr_reasons = [
    AbsoluteDifferenceReason,
    RelativeDifferenceReason,
    StandardizedErrorReason,
]

clf_datasets = [
    load_iris(return_X_y=True),
    load_wine(return_X_y=True),
    load_breast_cancer(return_X_y=True),
]


@pytest.mark.parametrize("reason, dataset", it.product(clf_reasons, clf_datasets))
def test_clf_reason_between_0_and_1(reason, dataset):
    """Simple model based models should have confidence in [0, 1]"""
    X, y = dataset
    model = LogisticRegression(max_iter=1_000)
    model.fit(X, y)
    assert np.all(reason(model=model)(X, y) >= 0)
    assert np.all(reason(model=model)(X, y) <= 1)


@pytest.mark.parametrize("dataset", clf_datasets)
def test_clf_disagree_reason_between_0_and_1(dataset):
    """Disagreement based models should have confidence in [0, 1]"""
    X, y = dataset
    model1 = LogisticRegression(max_iter=1_000, C=0.001)
    model2 = LogisticRegression(max_iter=1_000, C=1000)
    model1.fit(X, y)
    model2.fit(X, y)
    reason = DisagreeReason(model1, model2)
    assert np.all(reason(X, y) >= 0)
    assert np.all(reason(X, y) <= 1)


@pytest.mark.parametrize("dataset", clf_datasets)
def test_clf_outlier_between_0_and_1(dataset):
    """Outlier based models should have confidence in [0, 1]"""
    X, y = dataset
    model = IsolationForest()
    model.fit(X, y)
    reason = OutlierReason(model)
    assert np.all(reason(X, y) >= 0)
    assert np.all(reason(X, y) <= 1)
