import numpy as np


class ProbaReason:
    """
    Assign doubt based on low proba-confidence values from a scikit-learn model.

    Arguments:
        model: scikit-learn classifier
        max_proba: maximum probability threshold for doubt assignment

    Usage:

    ```python
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    from doubtlab.ensemble import DoubtEnsemble
    from doubtlab.reason import ProbaReason

    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=1_000)
    model.fit(X, y)

    doubt = DoubtEnsemble(reason = ProbaReason(model, max_proba=0.55))

    indices = doubt.get_indices(X, y)
    ```
    """

    def __init__(self, model, max_proba=0.55):
        self.model = model
        self.max_proba = max_proba

    def __call__(self, X, y=None):
        result = self.model.predict_proba(X).max(axis=1) <= self.max_proba
        return result.astype(np.float16)


class RandomReason:
    """
    Assign doubt based on a random value.

    Arguments:
        probability: probability of assigning a doubt
        random_seed: seed for random number generator

    Usage:

    ```python
    from sklearn.datasets import load_iris

    from doubtlab.ensemble import DoubtEnsemble
    from doubtlab.reason import RandomReason

    X, y = load_iris(return_X_y=True)

    doubt = DoubtEnsemble(reason = RandomReason(probability=0.05, random_seed=42))

    indices = doubt.get_indices(X, y)
    ```
    """

    def __init__(self, probability=0.01, random_seed=42):
        self.probability = probability
        self.random_seed = random_seed

    def __call__(self, X, y=None):
        np.random.seed(self.random_seed)
        rvals = np.random.random(size=len(X))
        return np.where(rvals < self.probability, rvals, 0)


class WrongPredictionReason:
    """
    Assign doubt when the model prediction doesn't match the label.

    Arguments:
        model: sci-kit learn classifier

    Usage:

    ```python
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    from doubtlab.ensemble import DoubtEnsemble
    from doubtlab.reason import WrongPredictionReason

    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=1_000)
    model.fit(X, y)

    doubt = DoubtEnsemble(reason = WrongPredictionReason(model=model))

    indices = doubt.get_indices(X, y)
    ```
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, X, y):
        return (self.model.predict(X) != y).astype(np.float16)


class LongConfidenceReason:
    """
    Assign doubt when a wrong class gains too much confidence.

    Arguments:
        model: sci-kit learn classifier
        threshold: confidence threshold for doubt assignment

    Usage:

    ```python
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    from doubtlab.ensemble import DoubtEnsemble
    from doubtlab.reason import LongConfidenceReason

    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=1_000)
    model.fit(X, y)

    doubt = DoubtEnsemble(reason = LongConfidenceReason(model=model))

    indices = doubt.get_indices(X, y)
    ```
    """

    def __init__(self, model, threshold=0.2):
        self.model = model
        self.threshold = threshold

    def _max_bad_class_confidence(self, X, y):
        probas = self.model.predict_proba(X)
        values = []
        for i, proba in enumerate(probas):
            proba_dict = {
                self.model.classes_[j]: v for j, v in enumerate(proba) if j != y[i]
            }
            values.append(max(proba_dict.values()))
        return np.array(values)

    def __call__(self, X, y):
        confidences = self._max_bad_class_confidence(X, y)
        return np.where(confidences > self.threshold, confidences, 0)


class ShortConfidenceReason:
    """
    Assign doubt when the correct class gains too little confidence.

    Arguments:
        model: sci-kit learn classifier
        threshold: confidence threshold for doubt assignment

    Usage:

    ```python
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    from doubtlab.ensemble import DoubtEnsemble
    from doubtlab.reason import ShortConfidenceReason

    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=1_000)
    model.fit(X, y)

    doubt = DoubtEnsemble(reason = ShortConfidenceReason(model=model))

    indices = doubt.get_indices(X, y)
    ```
    """

    def __init__(self, model, threshold=0.2):
        self.model = model
        self.threshold = threshold

    def _correct_class_confidence(self, X, y):
        """
        Gives the predicted confidence (or proba) associated
        with the correct label `y` from a given model.
        """
        probas = self.model.predict_proba(X)
        values = []
        for i, proba in enumerate(probas):
            proba_dict = {self.model.classes_[j]: v for j, v in enumerate(proba)}
            values.append(proba_dict[y[i]])
        return np.array(values)

    def __call__(self, X, y):
        confidences = self._correct_class_confidence(X, y)
        return np.where(confidences < self.threshold, 1 - confidences, 0)


class DisagreeReason:
    """
    Assign doubt when two scikit-learn models disagree on a prediction.

    Arguments:
        model1: sci-kit learn classifier
        model2: a different sci-kit learn classifier

    Usage:

    ```python
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier

    from doubtlab.ensemble import DoubtEnsemble
    from doubtlab.reason import DisagreeReason

    X, y = load_iris(return_X_y=True)
    model1 = LogisticRegression(max_iter=1_000)
    model2 = KNeighborsClassifier()
    model1.fit(X, y)
    model2.fit(X, y)

    doubt = DoubtEnsemble(reason = DisagreeReason(model1, model2))

    indices = doubt.get_indices(X, y)
    ```
    """

    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def __call__(self, X, y):
        result = self.model1.predict(X) != self.model2.predict(X)
        return result.astype(np.float16)


class OutlierReason:
    """
    Assign doubt when a scikit-learn outlier model detects an outlier.

    Arguments:
        model: scikit-learn outlier model

    Usage:

    ```python
    from sklearn.datasets import load_iris
    from sklearn.ensemble import IsolationForest

    from doubtlab.ensemble import DoubtEnsemble
    from doubtlab.reason import OutlierReason

    X, y = load_iris(return_X_y=True)
    model = IsolationForest()
    model.fit(X)

    doubt = DoubtEnsemble(reason = OutlierReason(model))

    indices = doubt.get_indices(X, y)
    ```
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, X, y):
        return (self.model.predict(X) == -1).astype(np.float16)


class RegressionGapReason:
    """
    Assign doubt when a label differs too much from a scikit-learn regression model.

    Arguments:
        model: scikit-learn outlier model

    Usage:

    ```python
    from sklearn.datasets import load_iris
    from sklearn.ensemble import IsolationForest

    from doubtlab.ensemble import DoubtEnsemble
    from doubtlab.reason import RegressionGapReason

    X, y = load_iris(return_X_y=True)
    model = IsolationForest()
    model.fit(X)

    doubt = DoubtEnsemble(reason = RegressionGapReason(model))

    indices = doubt.get_indices(X, y)
    ```
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, X, y):
        return (self.model.predict(X) == -1).astype(np.float16)
