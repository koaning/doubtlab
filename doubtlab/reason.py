import numpy as np
from cleanlab.pruning import get_noise_indices


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
        model: scikit-learn classifier

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
        model: scikit-learn classifier
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


class MarginConfidenceReason:
    """
    Assign doubt when a the difference between the top two most confident classes is too large.

    Throws an error when there are only two classes.

    Arguments:
        model: scikit-learn classifier
        threshold: confidence threshold for doubt assignment

    Usage:

    ```python
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    from doubtlab.ensemble import DoubtEnsemble
    from doubtlab.reason import MarginConfidenceReason

    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=1_000)
    model.fit(X, y)

    doubt = DoubtEnsemble(reason = MarginConfidenceReason(model=model))

    indices = doubt.get_indices(X, y)
    ```
    """

    def __init__(self, model, threshold=0.2):
        self.model = model
        self.threshold = threshold

    def _calc_margin(self, probas):
        sorted = np.sort(probas, axis=1)
        return sorted[:, -1] - sorted[:, -2]

    def __call__(self, X, y):
        probas = self.model.predict_proba(X)
        margin = self._calc_margin(probas)
        return np.where(margin > self.threshold, margin, 0)


class ShortConfidenceReason:
    """
    Assign doubt when the correct class gains too little confidence.

    Arguments:
        model: scikit-learn classifier
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
        model1: scikit-learn classifier
        model2: a different scikit-learn classifier

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


class AbsoluteDifferenceReason:
    """
    Assign doubt when the absolute difference between label and regression is too large.

    Arguments:
        model: scikit-learn outlier model
        threshold: cutoff for doubt assignment

    Usage:

    ```python
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression

    from doubtlab.ensemble import DoubtEnsemble
    from doubtlab.reason import AbsoluteDifferenceReason

    X, y = load_diabetes(return_X_y=True)
    model = LinearRegression()
    model.fit(X, y)

    doubt = DoubtEnsemble(reason = AbsoluteDifferenceReason(model, threshold=100))

    indices = doubt.get_indices(X, y)
    ```
    """

    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def __call__(self, X, y):
        difference = np.abs(self.model.predict(X) - y)
        return (difference >= self.threshold).astype(np.float16)


class RelativeDifferenceReason:
    """
    Assign doubt when the relative difference between label and regression is too large.

    Arguments:
        model: scikit-learn outlier model
        threshold: cutoff for doubt assignment

    Usage:

    ```python
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression

    from doubtlab.ensemble import DoubtEnsemble
    from doubtlab.reason import RelativeDifferenceReason

    X, y = load_diabetes(return_X_y=True)
    model = LinearRegression()
    model.fit(X, y)

    doubt = DoubtEnsemble(reason = RelativeDifferenceReason(model, threshold=0.5))

    indices = doubt.get_indices(X, y)
    ```
    """

    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def __call__(self, X, y):
        difference = np.abs(self.model.predict(X) - y) / y
        return (difference >= self.threshold).astype(np.float16)


class CleanlabReason:
    """
    Assign doubt when using the cleanlab heuristic.

    Arguments:
        model: scikit-learn outlier model
        sorted_index_method: method used by cleanlab for sorting indices
        min_doubt: the minimum doubt output value used for sorting by the ensemble

    Usage:

    ```python
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier

    from doubtlab.ensemble import DoubtEnsemble
    from doubtlab.reason import CleanlabReason

    X, y = load_iris(return_X_y=True)
    model = LogisticRegression()
    model.fit(X, y)

    doubt = DoubtEnsemble(reason = CleanlabReason(model))

    indices = doubt.get_indices(X, y)
    ```
    """

    def __init__(self, model, sorted_index_method="normalized_margin", min_doubt=0.5):
        self.model = model
        self.sorted_index_method = sorted_index_method
        self.min_doubt = min_doubt

    def __call__(self, X, y):
        probas = self.model.predict_proba(X)
        ordered_label_errors = get_noise_indices(y, probas, self.sorted_index_method)
        result = np.zeros_like(y)
        conf_arr = np.linspace(1, self.min_doubt, result.shape[0])
        for idx, conf in zip(ordered_label_errors, conf_arr):
            result[idx] = conf
        return result
