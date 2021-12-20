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

    @staticmethod
    def from_proba(proba, max_proba=0.55):
        """
        Outputs a reason array from a proba array, skipping the need for a model.

        Usage:

        ```python
        import numpy as np
        from doubtlab.reason import ProbaReason

        probas = np.array([[0.9, 0.1], [0.5, 0.5]])
        predicate = ProbaReason.from_proba(probas)
        assert np.all(predicate == np.array([0.0, 1.0]))
        ```
        """
        return (proba.max(axis=1) <= max_proba).astype(np.float16)


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
        return (rvals < self.probability).astype(np.float16)


class ShannonEntropyReason:
    """
    Assign doubt when the normalized Shannon entropy is too high, see
    [here](https://math.stackexchange.com/questions/395121/how-entropy-scales-with-sample-size)
    for a discussion.

    Arguments:
        model: scikit-learn classifier
        threshold: confidence threshold for doubt assignment
        smoothing: constant value added to probas to prevent division by zeor

    Usage:

    ```python
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    from doubtlab.ensemble import DoubtEnsemble
    from doubtlab.reason import ShannonEntropyReason

    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=1_000)
    model.fit(X, y)

    doubt = DoubtEnsemble(reason = ShannonEntropyReason(model=model))

    indices = doubt.get_indices(X, y)
    ```
    """

    def __init__(self, model, threshold=0.5, smoothing=1e-5):
        self.model = model
        self.threshold = threshold
        self.smoothing = smoothing

    def __call__(self, X, y):
        probas = self.model.predict_proba(X)
        return self.from_proba(
            probas, threshold=self.threshold, smoothing=self.smoothing
        )

    @staticmethod
    def from_proba(proba, threshold=0.5, smoothing=1e-5):
        """
        Outputs a reason array from a prediction array, skipping the need for a model.

        Usage:

        ```python
        import numpy as np
        from doubtlab.reason import ShannonEntropyReason

        probas = np.array([[0.9, 0.1, 0.0], [0.5, 0.4, 0.1]])
        predicate = ShannonEntropyReason.from_proba(probas, threshold=0.8)
        assert np.all(predicate == np.array([0.0, 1.0]))
        ```
        """
        probas = proba + smoothing
        entropies = -(probas * np.log(probas) / np.log(probas.shape[1])).sum(axis=1)
        return (entropies > threshold).astype(np.float16)


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
        preds = self.model.predict(X)
        return self.from_predict(preds, y)

    @staticmethod
    def from_predict(pred, y):
        """
        Outputs a reason array from a prediction array, skipping the need for a model.

        Usage:

        ```python
        import numpy as np
        from doubtlab.reason import WrongPredictionReason

        preds = np.array(["positive", "negative"])
        y = np.array(["positive", "neutral"])
        predicate = WrongPredictionReason.from_predict(preds, y)
        assert np.all(predicate == np.array([0.0, 1.0]))
        ```
        """
        return (pred != y).astype(np.float16)


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

    @staticmethod
    def from_proba(proba, y, classes, threshold):
        """
        Outputs a reason array from a proba array, skipping the need for a model.

        Usage:

        ```python
        import numpy as np
        from doubtlab.reason import LongConfidenceReason

        probas = np.array([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]])
        y = np.array([0, 1, 0])
        classes = np.array([0, 1])
        threshold = 0.4
        predicate = LongConfidenceReason.from_proba(probas, y, classes, threshold)
        assert np.all(predicate == np.array([0.0, 1.0, 1.0]))
        ```
        """
        values = []
        for i, proba in enumerate(proba):
            proba_dict = {classes[j]: v for j, v in enumerate(proba) if j != y[i]}
            values.append(max(proba_dict.values()))
        confidences = np.array(values)
        return (confidences > threshold).astype(np.float16)

    def __call__(self, X, y):
        probas = self.model.predict_proba(X)
        return self.from_proba(probas, y, self.model.classes_, self.threshold)


class MarginConfidenceReason:
    """
    Assign doubt when the difference between the top two most confident classes is too small.

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

    @staticmethod
    def from_proba(proba, threshold=0.2):
        """
        Outputs a reason array from a proba array, skipping the need for a model.

        Usage:

        ```python
        import numpy as np
        from doubtlab.reason import MarginConfidenceReason

        probas = np.array([[0.9, 0.1, 0.0], [0.5, 0.4, 0.1]])
        predicate = MarginConfidenceReason.from_proba(probas, threshold=0.3)
        assert np.all(predicate == np.array([0.0, 1.0]))
        ```
        """
        sorted = np.sort(proba, axis=1)
        margin = sorted[:, -1] - sorted[:, -2]
        return (margin < threshold).astype(np.float16)

    def __call__(self, X, y):
        probas = self.model.predict_proba(X)
        return self.from_proba(probas, self.threshold)


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

    doubt = DoubtEnsemble(reason = ShortConfidenceReason(model=model, threshold=0.4))

    indices = doubt.get_indices(X, y)
    ```
    """

    def __init__(self, model, threshold=0.2):
        self.model = model
        self.threshold = threshold

    @staticmethod
    def from_proba(proba, y, classes, threshold=0.2):
        """
        Outputs a reason array from a proba array, skipping the need for a model.

        Usage:

        ```python
        import numpy as np
        from doubtlab.reason import ShortConfidenceReason

        probas = np.array([[0.9, 0.1], [0.5, 0.5], [0.3, 0.7]])
        y = np.array([0, 1, 0])
        classes = np.array([0, 1])
        threshold = 0.4
        predicate = ShortConfidenceReason.from_proba(probas, y, classes, threshold)
        assert np.all(predicate == np.array([0.0, 0.0, 1.0]))
        ```
        """
        values = []
        for i, p in enumerate(proba):
            proba_dict = {classes[j]: v for j, v in enumerate(p)}
            values.append(proba_dict[y[i]])
        confidences = np.array(values)
        return (confidences < threshold).astype(np.float16)

    def __call__(self, X, y):
        probas = self.model.predict_proba(X)
        return self.from_proba(probas, y, self.model.classes_, self.threshold)


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

    @staticmethod
    def from_pred(pred1, pred2):
        """
        Outputs a reason array from two pred arrays, skipping the need for a model.

        Usage:

        ```python
        import numpy as np
        from doubtlab.reason import DisagreeReason

        pred1 = [0, 1, 2]
        pred2 = [0, 1, 1]
        predicate = DisagreeReason.from_pred(pred1, pred2)
        assert np.all(predicate == np.array([0.0, 0.0, 1.0]))
        ```
        """
        return (np.array(pred1) != np.array(pred2)).astype(np.float16)

    def __call__(self, X, y):
        pred1 = self.model1.predict(X)
        pred2 = self.model2.predict(X)
        return self.from_pred(pred1, pred2)


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
        model: scikit-learn regression model
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
        model: scikit-learn regression model
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

    @staticmethod
    def from_proba(proba, y, min_doubt=0.5, sorted_index_method="normalized_margin"):
        """
        Outputs a reason array from a proba array, skipping the need for a model.

        Usage:

        ```python
        import numpy as np
        from doubtlab.reason import CleanlabReason

        probas = np.array([[0.9, 0.1], [0.5, 0.5]])
        y = np.array([0, 1])
        predicate = CleanlabReason.from_proba(probas, y)
        ```
        """
        ordered_label_errors = get_noise_indices(y, proba, sorted_index_method)
        result = np.zeros_like(y)
        conf_arr = np.linspace(1, min_doubt, result.shape[0])
        for idx, _ in zip(ordered_label_errors, conf_arr):
            result[idx] = 1
        return result.astype(np.float16)

    def __call__(self, X, y):
        probas = self.model.predict_proba(X)
        return self.from_proba(probas, y, self.min_doubt, self.sorted_index_method)


class StandardizedErrorReason:
    """
    Assign doubt when the absolute standardized residual is too high.

    Arguments:
        model: scikit-learn regression model
        threshold: cutoff for doubt assignment

    Usage:

    ```python
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression

    from doubtlab.ensemble import DoubtEnsemble
    from doubtlab.reason import StandardizedErrorReason

    X, y = load_diabetes(return_X_y=True)
    model = LinearRegression()
    model.fit(X, y)

    doubt = DoubtEnsemble(reason = StandardizedErrorReason(model, threshold=2.))
    indices = doubt.get_indices(X, y)
    ```
    """

    def __init__(self, model, threshold=2.0):
        if threshold <= 0:
            raise ValueError("threshold value should be positive")
        self.model = model
        self.threshold = threshold

    def __call__(self, X, y):
        preds = self.model.predict(X)
        return self.from_predict(preds, y, self.threshold)

    @staticmethod
    def from_predict(pred, y, threshold):
        """
        Outputs a reason array from a prediction array, skipping the need for a model.

        Usage:
        ```python
        import numpy as np
        from doubtlab.reason import StandardizedErrorReason

        y = np.random.randn(100)
        preds = np.random.randn(100)

        predicate = StandardizedErrorReason.from_predict(preds, y)
        ```
        """
        res = y - pred
        res_std = res / np.std(res, ddof=1)
        return (np.abs(res_std) >= threshold).astype(np.float16)
