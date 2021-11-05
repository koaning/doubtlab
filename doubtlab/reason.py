# TODO: many of these methods appreciate a refit parameter, which will
# control if we get to retrain a model on this dataset

import numpy as np


class ProbaReason:
    """
    Assign doubt based on low proba-confidence values from a scikit-learn model.
    """

    def __init__(self, model, max_proba=0.55):
        self.model = model
        self.max_proba = max_proba

    def __call__(self, X, y=None):
        return (self.model.predict_proba(X).max(axis=1) <= self.max_proba).astype(
            np.float16
        )


class RandomReason:
    """
    Assign doubt based on a random value.
    """

    def __init__(self, probability=0.01, seed=42):
        self.probability = probability
        self.seed = seed

    def __call__(self, X, y=None):
        np.random.seed(self.seed)
        rvals = np.random.random(size=len(X))
        return np.where(rvals < self.probability, rvals, 0)


class WrongPredictionReason:
    """
    Assign doubt when the model prediction doesn't match the label.
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, X, y):
        return (self.model.predict(X) != y).astype(np.float16)


class LongConfidenceReason:
    """
    Assign doubt when a wrong class gains too much confidence.
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, X, y):
        max_proba = self.model.predict_proba(X).max(axis=1)
        return np.where(self.model.predict(X) != y, max_proba, 0)


class ShortConfidenceReason:
    """
    Assign doubt when the correct class gains too little confidence.
    """

    def __init__(self, model, threshold=0.2):
        self.model = model
        self.threshold = threshold

    def correct_class_confidence(self, X, y):
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
        confidences = self.correct_class_confidence(X, y)
        return np.where(confidences < self.threshold, 1 - confidences, 0)


class DisagreeReason:
    """
    Assign doubt when two scikit-learn models disagree on a prediction.
    """

    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def __call__(self, X, y):
        return self.model1.predict(X) != self.model2.predic(X)
