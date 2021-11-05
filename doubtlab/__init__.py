try:
    from importlib import metadata
except ImportError:  # for Python<3.8
    import importlib_metadata as metadata

import numpy as np
import pandas as pd


__title__ = __name__
__version__ = metadata.version(__title__)


class DoubtLab:
    """
    A pipeline to find bad labels.

    Arguments:
        reasons: kwargs with (name, reason)-pairs

    Usage:

    ```python
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    from doubtlab import DoubtLab
    from doubtlab.reason import ProbaReason, WrongPredictionReason

    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=1_000)
    model.fit(X, y)

    reasons = {
        "proba": ProbaReason(model=model),
        "wrong_pred": WrongPredictionReason(model=model),
    }

    doubt = DoubtLab(**reasons)
    ```
    """

    def __init__(self, **reasons):
        self.reasons = reasons

    def get_predicates(self, X, y=None):
        """
        Returns a sorted dataframe that shows the reasoning behind the sorting.

        Arguments:
            X: the `X` data to be processed
            y: the `y` data to be processed

        Usage:

        ```python
        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression

        from doubtlab import DoubtLab
        from doubtlab.reason import ProbaReason, WrongPredictionReason

        X, y = load_iris(return_X_y=True)
        model = LogisticRegression(max_iter=1_000)
        model.fit(X, y)

        reasons = {
            "proba": ProbaReason(model=model),
            "wrong_pred": WrongPredictionReason(model=model),
        }

        doubt = DoubtLab(**reasons)

        predicates = doubt.get_predicates(X, y)
        ```
        """
        df = pd.DataFrame({"i": range(len(X))})
        for name, func in self.reasons.items():
            df[f"predicate_{name}"] = func(X, y)
        predicates = [c for c in df.columns if "predicate" in c]
        df["s"] = df.drop(columns=["i"]).sum(axis=1)
        return df.sort_values(predicates, ascending=False).drop(columns=["i", "s"])

    def get_indices(self, X, y=None):
        """
        Calculates indices worth checking again.

        Arguments:
            X: the `X` data to be processed
            y: the `y` data to be processed

        Usage:

        ```python
        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression

        from doubtlab import DoubtLab
        from doubtlab.reason import ProbaReason, WrongPredictionReason

        X, y = load_iris(return_X_y=True)
        model = LogisticRegression(max_iter=1_000)
        model.fit(X, y)

        reasons = {
            "proba": ProbaReason(model=model),
            "wrong_pred": WrongPredictionReason(model=model),
        }

        doubt = DoubtLab(**reasons)

        indices = doubt.get_indices(X, y)
        ```
        """
        df = self.get_predicates(X, y)
        predicates = [
            c for c in df.columns if isinstance(c, str) and ("predicate" in c)
        ]
        return np.array(
            [int(i) for i in df.loc[lambda d: d[predicates].sum(axis=1) > 0].index]
        )

    def get_candidates(self, X, y=None):
        """
        Returns the candidates worth checking again.

        Arguments:
            X: the `X` data to be processed
            y: the `y` data to be processed

        Usage:

        ```python
        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression

        from doubtlab import DoubtLab
        from doubtlab.reason import ProbaReason, WrongPredictionReason

        X, y = load_iris(return_X_y=True)
        model = LogisticRegression(max_iter=1_000)
        model.fit(X, y)

        reasons = {
            "proba": ProbaReason(model=model),
            "wrong_pred": WrongPredictionReason(model=model),
        }

        doubt = DoubtLab(**reasons)

        predicates = doubt.get_predicates(X, y)
        ```
        """
        indices = self.get_indices(X, y)
        if isinstance(X, np.ndarray):
            return X[indices], y[indices]
        if isinstance(X, pd.DataFrame):
            return X.iloc[indices], y[indices]
        if isinstance(X, list):
            return [x for i, x in enumerate(X) if i in indices], [
                y for i, y in enumerate(y) if i in indices
            ]
