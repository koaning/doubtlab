import numpy as np
import pandas as pd


class DoubtEnsemble:
    """
    A pipeline to find bad labels.

    Arguments:
        reasons: kwargs with (name, reason)-pairs

    Usage:

    ```python
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    from doubtlab.ensemble import DoubtEnsemble
    from doubtlab.reason import ProbaReason, WrongPredictionReason

    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=1_000)
    model.fit(X, y)

    reasons = {
        "proba": ProbaReason(model=model),
        "wrong_pred": WrongPredictionReason(model=model),
    }

    doubt = DoubtEnsemble(**reasons)
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

        from doubtlab.ensemble import DoubtEnsemble
        from doubtlab.reason import ProbaReason, WrongPredictionReason

        X, y = load_iris(return_X_y=True)
        model = LogisticRegression(max_iter=1_000)
        model.fit(X, y)

        reasons = {
            "proba": ProbaReason(model=model),
            "wrong_pred": WrongPredictionReason(model=model),
        }

        doubt = DoubtEnsemble(**reasons)

        predicates = doubt.get_predicates(X, y)
        ```
        """
        df = pd.DataFrame(
            {f"predicate_{name}": func(X, y) for name, func in self.reasons.items()}
        )
        sorted_index = df.sum(axis=1).sort_values(ascending=False).index
        return df.reindex(sorted_index)

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

        from doubtlab.ensemble import DoubtEnsemble
        from doubtlab.reason import ProbaReason, WrongPredictionReason

        X, y = load_iris(return_X_y=True)
        model = LogisticRegression(max_iter=1_000)
        model.fit(X, y)

        reasons = {
            "proba": ProbaReason(model=model),
            "wrong_pred": WrongPredictionReason(model=model),
        }

        doubt = DoubtEnsemble(**reasons)

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
