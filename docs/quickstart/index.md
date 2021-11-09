The goal of this document is to explain how the library works on a high-level.
Another tutorial will show a better example.

## Simple Model

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=1_000)
model.fit(X, y)
```

## Pipeline of Doubt Reasons

```python
from doubtlab import DoubtLab
from doubtlab.reason import ProbaReason, WrongPredictionReason

reasons = {
    "proba": ProbaReason(model=model),
    "wrong_pred": WrongPredictionReason(model=model),
}

doubt = DoubtLab(**reasons)
```

## What does this Pipeline do?

### Internal Details

## Retreiving Examples to Check

There are multiple ways of retreiving the examples to check
from the doubt pipeline.

### Get Indices

```python
# Get the predicates, or reasoning, behind the order
predicates = doubt.get_predicates(X, y)
```

### Get Predicates

```python
# Get the ordered indices of examples worth checking again
indices = doubt.get_indices(X, y)
```

### Get Candidates

```python
# Get the (X, y) candidates worth checking again
X_check, y_check = doubt.get_candidates(X, y)
```
