The goal of this document is to explain how the library works on a high-level.

## Datasets and Models

You can use doubtlab to check your own datasets for bad labels. Many of the
methods that we provide are based on the interaction between a dataset and a
model trained on that dataset. For example;


```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=1_000)
model.fit(X, y)
```

This examples shows a logistic regression model trained on the `load_iris` dataset.
The `model` is able to make predictions on the dataset and it's also able to
output a confidence score via `model.predict_proba(X)`.

You could wonder. What might it mean if the confidence values are low? What might
it mean if our model cannot make an accurate prediction on a datapoint that it's trained on?
In both of these cases, it could be that nothing is wrong. But you could argue that
these datapoints may be worth double-checking.

## Pipeline of Doubt Reasons

The doubtlab library allows you to define "reasons" to doubt the validity of a datapoint.
The code below shows you how to build an ensemble of the two aforementioned reasons.

```python
from doubtlab.ensemble import DoubtEnsemble
from doubtlab.reason import ProbaReason, WrongPredictionReason

# Define the reasons with a name.
reasons = {
    "proba": ProbaReason(model=model, threshold=0.4),
    "wrong_pred": WrongPredictionReason(model=model),
}

# Put all the reasons into an ensemble
doubt = DoubtEnsemble(**reasons)
```

## What does this Ensemble do?


### Internal Details

## Retreiving Examples to Check

There are multiple ways of retreiving the examples to check
from the doubt pipeline.

### Get Indices

```python
# Get the ordered indices of examples worth checking again
indices = doubt.get_indices(X, y)
```

### Get Predicates

```python
# Get the predicates, or reasoning, behind the order
predicates = doubt.get_predicates(X, y)
```

## Why this matters
