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
    "proba": ProbaReason(model=model, max_proba=0.55),
    "wrong_pred": WrongPredictionReason(model=model),
}

# Put all the reasons into an ensemble
doubt = DoubtEnsemble(**reasons)
```

This ensemble represents a pipeline of reasons to doubt the validity of a label.

### Internal Details

A `DoubtEnsemble`, technically, is just an ensemble of callables. You could
also choose to use lambda functions to define a reason for doubt. The example
below shows an example of a lambda function that's equivalent to what `WrongPredictionReason`
would do.

```python
DoubtEnsemble(
    wrong_pred=lambda X, y: (model.predict(X) != y).astype(float16)
)
```

When it's time to infer doubt, the `DoubtEnsemble` will call each callable reason in order,
passing `X`, `y` and listening for an array that contains "doubt-scores". These scores are just
numbers, but they follow a few rules.

- when there is no doubt, the score should be zero
- the maximum doubt that a reason can emit is one
- the higher the doubt-score, the more likely doubt should be



## Retreiving Examples to Check

There are multiple ways of retreiving the examples to check
from the doubt pipeline.

### Get Indices

You could simply use the `DoubtEnsemble.get_indices` method to get the indices of
the original data that are in doubt.

```python
# Get the ordered indices of examples worth checking again
indices = doubt.get_indices(X, y)
```

In this case, you'd get an array with 6 elements.

```
array([ 77, 106, 126, 133,  83, 119,  70])
```

You can inspect the associated rows/labels of the examples via:

```python
X[indices], y[indices]
```

### Get Predicates

While the indices are useful they don't tell you much about how the
ordering took place. If you'd like to see more details, you can also
retreive a dataframe with predicates that explain which rows triggered
which reasons.

```python
# Get the predicates, or reasoning, behind the order
predicates = doubt.get_predicates(X, y)
```

The `predicates` dataframe contains a column for each reason. The index
refers to the row number in the original dataset. Let's check the top 10 rows.

```python
predicates.head(10)
```

|     |   predicate_proba |   predicate_wrong_pred |
|----:|------------------:|-----------------------:|
|  77 |                 1 |                      1 |
| 106 |                 1 |                      1 |
| 126 |                 1 |                      0 |
| 133 |                 1 |                      0 |
|  83 |                 0 |                      1 |
| 119 |                 1 |                      0 |
|  70 |                 0 |                      1 |
| 105 |                 0 |                      0 |
| 107 |                 0 |                      0 |
| 104 |                 0 |                      0 |

There's a few things to observe here.

- The ensemble assumes that overlap between reasons matter is a reason to give a row priority, moving it up in the dataframe.
- The `.get_indices` method tells you *what* deserves checking and only returns candidates worth checking. The `.get_predicates` method tries to explain *why* these rows deserve to be checked and therefore returns a dataframe with a row for each row in `X`.
- The index of the predicates dataframe refers to rows in our original `X`, `y` arrays.

## Why do this exercise?

## Next Steps

This library contains many more "reasons" that you can use.

### Classification Reasons

### Regression Reasons
