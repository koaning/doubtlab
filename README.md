<img src="docs/doubt.png" width=125 height=125 align="right">

# doubtlab

> A lab for bad labels.

This repository contains general tricks that may help you find bad, or noisy, labels in your dataset. The hope is that this repository makes it easier for folks to quickly check their own datasets before they invest too much time and compute on gridsearch.

## Install

You can install the tool via pip.

```
python -m pip install doubtlab
```

## Quickstart

Doubtlab allows you to define "reasons" for a row of data to deserve another look. These reasons can form a pipeline which can be used to retreive a sorted list of examples worth checking again.

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from doubtlab.ensemble import DoubtEnsemble
from doubtlab.reason import ProbaReason, WrongPredictionReason

# Let's say we have some dataset/model already
X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=1_000)
model.fit(X, y)

# Next we can add reasons for doubt. In this case we're saying
# that examples deserve another look if the associated proba values
# are low or if the model output doesn't match the associated label.
reasons = {
    'proba': ProbaReason(model=model),
    'wrong_pred': WrongPredictionReason(model=model)
}

# Pass these reasons to a doubtlab instance.
doubt = DoubtEnsemble(**reasons)

# Get the ordered indices of examples worth checking again
indices = doubt.get_indices(X, y)
# Get dataframe with "reason"-ing behind the sorting
predicates = doubt.get_predicates(X, y)
```

## Features

The library implemented many "reaons" for doubt.

### General Reasons

- `RandomReason`: assign doubt randomly, just for sure
- `OutlierReason`: assign doubt when the model declares a row an outlier

### Classification Reasons

- `ProbaReason`: assign doubt when a models' confidence-values are low
- `LongConfidenceReason`: assign doubt when a wrong class gains too much confidence
- `ShortConfidenceReason`: assign doubt when the correct class gains too little confidence
- `DisagreeReason`: assign doubt when two models disagree on a prediction
- `OutlierReason`: assign doubt when the model declares a row an outlier
- `CleanLabReason`: assign doubt according to [cleanlab](https://github.com/cleanlab/cleanlab)

### Regression Reasons

- `AbsoluteDifferenceReason`: assign doubt when the absolute difference is too high
- `RelativeDifferenceReason`: assign doubt when the relative difference is too high

## Related Projects

- The [cleanlab](https://github.com/cleanlab/cleanlab) project was an inspiration for this one. They have a great heuristic for bad label detection but I wanted to have a library that implements many. Be sure to check out their work on the [labelerrors.com](https://labelerrors.com) project.
- My employer, [Rasa](https://rasa.com/), has always had a focus on data quality. Some of that attitude is bound to have seeped in here. Be sure to check the [Conversation Driven Development](https://rasa.com/docs/rasa/conversation-driven-development/) approach and [Rasa X](https://rasa.com/docs/rasa-x/) if you're working on virtual assistants.
