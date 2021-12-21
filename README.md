<!--- BADGES: START --->
[![GitHub - License](https://img.shields.io/github/license/koaning/doubtlab?logo=github&style=flat&color=green)][#github-license]
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/doubtlab?logo=pypi&style=flat&color=blue)][#pypi-package]
[![PyPI - Package Version](https://img.shields.io/pypi/v/doubtlab?logo=pypi&style=flat&color=orange)][#pypi-package]
[![Conda - Platform](https://img.shields.io/conda/pn/conda-forge/doubtlab?logo=anaconda&style=flat)][#conda-forge-package]
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/doubtlab?logo=anaconda&style=flat&color=orange)][#conda-forge-package]
[![Docs - GitHub.io](https://img.shields.io/static/v1?logo=github&style=flat&color=pink&label=docs&message=doubtlab)][#docs-package]


[#github-license]: https://github.com/koaning/doubtlab/blob/main/LICENSE
[#pypi-package]: https://pypi.org/project/doubtlab/
[#conda-forge-package]: https://anaconda.org/conda-forge/doubtlab
[#docs-package]: https://koaning.github.io/doubtlab/
<!--- BADGES: END --->

<img src="docs/doubt.png" width=125 height=125 align="right">

# doubtlab

> A lab for bad labels.

This repository contains general tricks that may help you find bad, or noisy, labels in your dataset. The hope is that this repository makes it easier for folks to quickly check their own datasets before they invest too much time and compute on gridsearch.

## Install

You can install the tool via `pip` or `conda`.

**Install with pip**

```
python -m pip install doubtlab
```

**Install with conda**

```
conda install -c conda-forge doubtlab
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

The library implemented many "reasons" for doubt.

### General Reasons

- `RandomReason`: assign doubt randomly, just for sure
- `OutlierReason`: assign doubt when the model declares a row an outlier

### Classification Reasons

- `ProbaReason`: assign doubt when a models' confidence-values are low for any label
- `WrongPredictionReason`: assign doubt when a model cannot predict the listed label
- `ShortConfidenceReason`: assign doubt when the correct label gains too little confidence
- `LongConfidenceReason`: assign doubt when a wrong label gains too much confidence
- `DisagreeReason`: assign doubt when two models disagree on a prediction
- `CleanlabReason`: assign doubt according to [cleanlab](https://github.com/cleanlab/cleanlab)
- `MarginConfidenceReason`: assign doubt when there's a small difference between the top two class confidences

### Regression Reasons

- `AbsoluteDifferenceReason`: assign doubt when the absolute difference is too high
- `RelativeDifferenceReason`: assign doubt when the relative difference is too high
- `StandardizedErrorReason`: assign doubt when the absolute standardized residual is too high

## Feedback

It is early days for the project. The project should be plenty useful as-is, but we
prefer to be honest. Feedback and anekdotes are very welcome!

## Related Projects

- The [cleanlab](https://github.com/cleanlab/cleanlab) project was an inspiration for this one. They have a great heuristic for bad label detection but I wanted to have a library that implements many. Be sure to check out their work on the [labelerrors.com](https://labelerrors.com) project.
- My employer, [Rasa](https://rasa.com/), has always had a focus on data quality. Some of that attitude is bound to have seeped in here. Be sure to check the [Conversation Driven Development](https://rasa.com/docs/rasa/conversation-driven-development/) approach and [Rasa X](https://rasa.com/docs/rasa-x/) if you're working on virtual assistants.
