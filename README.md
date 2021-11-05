<img src="doubt.png" width=125 height=125 align="right">

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
from doubtlab import DoubtLab
from doubtlab.reasons import ProbaReason, WrongPredictionReason

# Let's say we have some model already
model.fit(X, y)

# Next we can the reasons for doubt. In this case we're saying
# that examples deserve another look if the associated proba values
# are low or if the model output doesn't match the associated label.
reasons = {
    'proba': ProbaReason(model=model),
    'wrong_pred': WrongPredictionReason(model=model)
}

# Pass these reasons to a doubtlab instance.
doubt = DoubtLab(**reasons)

# Get the predicates, or reasoning, behind the order
predicates       = doubt.get_predicates(X, y)
# Get the ordered indices of examples worth checking again
indices          = doubt.get_indices(X, y)
# Get the (X, y) candidates worth checking again
X_check, y_check = doubt.candidates(X, y)
```

## Features

The library implemented many "reaons" for doubt.

- `ProbaReason`: assign doubt when a models' confidence-values are low
- `RandomReason`: assign doubt randomly, just for sure
- `LongConfidenceReason`: assign doubt when a wrong class gains too much confidence
- `ShortConfidenceReason`: assign doubt when the correct class gains too little confidence
- `DisagreeReason`: assign doubt when two models disagree on a prediction
- `CleanLabReason`: assign doubt according to [cleanlab](https://github.com/cleanlab/cleanlab)

## Related Projects 

- The [cleanlab](https://github.com/cleanlab/cleanlab) project was an inspiration for this one. They have a great heuristic for bad label detection but I wanted to have a library that implements many. Be sure to check out their work on the [labelerrors.com](https://labelerrors.com) project.
- My employer, [Rasa](https://rasa.com/), has always had a focus on data quality. Some of that attitude is bound to have seeped in here. Be sure to check out [Rasa X](https://rasa.com/docs/rasa-x/) if you're working on virtual assistants.
