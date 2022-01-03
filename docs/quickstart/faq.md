## How do I add a reason for `nan` values?

A reason in doubtlab is little more than a function that can attach a 0/1
doubt-label to a row of data. As explained [here](), that means that you
can totally use `lambda` functions!

To implement this, you'll likely need to write something like:

```python
from doubtlab.ensemble import DoubtEnsemble

ensemble = DoubtEnsemble(
    wrong_pred=lambda X, y: (model.predict(X) != y).astype(float16),
    nan_label=lambda X, y: y.isnan(),
)
```

Note that you can also add another reason for `nan` values that appear
in `X`.

## How do I prevent models from re-computing?

Suppose you have a setup that looks something like:

```python
from doubtlab.ensemble import DoubtEnsemble
from doubtlab.reason import ProbaReason, ShortConfidenceReason, LongConfidenceReason

# Suppose this dataset is very big and that this computation is heavy.
X, y = load_big_dataset()
model = LogisticRegression(max_iter=1_000)
model.fit(X, y)

# This step might be expensive because internally we will be calling
# `model.predict_proba(X)` a lot!
ensemble = DoubtEnsemble(
    proba=ProbaReason(model)
    short=ShortConfidenceReason(model),
    long=LongConfidenceReason(model)
)
```

Then you might wonder if we're able to speed things up by precomputing our
`.predict_proba()`-values. You could use `lambda`s, but you can also use
common utility methods that have been added to the reason classes. Most of
our reasons implement a `from_pred` or `from_proba` method that you can use.
See the [API](https://koaning.github.io/doubtlab/api/reasons/) for more details.

That way, we can rewrite the code for a speedup.

```python
from doubtlab.ensemble import DoubtEnsemble
from doubtlab.reason import ProbaReason, ShortConfidenceReason, LongConfidenceReason

# Suppose this dataset is very big and that this computation is heavy.
X, y = load_big_dataset()
model = LogisticRegression(max_iter=1_000)
model.fit(X, y)

# Let's precalculate the proba values.
probas = model.predict_proba(X)

# We can re-use the probas below. Note that some reasons require extra information.
ensemble = DoubtEnsemble(
    proba=ProbaReason.from_proba(probas)
    short=ShortConfidenceReason.from_proba(probas, y, classes=["pos", "neg"], threshold=0.2),
    long=LongConfidenceReason.from_proba(probas, y, classes=["pos", "neg"], threshold=0.4)
)
```
