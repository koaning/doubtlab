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
