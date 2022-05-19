The library implements a few general tricks to find bad labels, but you
can also re-use the components for more elaborate approaches. In this section
we hope to demonstrate some more techniques.

## Bootstrapping

Bootstrapping can be used as a technique to train many similar, but different,
models on the same dataset. The predictive difference between these models can
be used as a proxy for confidence as well as bad labels.

<br>

<script src="https://cdn.tailwindcss.com"></script>

<style>
    .tooltip {
    background-color: white;
    color: black;
    padding: 5px 10px;
    border-radius: 4px;
    font-size: 13px;
    }
</style>

<div style="position: relative; width: 100%; min-width: 400px;">
    <img class="h-auto w-full" src="../images/diagram.png">
    <tooltip style="top:60%; left:15%;" class="font-bold text-sm px-1 py-0 absolute rounded-full bg-blue-500 text-white" id="button1" aria-describedby="tooltip">A</tooltip>
    <tooltip style="top:60%; left:50%;" class="font-bold text-sm px-1 py-0 absolute rounded-full bg-blue-500 text-white" id="button2" aria-describedby="tooltip">B</tooltip>
    <tooltip style="top:60%; left:76%;" class="font-bold text-sm px-1 py-0 absolute rounded-full bg-blue-500 text-white" id="button3" aria-describedby="tooltip">C</tooltip>
</div>

<div class="tooltip text-lg w-64" id="tooltip1" role="tooltip" style="line-height: 150%; position: absolute; inset: 0px auto auto 0px; margin: 0px; transform: translate3d(159.333px, 112.667px, 0px);" data-popper-placement="bottom">
    Suppose that we start with an original dataset and that we resample it a bunch, using bootstrap samples.
</div>
<div class="tooltip text-lg w-64" id="tooltip2" role="tooltip" style="line-height: 150%; position: absolute; inset: 0px auto auto 0px; margin: 0px; transform: translate3d(-15.333px, -146px, 0px);" data-popper-placement="bottom">
    Then we can train a model on each of these subsets. Each of these models would be different, but given enough samples they should all be reasonable.
</div>
<div class="tooltip text-lg w-64" id="tooltip3" role="tooltip" style="line-height: 150%; position: absolute; inset: 0px auto auto 0px; margin: 0px; transform: translate3d(441.333px, 132.667px, 0px);" data-popper-placement="bottom">
    Because these models are different, the predictions that come out will also differ. If the predictions vary, that's an indication of less confidence, which in turn could indicate bad labels.
</div>

<script src="https://unpkg.com/@popperjs/core@2"></script>
<style>
    .tooltip {
        display: none;
    }
    
    .tooltip[data-show] {
        display: block;
    }
</style>
<script>
    let array = ["#button1", "#button2", "#button3"];
    for (let index = 0; index < array.length; index++) {
        let btn = array[index];
        let button = document.querySelector(btn);
        let tooltip = document.querySelector(btn.replace("button", "tooltip"));
        popperInstance = Popper.createPopper(button, tooltip, {
            placement: "bottom"
        });

        function show() {
            tooltip.setAttribute('data-show', '');

            // We need to tell Popper to update the tooltip position
            // after we show the tooltip, otherwise it will be incorrect
            popperInstance.update();
        }

        function hide() {
            tooltip.removeAttribute('data-show'); 
        }

        const showEvents = ['mouseenter', 'focus'];
        const hideEvents = ['mouseleave', 'blur'];

        showEvents.forEach((event) => {
            button.addEventListener(event, show);
        });

        hideEvents.forEach((event) => {
            button.addEventListener(event, hide);
        });   
    }
</script>

### Classification 

You can use scikit-learn to construct a bootstrapped model for classification which can also be used
in this library. You'll want to use the [bagging ensemble models](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) for this. 

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Load in a demo dataset
X, y = make_classification()

# Train a classifier based on bootstrap samples
bc = BaggingClassifier(
    base_estimator=LogisticRegression(),
    max_samples=0.5,
    n_estimators=20
)
bc.fit(X, y)

# You can inspect the trained estimators manually
bc.estimators_

# But you can also predict the probabilities. 
bc.predict_proba(X)
```

These probability values indicate how many internal models predicted a class.
To turn these predicted proba values into a reason for label doubt we can use
the `ProbaReason`, `LongConfidenceReason` or the `ShortConfidenceReason`. 

### Regression 

There's a similar trick we might be able to do for regression too!

```python
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Load in a demo dataset
X, y = make_regression()

# Train a classifier based on bootstrap samples
bc = BaggingRegressor(
    base_estimator=LinearRegression(),
    max_samples=0.5,
    n_estimators=20
)
bc.fit(X, y)

# You can inspect the trained estimators manually
bc.estimators_

# So you could check the variance between predictions
dev = np.array([e.predict(X) for e in bc.estimators]).std(axis=1)
```

The deviations in `dev` could again be interpreted as a proxy for doubt. Because
a doubt ensemble is just an ensemble of callables you can implemented a reason
via:

```python
from doubtlab.ensemble import DoubtEnsemble 

threshold = 2

DoubtEnsemble(
    wrong_pred=lambda X, y: np.array([e.predict(X) for e in bc.estimators]).std(axis=1) > threshold
)
```