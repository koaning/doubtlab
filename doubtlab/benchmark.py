import numpy as np
import pandas as pd


def _parse_check_p_n_y(p, n, y):
    """Parses and checks `n`, `y` and `p`, returns (inferred) `n`."""
    if p:
        if p < 0:
            raise ValueError("Probability value `p` must be larger than 0.")
        if p > 1:
            raise ValueError("Probability value `p` must be less than 1.")
        n = round(len(y) * p)
    if not n:
        raise ValueError("Either `n` or `p` must be given to `shuffle_labels`.")
    if n <= 1:
        raise ValueError("Must shuffle at least 2 values. Increase `n` or `p`.")
    return n


def shuffle_labels(y, random_seed=42, n=None, p=None):
    """
    Shuffles subset of labels for benchmarking. Recommended for regression.

    Either `p` or `n` should be given. Returns a tuple `(y_out, indicator)`-tuple.

    Arguments:
        y: array of labels
        random_seed: random seed
        n: number of labels to flip
        p: percentage of labels to flip

    Usage:

    ```python
    import numpy as np
    from doubtlab.benchmark import shuffle_labels

    # Let's pretend these are the actual labels
    y = np.random.normal(0, 1, 10000)

    # You now have some shuffled labels and an indicator
    y_out, indicator = shuffle_labels(y, n=100)
    ```
    """
    np.random.seed(random_seed)
    y = np.array(y)
    n = _parse_check_p_n_y(p=p, n=n, y=y)

    y_out = y.copy()
    sample = np.random.choice(np.arange(y.shape[0]), size=n, replace=False)

    # Since `sample` is already randomly shuffled, we can move everything
    # over by one index to guarantee a shuffle of the values from another index.
    y_out[sample] = np.concatenate([sample[1:], sample[0:1]])
    return y_out, (y != y_out).astype(int)


def flip_labels(y, random_seed=42, n=None, p=None):
    """
    Flips subset of labels for benchmarking. Recommended for classification.

    Either `p` or `n` should be given. Returns a tuple `(y_out, indicator)`-tuple.

    Arguments:
        y: array of labels
        random_seed: random seed
        n: number of labels to flip
        p: percentage of labels to flip

    Usage:

    ```python
    import numpy as np
    from doubtlab.benchmark import flip_labels

    # Let's pretend these are the actual labels
    y = np.random.randint(0, 3, 10000)

    # You now have some shuffled labels and an indicator
    y_out, indicator = flip_labels(y, n=100)
    ```
    """
    np.random.seed(random_seed)
    y = np.array(y)
    n = _parse_check_p_n_y(p=p, n=n, y=y)

    y_out = y.copy()
    classes = np.unique(y)
    if len(classes) == 1:
        raise ValueError("Need more that 1 class in `y`.")

    # Only sample classes that didn't appear before.
    idx = np.random.choice(np.arange(y.shape[0]), size=n, replace=False)
    y_out[idx] = [np.random.choice(classes[classes != _]) for _ in y_out[idx]]
    return y_out, (y != y_out).astype(int)


def calculate_precision_recall_at_k(
    predicate_df, idx_flip, max_k=100, give_random=False, give_ensemble=True
):
    """
    Plots precision/recall at `k` values for flipped label experiments.

    Returns an interactive altair visualisation. Make sure it is installed beforehand.

    Arguments:
        predicate_df: the dataframe with predicates from `ensemble.get_predicates`
        idx_flip: array that indicates if labels are wrong
        max_k: the maximum value for `k` to consider
        give_random: plot the "at k" statistics for the randomly selected lower bound
        give_ensemble: plot the "at k" statistics from the reason ensemble
    """
    # First we need to ensure that the original dataframe with X values is
    # is combined with our reasons dataframe and sorted appropriately.
    df = predicate_df.assign(
        s=lambda d: d[[c for c in d.columns if "predicate" in c]].sum(axis=1),
        flipped=idx_flip,
    ).sort_values("s", ascending=False)

    # Next we calculate the precision/recall at k values
    data = []
    for k in range(1, max_k):
        recall_at_k = df["flipped"][:k].sum() / df["flipped"].sum()
        precision_at_k = (df["flipped"][:k] == np.ones(k)).sum() / k
        random_recall = df["flipped"].mean() * k / df["flipped"].sum()
        random_precision = df["flipped"].mean()
        data.append(
            {
                "recall_at_k": recall_at_k,
                "precision_at_k": precision_at_k,
                "k": k,
                "setting": "ensemble",
            }
        )
        data.append(
            {
                "recall_at_k": random_recall,
                "precision_at_k": random_precision,
                "k": k,
                "setting": "random",
            }
        )
    result = pd.DataFrame(data).melt(["k", "setting"])
    # Give the user the option to only return draw a subset
    if not give_random:
        result = result.loc[lambda d: d["setting"] != "random"]
    if not give_ensemble:
        result = result.loc[lambda d: d["setting"] != "ensemble"]

    # Return the data in a tidy format.
    return result


def plot_precision_recall_at_k(
    predicate_df, idx_flip, max_k=100, give_random=True, give_ensemble=True
):
    """
    Plots precision/recall at `k` values for flipped label experiments.

    Returns an interactive altair visualisation. Make sure it is installed beforehand.

    Arguments:
        predicate_df: the dataframe with predicates from `ensemble.get_predicates`
        idx_flip: array that indicates if labels are wrong
        max_k: the maximum value for `k` to consider
        give_random: plot the "at k" statistics for the randomly selected lower bound
        give_ensemble: plot the "at k" statistics from the reason ensemble
    """
    import altair as alt

    alt.data_transformers.disable_max_rows()

    # We combine the results in dataframes
    plot_df = calculate_precision_recall_at_k(
        predicate_df=predicate_df,
        idx_flip=idx_flip,
        max_k=max_k,
        give_random=give_random,
        give_ensemble=give_ensemble,
    )

    # So that we may plot it.
    return (
        alt.Chart(plot_df)
        .mark_line()
        .encode(x="k", y="value", color="variable", strokeDash="setting")
        .interactive()
    )
