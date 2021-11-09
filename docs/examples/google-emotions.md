This example is based on [this blogpost](https://koaning.io/posts/labels/). It is also *the* example
that motivated the creation of this project.

## Google Emotions

We're going to check for bad labels in the Google Emotions dataset. This dataset
contains text from Reddit (so expect profanity) with emotion tags attached. There are
28 different tags and a single text can belong to more than one emotion. We'll explore
the "excitement" emotion here, but the exercise can be repeated for many other emotions too.

The dataset comes with [a paper that lists details](https://arxiv.org/abs/2005.00547). When
you read the paper, you'll observe that a genuine effort was taken to make a high quality dataset.

- There are 82 raters involved n labelling this dataset. Each example should have been at least 3 people checking it. The paper mentions that all the folks who rated were from India but spoke English natively.
- An effort was made to remove subreddits that were not safe for work or that contained too much vulgar tokens (according to a predefined word-list).
- An effort was made to balance different subreddits such that larger subreddits wouldn’t bias the dataset.
- An effort was made to remove subreddits that didn’t offer a variety of emotions.
- An effort was made to mask names of people as well as references to religions.
- An effort was made to, in hindsight, confirm that there is sufficient interrated correlation.

Given that this is a dataset *from Google*, and the fact that there's a paper about it ... how
hard would it be to find bad labels?

## Data Loading

Let's load in a portion of the dataset.

```python
import pandas as pd

df = pd.read_csv("https://github.com/koaning/optimal-on-paper/raw/main/data/goemotions_1.csv")
```

Let's sample a few random rows and zoom in on the `excitement` column.

```python
label_of_interest = 'excitement'

(df[['text', label_of_interest]]
  .loc[lambda d: d[label_of_interest] == 0]
  .sample(4))
```

This is a sample.

|       | text                                                                    |   excitement |
|------:|:------------------------------------------------------------------------|-------------:|
| 27233 | my favourite singer ([NAME]) helped write one of the songs so i love it |            0 |
|  1385 | No i didn’t all i know is that i binged 3 seasoms of it.                |            0 |
| 17077 | I liked [NAME]...                                                       |            0 |
| 55699 | A "wise" man once told me: > DO > YOUR > OWN >RESEARCH >!               |            0 |

Again, we should remind folks that this is reddit data. Beware vulgar language.

## Models

Let's set up two modelling pipelines to detect the emotion. Let's start with a simple CountVectorizer model.

```python
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

X, y = list(df['text']), df[label_of_interest]

pipe = make_pipeline(
    CountVectorizer(),
    LogisticRegression(class_weight='balanced', max_iter=1000)
)
```

Next, let's also make a pipeline that uses text embeddings. We'll use the
[whatlies library](https://github.com/RasaHQ/whatlies) to do this.

```python
from sklearn.pipeline import make_union
from whatlies.language import BytePairLanguage

pipe_emb = make_pipeline(
    make_union(
        BytePairLanguage("en", vs=1_000),
        BytePairLanguage("en", vs=100_000)
    ),
    LogisticRegression(class_weight='balanced', max_iter=1000)
)
```

Let's train both pipelines before moving on.

```python
pipe.fit(X, y)
pipe_emb.fit(X, y)
```

## Assign Doubt

Let's now create a doubt ensemble using these two pipelines.

```python
from doubtlab.ensemble import DoubtEnsemble
from doubtlab.reason import ProbaReason, DisagreeReason, ShortConfidenceReason

reasons = {
    'proba': ProbaReason(pipe),
    'disagree': DisagreeReason(pipe, pipe_emb),
    'short_pipe': ShortConfidenceReason(pipe),
    'short_pipe_emb': ShortConfidenceReason(pipe_emb),
}

doubt = DoubtEnsemble(**reasons)
```

There are four reasons in this ensemble.

1. `proba`: This reason will assign doubt when the `pipe` pipeline doesn't predict any label with a high confidence.
2. `disagree`: This reason will assign doubt when the `pipe` pipeline doesn't agree with the `pipe_emb` pipeline.
3. `short_pipe`: This reason will assign doubt when the `pipe` pipeline predicts the correct label with a low confidence.
4. `short_pipe_emb`: This reason will assign doubt when the `pipe_emb` pipeline predicts the correct label with a low confidence.

All of these reasons have merit to it, but when they overlap we should assign extra attention. The `DoubtEnsemble` will assign the priority based on overlap on your behalf.

### Exploring Examples

Let's explore some of the labels that deserve attention.

```python
# Return a dataframe with reasoning behind sorting
predicates = doubt.get_predicates(X, y)

# Use predicates to sort original dataframe
df_sorted = df.iloc[predicates.index][['text', label_of_interest]]

# Create a dataframe containing predicates and original data
df_label = pd.concat([df_sorted, predicates], axis=1)
```

Let's check the first few rows of this dataframe.

```python
(df_label[['text', label_of_interest]]
  .head(10))
```

| text                             |   excitement |
|:---------------------------------|-------------:|
| Happy Easter everyone!!          |            0 |
| Happy Easter everyone!!          |            0 |
| Happy Easter everyone!!          |            0 |
| Congratulations mate!!           |            0 |
| Yes every time                   |            0 |
| New flavour! I love it!          |            0 |
| Wow! Prayers for everyone there. |            0 |
| Wow! Prayers for everyone there. |            0 |
| Hey vro!                         |            0 |
| Oh my gooooooooood               |            0 |


There's some examples that certainly contain excitement. However, these are all examples where the label is 0. Let's re-use this dataframe one more time but now to explore examples where the data says there should be excitement.

```python
(df_label[['text', label_of_interest]]
  .loc[lambda d: d['excitement'] == 1]
  .head(10))
```

| text                                                              |   excitement |
|:------------------------------------------------------------------|-------------:|
| Hate Hate Hate, feels so good.                                    |            1 |
| dear... husband                                                   |            1 |
| The old bear                                                      |            1 |
| I'd love to do that one day                                       |            1 |
| [NAME] damn I love [NAME].                                        |            1 |
| [NAME] is a really cool name!                                     |            1 |
| No haha but this is our first day on Reddit!                      |            1 |
| Yeah that pass                                                    |            1 |
| True! He probably is just lonely. Thank you for the kind words :) |            1 |
| a surprise to be sure                                             |            1 |

While some of the examples seem fine, I would argue that "dear ... husband" and "The old bear" are examples where the label is should be 0.

## Exploring Reasons

It's worth doing a minor deep dive in the behavior behind the different reasons. None of the reasons are perfect, but they all favor different examples for reconsideration.

### CountVectorizer short on Confidence

This is a "high"-bias bag-of-words model. It's going to likely overfit on the apperance of a token in the text.

```python
(df_label
 .sort_values("predicate_short_pipe", ascending=False)
 .head(10)[['text', label_of_interest]]
 .drop_duplicates())
```

| text                                                                                                                                      |   excitement |
|:------------------------------------------------------------------------------------------------------------------------------------------|-------------:|
| I am inexplicably excited by [NAME]. I get so excited by how he curls passes                                                              |            0 |
| Omg this is so amazing ! Keep up the awesome work and have a fantastic New Year !                                                         |            0 |
| Sounds like a fun game. Our home game around here is .05/.10. Its fun but not very exciting.                                              |            0 |
| So no replays for arsenal penalty calls.. Cool cool cool cool cool cool cool cool                                                         |            0 |
| Wow, your posting history is a real... interesting ride.                                                                                  |            0 |
| No different than people making a big deal about their team winning the super bowl. People find it interesting.                           |            0 |
| Hey congrats!! That's amazing, you've done such amazing progress! Hope you have a great day :)                                            |            0 |
| I just read your list and now I can't wait, either!! Hurry up with the happy, relieved and peaceful onward and upward!! Congratulations😎 |            0 |

### CountVectorizer with Low Proba

This is a "high"-bias bag-of-words model when it isn't confident. It's going to likely overfit examples with tokens that appear in both classes.

```python
(df_label
 .sort_values("predicate_proba", ascending=False)
 .head(10)[['text', label_of_interest]]
 .drop_duplicates())
```

| text                                                                                                                          |   excitement |
|:------------------------------------------------------------------------------------------------------------------------------|-------------:|
| Happy Easter everyone!!                                                                                                       |            0 |
| This game is on [NAME]...                                                                                                     |            0 |
| I swear if it's the Cowboys and the Patriots in the Super Bowl I'm going to burn something down.                              |            0 |
| I'm on red pills :)                                                                                                           |            0 |
| Wow. I hope that asst manager will be looking for a new job soon.                                                             |            0 |
| No lie I was just fucking watching the office but I paused it and am know listening to graduation and browsing this subreddit |            0 |
| I was imagining her just coming in to work wearing the full [NAME] look.                                                      |            0 |
| Like this game from a week ago? 26 points 14                                                                                  |            0 |
| You should come. You'd enjoy it.                                                                                              |            0 |
| I almost pissed myself waiting so long in the tunnel. Not a fun feeling                                                       |            0 |

### BytePair Embeddings short on Confidence

This is model based on just word embeddings. These embeddings are pooled together before being passed to the classifier which is likely why it favors short texts.

```python
(df_label
 .sort_values("predicate_short_pipe_emb", ascending=False)
 .head(20)[['text', label_of_interest]]
 .drop_duplicates())
```

| text                       |   excitement |
|:---------------------------|-------------:|
| Woot woot!                 |            0 |
| WOW!!!                     |            0 |
| Happy birthday!            |            0 |
| Happy Birthday!            |            0 |
| Happy one week anniversary |            0 |
| Happy Birthday!!!          |            0 |
| Pop pop!                   |            0 |
| Enjoy the ride!            |            0 |
| Very interesting!!!        |            0 |
| My exact reaction          |            0 |
| happy birthday dude!       |            0 |
| Enjoy                      |            0 |
| Oh wow!!!                  |            0 |
| This sounds interesting    |            0 |

## Conclusion

This example demonstrates two things.

1. By combining reasons into an ensemble, we get a pretty good system to spot examples worth double checking.
2. It's fairly easy to find bad labels, even in a dataset by Google, even when there's an article written about it. This does not bode well for any models trained on this dataset.

## Next Steps

Feel free to repeat this exercise but with a different emotion or with different reasoning in the ensemble.
