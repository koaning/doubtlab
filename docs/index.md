<img src="doubt.png" width=150 height=150 align="right">

<b><h1 style="color:black;">doubtlab</h1></b>

> DoubtLab helps you find bad labels.

<br>

This repository contains general tricks that may help you find bad, or noisy, labels in your dataset. The hope is that this repository makes it easier for folks to quickly check their own datasets before they invest too much time and compute on gridsearch.

## Installation

You can install the tool via pip.

```
python -m pip install doubtlab
```

## Getting Started

If you want to get started, we recommend starting [here](./quickstart/).

## Related Projects

- The [cleanlab](https://github.com/cleanlab/cleanlab) project was an inspiration for this one. They have a great heuristic for bad label detection but I wanted to have a library that implements many. Be sure to check out their work on the [labelerrors.com](https://labelerrors.com) project.
- My employer, [Rasa](https://rasa.com/), has always had a focus on data quality. Some of that attitude is bound to have seeped in here. Be sure to check the [Conversation Driven Development](https://rasa.com/docs/rasa/conversation-driven-development/) approach and [Rasa X](https://rasa.com/docs/rasa-x/) if you're working on virtual assistants.
