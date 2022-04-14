import pytest
from mktestdocs import check_docstring, check_md_file, get_codeblock_members

from doubtlab.reason import (
    ProbaReason,
    RandomReason,
    OutlierReason,
    DisagreeReason,
    LongConfidenceReason,
    ShortConfidenceReason,
    MarginConfidenceReason,
    WrongPredictionReason,
    AbsoluteDifferenceReason,
    RelativeDifferenceReason,
    CleanlabReason,
    ShannonEntropyReason,
)
from doubtlab.ensemble import DoubtEnsemble
from doubtlab.benchmark import flip_labels, shuffle_labels


all_objects = [
    ProbaReason,
    RandomReason,
    OutlierReason,
    DisagreeReason,
    LongConfidenceReason,
    ShortConfidenceReason,
    MarginConfidenceReason,
    WrongPredictionReason,
    AbsoluteDifferenceReason,
    RelativeDifferenceReason,
    CleanlabReason,
    ShannonEntropyReason,
    DoubtEnsemble,
]


def flatten(items):
    """Flattens a list"""
    return [item for sublist in items for item in sublist]


# This way we ensure that each item in `all_members` points to a method
# that could have a docstring.
all_members = flatten([get_codeblock_members(o) for o in all_objects]) + [
    flip_labels,
    shuffle_labels,
]


@pytest.mark.parametrize("func", all_members, ids=lambda d: d.__qualname__)
def test_function_docstrings(func):
    """Test the python example in each method in each object."""
    check_docstring(obj=func)


@pytest.mark.parametrize(
    "fpath",
    [
        "README.md",
        "docs/quickstart/benchmarks.md",
        "docs/examples/google-emotions.md",
    ],
)
def test_quickstart_docs_file(fpath):
    """Test the quickstart files."""
    check_md_file(fpath, memory=True)
