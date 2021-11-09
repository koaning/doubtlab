import pytest
from mktestdocs import check_docstring, check_md_file

from doubtlab.reason import (
    ProbaReason,
    RandomReason,
    OutlierReason,
    DisagreeReason,
    LongConfidenceReason,
    ShortConfidenceReason,
    WrongPredictionReason,
    AbsoluteDifferenceReason,
    RelativeDifferenceReason,
    CleanlabReason,
)
from doubtlab.ensemble import DoubtEnsemble

all_reasons = [
    ProbaReason,
    RandomReason,
    OutlierReason,
    DisagreeReason,
    LongConfidenceReason,
    ShortConfidenceReason,
    WrongPredictionReason,
    AbsoluteDifferenceReason,
    RelativeDifferenceReason,
    CleanlabReason,
]


@pytest.mark.parametrize(
    "func", all_reasons + [DoubtEnsemble], ids=lambda d: d.__name__
)
def test_function_docstrings(func):
    """Test the docstring code of some functions."""
    check_docstring(obj=func)


@pytest.mark.parametrize(
    "fpath",
    ["README.md", "docs/quickstart/index.md", "docs/examples/google-emotions.md"],
)
def test_quickstart_docs_file(fpath):
    """Test the quickstart files."""
    check_md_file(fpath, memory=True)
