import pytest
import pathlib
from mktestdocs import check_docstring, grab_code_blocks

from doubtlab.reason import (
    ProbaReason,
    RandomReason,
    OutlierReason,
    DisagreeReason,
    LongConfidenceReason,
    ShortConfidenceReason,
    WrongPredictionReason,
    RegressionGapReason,
)

all_reasons = [
    ProbaReason,
    RandomReason,
    OutlierReason,
    DisagreeReason,
    LongConfidenceReason,
    ShortConfidenceReason,
    WrongPredictionReason,
    RegressionGapReason,
]


@pytest.mark.parametrize("func", all_reasons, ids=lambda d: d.__name__)
def test_function_docstrings(func):
    """Test the docstring code of some functions."""
    check_docstring(obj=func)


@pytest.mark.parametrize("fpath", ["README.md"])
def test_quickstart_docs_file(fpath):
    """Test the quickstart files."""
    grab_code_blocks(pathlib.Path(fpath).read_text())
