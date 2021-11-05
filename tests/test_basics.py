import doubtlab


def test_has_version():
    """Can't have a library without a version number."""
    assert doubtlab.__version__
