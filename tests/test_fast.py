import pytest


@pytest.mark.fast
def test_import() -> None:
    import pythole  # noqa
