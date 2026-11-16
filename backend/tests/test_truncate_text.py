import pytest
from utils import truncate_text


def test_truncate_text_rejects_negative_and_avoids_a_negative_slice() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        truncate_text("hello world", max_length=-1)
    assert truncate_text("hello world", max_length=2) == "he"
    assert truncate_text("hello world", max_length=5) == "he..."
    assert truncate_text("hi", max_length=10) == "hi"
