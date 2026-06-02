import pytest
from src.chunk import chunk_text


def test_basic_chunking_no_overlap():
    assert chunk_text("one two three four five six", chunk_size=2) == [
        "one two",
        "three four",
        "five six",
    ]


def test_overlap_repeats_words():
    assert chunk_text("a b c d e", chunk_size=3, overlap=1) == ["a b c", "c d e", "e"]


def test_empty_text_returns_empty_list():
    assert chunk_text("", chunk_size=4) == []


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=0)
