import pytest
from imdb_sentiment.preprocessing import encode_review, tokenize, save_metadata, load_metadata


def test_tokenize_normalizes_case_and_punctuation():
    assert tokenize("Great, MOVIE! Isn't it?") == ["great", "movie", "isn't", "it"]


def test_encoding_uses_keras_offset_and_start_token():
    assert encode_review("Great movie unknown!", {"great": 10, "movie": 20}, 100) == [1, 13, 23, 2]


def test_out_of_vocabulary_word():
    assert encode_review("film", {"film": 99}, 100) == [1, 2]


def test_empty_text_rejected():
    with pytest.raises(ValueError):
        encode_review("  ", {})


def test_metadata_round_trip(tmp_path):
    path = tmp_path / "metadata.json"
    save_metadata(path, {"great": 5}, 100, 25)
    assert load_metadata(path) == {"word_index": {"great": 5}, "vocab_size": 100, "max_length": 25}
