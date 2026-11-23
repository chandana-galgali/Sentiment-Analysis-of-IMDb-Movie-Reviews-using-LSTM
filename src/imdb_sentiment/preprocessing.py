"""Tokenization consistent with the Keras IMDb dataset word index."""
import json
from pathlib import Path

# Same default filters as keras.preprocessing.text.text_to_word_sequence.
FILTERS = '"!#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
PAD = 0
START = 1
UNK = 2
INDEX_OFFSET = 3


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("Review must be a string")
    translation = str.maketrans({char: " " for char in FILTERS})
    return text.lower().translate(translation).split()


def encode_review(text: str, word_index: dict[str, int], vocab_size: int = 10000) -> list[int]:
    if not text or not text.strip():
        raise ValueError("Review cannot be empty")
    if vocab_size <= INDEX_OFFSET:
        raise ValueError("vocab_size must exceed 3")
    tokens = tokenize(text)
    return [START] + [
        idx + INDEX_OFFSET if (idx := word_index.get(token, -INDEX_OFFSET)) >= 0
        and idx + INDEX_OFFSET < vocab_size else UNK
        for token in tokens
    ]


def save_metadata(path: str | Path, word_index: dict[str, int], vocab_size: int, max_length: int):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"word_index": word_index, "vocab_size": vocab_size,
                                       "max_length": max_length}, separators=(",", ":")), encoding="utf-8")


def load_metadata(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("word_index", "vocab_size", "max_length"):
        if key not in data:
            raise ValueError(f"Metadata missing: {key}")
    return data
