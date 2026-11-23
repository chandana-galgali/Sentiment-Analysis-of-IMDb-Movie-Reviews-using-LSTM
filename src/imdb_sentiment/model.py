"""Model definition and inference helpers."""
from pathlib import Path
from .preprocessing import encode_review, load_metadata


def build_model(vocab_size=10000, max_length=250, embedding_dim=128, lstm_units=64):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_length,)),
        tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=False),
        tf.keras.layers.LSTM(lstm_units, dropout=0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def load_artifacts(directory="models"):
    import tensorflow as tf
    path = Path(directory)
    model_file = path / "sentiment_lstm.keras"
    metadata_file = path / "metadata.json"
    if not model_file.exists() or not metadata_file.exists():
        raise FileNotFoundError("Model artifacts missing. Run `python -m imdb_sentiment.train` first; see README.")
    return tf.keras.models.load_model(model_file), load_metadata(metadata_file)


def predict(text: str, model, metadata: dict) -> dict:
    from tensorflow.keras.utils import pad_sequences
    encoded = encode_review(text, metadata["word_index"], metadata["vocab_size"])
    padded = pad_sequences([encoded], maxlen=metadata["max_length"], padding="pre", truncating="pre")
    score = float(model.predict(padded, verbose=0)[0][0])
    return {"label": "Positive" if score >= 0.5 else "Negative", "positive_probability": score,
            "negative_probability": 1.0 - score}
