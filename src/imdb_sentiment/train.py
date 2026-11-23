"""Train on IMDb, validate on held-out training examples, evaluate once on test."""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from .model import build_model
from .preprocessing import save_metadata


def main():
    parser = argparse.ArgumentParser(description="Train the IMDb LSTM classifier")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--max-length", type=int, default=250)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import tensorflow as tf
    from tensorflow.keras.datasets import imdb
    from tensorflow.keras.utils import pad_sequences

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=args.vocab_size)
    x_train = pad_sequences(x_train, maxlen=args.max_length, padding="pre", truncating="pre")
    x_test = pad_sequences(x_test, maxlen=args.max_length, padding="pre", truncating="pre")

    # A reproducible validation split from TRAIN ONLY. Test stays unseen until final evaluation.
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(len(x_train))
    val_count = int(0.2 * len(x_train))
    validation_ids, training_ids = shuffled[:val_count], shuffled[val_count:]
    model = build_model(vocab_size=args.vocab_size, max_length=args.max_length)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]
    history = model.fit(x_train[training_ids], np.asarray(y_train)[training_ids],
                        validation_data=(x_train[validation_ids], np.asarray(y_train)[validation_ids]),
                        epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    model.save(output / "sentiment_lstm.keras")
    save_metadata(output / "metadata.json", imdb.get_word_index(), args.vocab_size, args.max_length)
    metrics = {"test_loss": float(loss), "test_accuracy": float(accuracy),
               "epochs_completed": len(history.history["loss"]), "seed": args.seed,
               "train_samples": int(len(training_ids)), "validation_samples": int(len(validation_ids)),
               "test_samples": int(len(x_test)), "history": {k: [float(v) for v in vals] for k, vals in history.history.items()}}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Test accuracy: {accuracy:.4%}; test loss: {loss:.4f}")
    print(f"Artifacts saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
