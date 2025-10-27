import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# --- 1. Parameters & Data Loading ---
VOCAB_SIZE = 10000  # Number of words to keep
MAX_LEN = 250      # Max length of reviews (in words)
EMBEDDING_DIM = 128  # Dimension for word embeddings
BATCH_SIZE = 64
EPOCHS = 5

print("Loading IMDb data...")
# Load data, limiting to top VOCAB_SIZE words
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
print(f"Loaded {len(x_train)} training sequences and {len(x_test)} test sequences.")

# --- 2. Text Pre-processing (Padding) ---
print("Padding sequences...")
# Pad/truncate sequences to be of length MAX_LEN
x_train = pad_sequences(x_train, maxlen=MAX_LEN)
x_test = pad_sequences(x_test, maxlen=MAX_LEN)
print(f"Shape of training data tensor: {x_train.shape}")
print(f"Shape of test data tensor: {x_test.shape}")

# --- 3. Model Building (Architecture) ---
print("Building the LSTM model...")
model = Sequential()

# 1. Embedding Layer
# Input: (batch_size, MAX_LEN) -> Output: (batch_size, MAX_LEN, EMBEDDING_DIM)
model.add(Embedding(input_dim=VOCAB_SIZE, 
                    output_dim=EMBEDDING_DIM, 
                    input_length=MAX_LEN))

# 2. LSTM Layer
# Processes the sequence of embeddings.
# We use 64 units and add dropout for regularization.
model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2))

# 3. Output Layer
# A single neuron with sigmoid activation for binary classification
model.add(Dense(units=1, activation='sigmoid'))

# --- 4. Model Compilation ---
print("Compiling the model...")
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Print a summary of the model architecture
model.summary()

# --- 5. Model Training ---
print("\nStarting model training...")
history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test)  # Use test data as validation
)

# --- 6. Model Evaluation ---
print("\nEvaluating model on test data...")
# Final evaluation on the test set
loss, accuracy = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# --- 7. Prediction Function for New Reviews ---

# Get the word index from keras.datasets.imdb
word_index = imdb.get_word_index()
# Reverse the index: integer -> word
reverse_word_index = {v: k for k, v in word_index.items()}

# Add 3 special-case tokens
# 0 = padding
# 1 = start of sequence
# 2 = unknown
reverse_word_index[0] = '<PAD>'
reverse_word_index[1] = '<START>'
reverse_word_index[2] = '<UNK>'

def decode_review(int_sequence):
    """Converts an integer sequence back to a human-readable review."""
    return ' '.join([reverse_word_index.get(i, '?') for i in int_sequence])

def predict_sentiment(review_text):
    """Predicts the sentiment of a new, raw text review."""
    
    # 1. Create word-to-int mapping
    word_to_int = {k: (v+3) for k, v in word_index.items() if (v+3) < VOCAB_SIZE}
    word_to_int['<PAD>'] = 0
    word_to_int['<START>'] = 1
    word_to_int['<UNK>'] = 2

    # 2. Pre-process the new text
    words = review_text.lower().split()
    # Convert words to ints, using <UNK> for words not in vocab
    int_sequence = [word_to_int.get(word, 2) for word in words] 
    
    # 3. Pad the sequence
    padded_sequence = pad_sequences([int_sequence], maxlen=MAX_LEN)
    
    # 4. Make prediction
    prediction = model.predict(padded_sequence)[0][0]
    
    # 5. Return result
    if prediction > 0.5:
        return f"Sentiment: Positive (Score: {prediction:.4f})"
    else:
        return f"Sentiment: Negative (Score: {prediction:.4f})"

# --- Test with new reviews ---
print("\n--- Testing with new reviews ---")
positive_review = "This movie was absolutely brilliant. The acting was superb and the plot was gripping."
print(f"Review: '{positive_review}'")
print(predict_sentiment(positive_review))

print("-" * 20)

negative_review = "I did not like this film at all. It was boring and a complete waste of time."
print(f"Review: '{negative_review}'")
print(predict_sentiment(negative_review))