# Sentiment-Analysis-of-IMDb-Movie-Reviews-using-LSTM

## 🚀 Project Overview

The goal of this project is to classify movie reviews as either **Positive** or **Negative** using a Recurrent Neural Network (RNN). Specifically, a **Long Short-Term Memory (LSTM)** network is built using TensorFlow and Keras.

## 🛠️ Model Architecture

The model architecture is a sequential Keras model consisting of three main layers:

1.  **Embedding Layer:** Takes the integer-encoded vocabulary (top 10,000 words) and generates dense 128-dimension word embeddings.
2.  **LSTM Layer:** A 64-unit LSTM layer processes the sequence of embeddings, capturing context and temporal dependencies. Dropout (0.2) is used for regularization.
3.  **Dense (Output) Layer:** A single neuron with a `sigmoid` activation function outputs a probability score between 0 (Negative) and 1 (Positive).

## 📊 Dataset

* **Source:** IMDb Movie Review Dataset
* **Size:** 50,000 reviews
* **Split:** 25,000 for training, 25,000 for testing.
* **Balance:** The dataset is perfectly balanced with 50% positive and 50% negative reviews.

## ⚙️ Setup and Usage

This project is best run on **Google Colab** to leverage free GPU access.

1.  **Clone the repository (optional):**
    ```bash
    git clone https://github.com/chandana-galgali/Sentiment-Analysis-of-IMDb-Movie-Reviews-using-LSTM.git
    ```
2.  **Create a virtual environment (Recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    (You should create a `requirements.txt` file with the content below)
    ```
    tensorflow
    numpy
    ```
    Then, run:
    ```bash
    pip install -r requirements.txt
    ```
    *(If you don't make the file, just run `pip install tensorflow numpy`)*

4.  **Run the Python script:**
    (Assuming you named your file `lstm_sentiment_analysis.py`)
    ```bash
    python lstm_sentiment_analysis.py
    ```

The script will automatically:
* Download and load the IMDb dataset from `tensorflow.keras.datasets`.
* Preprocess and pad the data.
* Build, compile, and train the LSTM model (this will take a few minutes).
* Evaluate the model and print the final test accuracy.
* Provide a demonstration by predicting the sentiment of two new, custom reviews.

## 📈 Results

After 5 epochs of training, the model achieved the following performance on the unseen test set:

* **Test Loss:** 0.3664
* **Test Accuracy:** 85.51%

*(Note: Replace the values above with the actual results from your training run)*