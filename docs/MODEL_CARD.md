# Model card — CineSentiment IMDb LSTM

## Intended task
English IMDb-style movie review binary sentiment classification: positive versus negative. Educational and portfolio use only.

## Data and model
Keras IMDb dataset: 25,000 original training reviews split into 20,000 fitting / 5,000 validation reviews, plus an untouched 25,000-review test set. Fixed seed 42. Vocabulary capped at 10,000 and sequences padded/truncated to 250 tokens. Architecture: 128-dimensional embedding → 64-unit LSTM (dropout 0.2) → sigmoid. Binary cross-entropy, Adam.

## Observed results
**Reported evaluation, Python 3.12.3 / Ubuntu WSL2:** 85.2760% test accuracy and 0.3494 test loss, after five epochs. Maximum validation accuracy 86.22% (epoch 5); minimum validation loss 0.3337 (epoch 4). Full epoch table and provenance: [Training results](TRAINING_RESULTS.md). No independent rerun or calibration was performed for this ZIP.

## Limitations
The positive score is not a calibrated confidence value. Sarcasm, mixed opinions, negation, non-English reviews, and out-of-domain inputs may reduce quality. Not appropriate for high-stakes decision-making, moderation enforcement, sensitive profiling, or claims about individual preferences.
