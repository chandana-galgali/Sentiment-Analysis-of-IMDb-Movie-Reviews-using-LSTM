# Training results

These numbers come from a 5-epoch training run and match [`models/metrics.json`](../models/metrics.json).

**Command:** `python -m imdb_sentiment.train --epochs 5`  
**Environment:** Ubuntu (WSL2), Python 3.12.3, CPU (CUDA unavailable)  
**Dataset:** Keras IMDb; 20,000 fit reviews, 5,000 validation reviews, 25,000 test reviews  
**Test accuracy:** **85.2760%**  
**Test loss:** **0.3494**  
**Training epochs completed:** **5**

| Epoch | Training accuracy | Training loss | Validation accuracy | Validation loss |
|---|---:|---:|---:|---:|
| 1 | 76.07% | 0.4965 | 68.14% | 0.6094 |
| 2 | 73.50% | 0.5423 | 77.24% | 0.4850 |
| 3 | 86.01% | 0.3391 | 85.40% | 0.3564 |
| 4 | 90.18% | 0.2526 | 85.94% | **0.3337** |
| 5 | 92.98% | 0.1960 | **86.22%** | 0.3478 |

### Interpretation

Validation accuracy continued to improve slightly in epoch 5, but validation loss began increasing after its minimum in epoch 4. This suggests emerging overfitting, so additional epochs are not automatically beneficial. The training script uses early stopping on validation loss (`patience=2`, `restore_best_weights=True`).

The test accuracy is for the final saved model. Results may differ slightly on retraining. The saved model and its artifacts are in `models/`: `sentiment_lstm.keras`, `metadata.json` and `metrics.json`.
