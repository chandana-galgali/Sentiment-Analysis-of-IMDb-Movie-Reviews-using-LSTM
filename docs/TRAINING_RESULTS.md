# Training results — 23 November 2026

These numbers were captured from the **user's completed Ubuntu WSL2 training terminal**.

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

Validation accuracy continued to improve slightly in epoch 5, but validation loss began increasing after its minimum in epoch 4. This suggests emerging overfitting; additional epochs are not automatically beneficial. The training script uses early stopping with `restore_best_weights=True` and `patience=2`. At a maximum of five epochs, the final serialized model should contain epoch-4 weights, assuming the reported run used the supplied unchanged training script. The saved model should be checked on the original training machine.

The reported test accuracy is for the final evaluated model. Results are specific to this run and may differ on retraining. Raw model `models/metrics.json`, `models/metadata.json`, and `models/sentiment_lstm.keras` are generated on the user's machine and **are not contained in this source-only ZIP**.
