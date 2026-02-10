# Model Artifacts

This folder should contain the trained model files:

## Required Files

| File | Description |
|------|-------------|
| `next_word_lstm.h5` | Trained Keras LSTM model |
| `tokenizer.pkl` | Fitted Keras Tokenizer (pickle format) |

## Model Specifications

- **Architecture:** Single-layer LSTM
- **Vocabulary Size:** 5,000 words
- **Context Length:** 5 tokens
- **Training Data:** Shakespeare corpus

## Note

Model files are excluded from version control via `.gitignore` due to their size.
For production deployments, consider using Git LFS or cloud storage (S3, GCS, Azure Blob).
