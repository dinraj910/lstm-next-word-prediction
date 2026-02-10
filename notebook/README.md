# Training Notebooks

This folder contains Jupyter notebooks for model training and experimentation.

## Contents

| Notebook | Description |
|----------|-------------|
| `training.ipynb` | Model training pipeline (optional) |

## Training Pipeline Overview

1. **Data Loading:** Load Shakespeare text corpus
2. **Preprocessing:** Tokenization, sequence creation
3. **Model Definition:** Embedding → LSTM → Dense (Softmax)
4. **Training:** Fit model with sparse categorical cross-entropy
5. **Export:** Save model (`.h5`) and tokenizer (`.pkl`)

## Note

Training notebooks are optional for running the inference application.
The `app/` folder contains the production inference code.
