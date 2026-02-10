# Next Word Prediction with LSTM

A production-ready **Next Word Prediction** application using an LSTM-based language model trained on Shakespeare's works. This project demonstrates core NLP concepts including sequence modeling, word embeddings, and inference optimization.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🎯 Project Overview

**Task:** Language Modeling (Next Word Prediction)

This application predicts the most likely next word given a text prefix, serving as the foundation for autocomplete systems and text generation. The model learns statistical patterns in Shakespeare's writing to generate contextually appropriate word predictions.

### Key Features

- **Real-time Inference:** Instant predictions with cached model loading
- **Generation Modes:** Single word prediction or multi-word sequence generation
- **Sampling Controls:** Temperature and Top-K sampling for creative variation
- **Production-Ready:** Clean error handling, input validation, and edge case management
- **Professional UI:** Minimal, recruiter-friendly Streamlit interface

---

## 🏗️ Architecture

```
Input Text → Preprocessing → Tokenization → Padding → LSTM → Softmax → Predicted Word
              (lowercase)     (word-level)   (seq=5)   (RNN)   (vocab)
```

### Model Specifications

| Component | Details |
|-----------|---------|
| **Tokenization** | Word-level with Keras Tokenizer |
| **Vocabulary Size** | 5,000 words |
| **Context Length** | 5 tokens (sliding window) |
| **Embedding** | Trainable embedding layer |
| **Sequence Model** | Single-layer LSTM |
| **Output** | Softmax over vocabulary |
| **Loss Function** | Sparse categorical cross-entropy |

---

## 📁 Project Structure

```
next-word-prediction-lstm/
├── app/
│   ├── app.py              # Streamlit application
│   └── requirements.txt    # Python dependencies
├── model/
│   ├── next_word_lstm.h5   # Trained Keras model
│   └── tokenizer.pkl       # Fitted tokenizer
├── notebook/
│   └── training.ipynb      # Model training notebook (optional)
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/next-word-prediction-lstm.git
   cd next-word-prediction-lstm
   ```

2. **Install dependencies:**
   ```bash
   pip install -r app/requirements.txt
   ```

3. **Ensure model artifacts exist:**
   - Place `next_word_lstm.h5` in `model/`
   - Place `tokenizer.pkl` in `model/`

4. **Run the application:**
   ```bash
   streamlit run app/app.py
   ```

5. **Open browser:** Navigate to `http://localhost:8501`

---

## 💡 Usage

### Single Word Prediction
Enter a text phrase and receive the most likely next word.

```
Input:  "to be or not to"
Output: "be"
```

### Sequence Generation
Generate multiple words in succession using looped prediction.

```
Input:  "the king shall"
Output: "the king shall be a man of the world and the..."
```

### Sampling Parameters

| Parameter | Effect |
|-----------|--------|
| **Temperature = 0.5** | More confident, repetitive predictions |
| **Temperature = 1.0** | Balanced creativity (default) |
| **Temperature = 1.5** | More random, creative outputs |
| **Top-K = 10** | Sample from top 10 candidates only |

---

## 🧠 Technical Implementation

### Preprocessing Pipeline

The inference pipeline applies identical transformations used during training:

```python
def preprocess_text(text: str) -> str:
    return text.lower().strip()
```

### OOV Handling

Out-of-vocabulary predictions are blocked during inference by zeroing the `<OOV>` token probability and renormalizing:

```python
if oov_index is not None:
    predictions[oov_index] = 0.0
    predictions = predictions / np.sum(predictions)
```

### Context Window

The model uses a sliding window of 5 tokens. Longer inputs are truncated to the last 5 words:

```python
sequence = sequence[-CONTEXT_LENGTH:]
padded = pad_sequences([sequence], maxlen=CONTEXT_LENGTH, padding="pre")
```

---

## 📊 Skills Demonstrated

This project showcases the following AI/ML engineering competencies:

### 1. **Sequence Modeling Fundamentals**
- Understanding of RNN/LSTM architectures for sequential data
- Proper handling of variable-length inputs via padding
- Context window management for fixed-length models

### 2. **NLP Preprocessing**
- Word-level tokenization with vocabulary constraints
- Text normalization matching training distribution
- OOV token handling strategies

### 3. **Inference Optimization**
- Model caching to avoid reloading (`@st.cache_resource`)
- Efficient numpy operations for probability manipulation
- Graceful degradation for edge cases

### 4. **Sampling Strategies**
- Temperature-based probability scaling
- Top-K filtering for controlled randomness
- Greedy vs. stochastic decoding comparison

### 5. **Production Engineering**
- Clean separation of concerns (config, inference, UI)
- Comprehensive error handling and user feedback
- Type hints and docstrings for maintainability

---

## ⚠️ Limitations

| Limitation | Explanation |
|------------|-------------|
| **Domain-Specific** | Trained on Shakespeare only — outputs will reflect Early Modern English |
| **Limited Context** | 5-word window cannot capture long-range dependencies |
| **Fixed Vocabulary** | Words outside the 5,000 top tokens are unknown |
| **No Attention** | Single LSTM layer lacks the expressiveness of transformer models |
| **Word-Level Only** | Cannot handle subword tokenization or rare word morphology |

---

## 🔮 Future Improvements

- [ ] Beam search decoding for higher-quality sequences
- [ ] Subword tokenization (BPE/WordPiece) for better OOV handling
- [ ] Bi-directional context with masked language modeling
- [ ] Attention mechanism for longer dependencies
- [ ] Model quantization for faster inference

---

## 📚 References

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Christopher Olah
- [The Unreasonable Effectiveness of RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) - Andrej Karpathy
- [Keras Text Classification](https://keras.io/examples/nlp/) - Official Keras Examples

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

Developed as a portfolio project demonstrating practical NLP and deep learning skills. Designed for technical review by recruiters and engineering teams.

---

*This project intentionally uses a simple LSTM architecture to demonstrate foundational language modeling concepts without the complexity of modern transformer-based approaches.*
