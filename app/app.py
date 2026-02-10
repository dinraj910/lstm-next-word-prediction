"""
Next Word Prediction - Streamlit Application
=============================================
A production-ready inference application for LSTM-based language modeling.
Uses a pre-trained model on Shakespeare corpus for next word prediction.

Author: AI/ML Portfolio Project
"""

import streamlit as st
import numpy as np
import pickle
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_PATH = Path(__file__).parent.parent / "model" / "next_word_lstm.h5"
TOKENIZER_PATH = Path(__file__).parent.parent / "model" / "tokenizer.pkl"
CONTEXT_LENGTH = 5  # Must match training configuration
OOV_TOKEN = "<OOV>"


# =============================================================================
# MODEL LOADING (Cached for performance)
# =============================================================================
@st.cache_resource
def load_artifacts():
    """
    Load the trained LSTM model and tokenizer.
    Uses Streamlit caching to avoid reloading on each interaction.
    """
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    if not TOKENIZER_PATH.exists():
        st.error(f"Tokenizer file not found: {TOKENIZER_PATH}")
        st.stop()
    
    model = load_model(str(MODEL_PATH))
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    
    return model, tokenizer


# =============================================================================
# INFERENCE UTILITIES
# =============================================================================
def preprocess_text(text: str) -> str:
    """
    Apply the same preprocessing used during training.
    """
    return text.lower().strip()


def predict_next_word(
    model,
    tokenizer,
    input_text: str,
    temperature: float = 1.0,
    top_k: int = 0
) -> str:
    """
    Predict the next word given an input text.
    
    Args:
        model: Trained Keras LSTM model
        tokenizer: Fitted Keras Tokenizer
        input_text: User input text
        temperature: Sampling temperature (1.0 = normal, <1.0 = more confident, >1.0 = more random)
        top_k: If > 0, sample from top-k predictions only
    
    Returns:
        Predicted next word as string
    """
    # Preprocess input
    processed_text = preprocess_text(input_text)
    
    # Handle empty input
    if not processed_text:
        return None
    
    # Tokenize and pad sequence
    sequence = tokenizer.texts_to_sequences([processed_text])[0]
    
    # Handle case where no valid tokens found
    if not sequence:
        return None
    
    # Take only the last CONTEXT_LENGTH tokens (sliding window)
    sequence = sequence[-CONTEXT_LENGTH:]
    
    # Pad sequence to fixed length
    padded = pad_sequences([sequence], maxlen=CONTEXT_LENGTH, padding="pre")
    
    # Get model predictions (logits before softmax or probabilities)
    predictions = model.predict(padded, verbose=0)[0]
    
    # Apply temperature scaling
    if temperature != 1.0:
        predictions = np.log(predictions + 1e-10) / temperature
        predictions = np.exp(predictions)
        predictions = predictions / np.sum(predictions)
    
    # Build reverse word index
    word_index = tokenizer.word_index
    index_word = {v: k for k, v in word_index.items()}
    
    # Get OOV token index if it exists
    oov_index = word_index.get(OOV_TOKEN.lower(), None)
    
    # Block OOV token from being predicted
    if oov_index is not None and oov_index < len(predictions):
        predictions[oov_index] = 0.0
        predictions = predictions / (np.sum(predictions) + 1e-10)  # Renormalize
    
    # Also block index 0 (padding token)
    predictions[0] = 0.0
    predictions = predictions / (np.sum(predictions) + 1e-10)
    
    # Apply top-k sampling
    if top_k > 0:
        top_indices = np.argsort(predictions)[-top_k:]
        mask = np.zeros_like(predictions)
        mask[top_indices] = predictions[top_indices]
        predictions = mask / (np.sum(mask) + 1e-10)
        predicted_index = np.random.choice(len(predictions), p=predictions)
    else:
        # Greedy decoding (argmax)
        predicted_index = np.argmax(predictions)
    
    # Convert index to word
    predicted_word = index_word.get(predicted_index, None)
    
    return predicted_word


def generate_sequence(
    model,
    tokenizer,
    seed_text: str,
    num_words: int,
    temperature: float = 1.0,
    top_k: int = 0
) -> str:
    """
    Generate a sequence of words using looped prediction.
    
    Args:
        model: Trained Keras LSTM model
        tokenizer: Fitted Keras Tokenizer
        seed_text: Starting text for generation
        num_words: Number of words to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
    
    Returns:
        Generated text sequence
    """
    current_text = seed_text
    generated_words = []
    
    for _ in range(num_words):
        next_word = predict_next_word(
            model, tokenizer, current_text, temperature, top_k
        )
        
        if next_word is None:
            break
        
        generated_words.append(next_word)
        current_text = current_text + " " + next_word
    
    return " ".join(generated_words)


# =============================================================================
# STREAMLIT UI
# =============================================================================
def main():
    # Page configuration
    st.set_page_config(
        page_title="Next Word Prediction",
        page_icon="📝",
        layout="centered"
    )
    
    # Header
    st.title("📝 Next Word Prediction")
    st.markdown(
        "An LSTM-based language model trained on Shakespeare's works. "
        "Enter text below to predict the next word."
    )
    
    st.divider()
    
    # Load model and tokenizer
    try:
        model, tokenizer = load_artifacts()
        vocab_size = len(tokenizer.word_index) + 1
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        st.info(
            "Please ensure the following files exist:\n"
            f"- `{MODEL_PATH}`\n"
            f"- `{TOKENIZER_PATH}`"
        )
        st.stop()
    
    # Sidebar for advanced options
    with st.sidebar:
        st.header("⚙️ Generation Settings")
        
        generation_mode = st.radio(
            "Mode",
            ["Single Word", "Multiple Words"],
            help="Single word predicts just the next word. Multiple words generates a sequence."
        )
        
        if generation_mode == "Multiple Words":
            num_words = st.slider(
                "Words to generate",
                min_value=1,
                max_value=50,
                value=10,
                help="Number of words to generate in sequence"
            )
        else:
            num_words = 1
        
        st.divider()
        
        use_sampling = st.checkbox(
            "Enable sampling",
            value=False,
            help="Use probabilistic sampling instead of greedy selection"
        )
        
        if use_sampling:
            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Higher = more random, Lower = more focused"
            )
            top_k = st.slider(
                "Top-K",
                min_value=0,
                max_value=50,
                value=10,
                help="Sample from top K predictions (0 = disabled)"
            )
        else:
            temperature = 1.0
            top_k = 0
        
        st.divider()
        st.caption("Model Information")
        st.markdown(f"- **Vocab Size:** {vocab_size:,}")
        st.markdown(f"- **Context Length:** {CONTEXT_LENGTH}")
        st.markdown("- **Architecture:** LSTM")
    
    # Main input area
    input_text = st.text_area(
        "Enter your text:",
        placeholder="e.g., To be or not to",
        height=100,
        help="Type a phrase and the model will predict what comes next"
    )
    
    # Predict button
    if st.button("Predict", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("Please enter some text to generate predictions.")
        else:
            with st.spinner("Generating..."):
                if generation_mode == "Single Word":
                    result = predict_next_word(
                        model, tokenizer, input_text, temperature, top_k
                    )
                    
                    if result:
                        st.success("Prediction complete!")
                        st.markdown("### Next Word:")
                        st.markdown(f"## `{result}`")
                        
                        # Show complete sentence
                        st.markdown("### Complete Phrase:")
                        st.markdown(f"> {input_text.strip()} **{result}**")
                    else:
                        st.warning(
                            "Could not generate a prediction. "
                            "The input may contain only unknown words."
                        )
                else:
                    result = generate_sequence(
                        model, tokenizer, input_text, num_words, temperature, top_k
                    )
                    
                    if result:
                        st.success("Generation complete!")
                        st.markdown("### Generated Text:")
                        st.markdown(f"> {input_text.strip()} **{result}**")
                    else:
                        st.warning(
                            "Could not generate text. "
                            "The input may contain only unknown words."
                        )
    
    # Example prompts
    st.divider()
    st.markdown("### 💡 Try these examples:")
    
    col1, col2, col3 = st.columns(3)
    examples = [
        "to be or not",
        "the king shall",
        "love is a"
    ]
    
    for col, example in zip([col1, col2, col3], examples):
        with col:
            if st.button(example, use_container_width=True):
                st.session_state["example_text"] = example
                st.rerun()
    
    # Handle example selection
    if "example_text" in st.session_state:
        st.info(f"Selected: '{st.session_state['example_text']}' - paste it above and click Predict!")
        del st.session_state["example_text"]
    
    # Footer with limitations
    st.divider()
    with st.expander("ℹ️ About this model"):
        st.markdown("""
        **Limitations:**
        - Trained exclusively on Shakespeare's works — predictions reflect Early Modern English style
        - Vocabulary limited to ~5,000 most frequent words
        - Context window of 5 words — longer dependencies may not be captured
        - Single-layer LSTM — simpler than modern transformer architectures
        
        **Best for:**
        - Demonstrating fundamental language modeling concepts
        - Shakespearean-style text generation
        - Educational purposes and portfolio demonstration
        
        **Architecture:**
        - Word-level tokenization with trainable embeddings
        - Single LSTM layer for sequence modeling
        - Softmax output over vocabulary for next-word probability distribution
        """)


if __name__ == "__main__":
    main()
