import streamlit as st
import numpy as np
import re
import joblib
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# ✅ MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💬",
    layout="centered"
)

# -------------------------
# 1. Load saved artifacts
# -------------------------

@st.cache_resource
def load_classical_artifacts():
    best_model = joblib.load("best_classical_model.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return best_model, tfidf_vectorizer, label_encoder

@st.cache_resource
def load_bilstm_artifacts():
    try:
        bilstm_model = load_model("bilstm_sentiment_model.h5")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        label_encoder = joblib.load("label_encoder.pkl")
        return bilstm_model, tokenizer, label_encoder
    except Exception:
        return None, None, None

# -------------------------
# 2. Text preprocessing
# -------------------------

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www.\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# -------------------------
# 3. Prediction functions
# -------------------------

def predict_with_classical(text: str):
    best_model, tfidf_vectorizer, label_encoder = load_classical_artifacts()
    cleaned = preprocess_text(text)
    vec = tfidf_vectorizer.transform([cleaned])
    pred_enc = best_model.predict(vec)[0]
    pred_label = label_encoder.inverse_transform([pred_enc])[0]
    return pred_label

def predict_with_bilstm(text: str, max_len: int = 50):
    bilstm_model, tokenizer, label_encoder = load_bilstm_artifacts()
    if bilstm_model is None:
        return None, "BiLSTM model not found!"

    cleaned = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    proba = bilstm_model.predict(pad)[0]
    pred_enc = int(np.argmax(proba))
    pred_label = label_encoder.inverse_transform([pred_enc])[0]
    return pred_label, None

# -------------------------
# 4. STREAMLIT UI
# -------------------------

st.title("💬 Sentiment Analysis using ML & Deep Learning")

st.write("Enter any sentence below and the model will predict its sentiment.")

model_choice = st.radio(
    "Choose Model:",
    ("Classical ML (TF-IDF)", "Deep Learning (BiLSTM)")
)

user_text = st.text_area("Enter your text here 👇", height=150)

if st.button("Analyze Sentiment"):
    if not user_text.strip():
        st.warning("Please enter some text!")
    else:
        with st.spinner("Analyzing..."):
            if model_choice == "Classical ML (TF-IDF)":
                result = predict_with_classical(user_text)
                st.success(f"Predicted Sentiment: **{result}**")
            else:
                result, error = predict_with_bilstm(user_text)
                if error:
                    st.error(error)
                else:
                    st.success(f"Predicted Sentiment: **{result}**")

st.markdown("---")
st.caption("Built using Machine Learning & Deep Learning")
