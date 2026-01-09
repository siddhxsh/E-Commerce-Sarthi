import os
import re
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="E-commerce Sentiment Analysis",
    layout="wide"
)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    model = joblib.load(os.path.join(MODEL_DIR, "logistic_model.joblib"))
    return tfidf, model

tfidf, model = load_model()

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------- PREDICTION ----------------
def predict_sentiment(text):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    probs = model.predict_proba(vector)[0]
    classes = model.classes_

    prediction = classes[probs.argmax()]
    confidence = probs.max()

    # Confidence-based safety
    if confidence < 0.55:
        return "Neutral"

    return prediction

# ---------------- CACHE SENTIMENT ----------------
@st.cache_data(show_spinner=False)
def compute_sentiment(df, text_col):
    df["Sentiment"] = df[text_col].apply(predict_sentiment)
    return df

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align:center;'>E-commerce Sentiment & Review Analysis</h1>
    <p style='text-align:center; font-size:18px; color:gray; max-width:900px; margin:auto;'>
    Upload an e-commerce review dataset and analyze customer sentiment using a trained
    <b>TF-IDF + Logistic Regression</b> model.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- FILE UPLOAD ----------------
st.subheader("📂 Upload Review Dataset (CSV)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# ---------------- MAIN LOGIC ----------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # 🔴 EXPLICIT REVIEW COLUMN (IMPORTANT)
    TEXT_COLUMN = "text"   # must match training

    if TEXT_COLUMN not in df.columns:
        st.error(f"Column '{TEXT_COLUMN}' not found in dataset.")
        st.stop()

    # Clean & deduplicate
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    df = df.drop_duplicates(subset=[TEXT_COLUMN])

    # Compute sentiment
    with st.spinner("Analyzing sentiment..."):
        df = compute_sentiment(df, TEXT_COLUMN)

    # ---------------- SENTIMENT OVERVIEW ----------------
    st.markdown("## 📊 Sentiment Overview")
    sentiment_counts = df["Sentiment"].value_counts()

    c1, c2, c3 = st.columns(3)
    c1.metric("Positive", sentiment_counts.get("Positive", 0))
    c2.metric("Neutral", sentiment_counts.get("Neutral", 0))
    c3.metric("Negative", sentiment_counts.get("Negative", 0))

    # ---------------- BAR CHART ----------------
    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.index, sentiment_counts.values)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.set_title("Sentiment Distribution")
    st.pyplot(fig)

    # ---------------- SAMPLE REVIEWS ----------------
    st.markdown("## 📝 Sample Reviews per Sentiment")

    def truncate(text, max_len=120):
        return text if len(text) <= max_len else text[:max_len] + "..."

    for sentiment in ["Positive", "Neutral", "Negative"]:
        st.write(f"### {sentiment} Reviews")

        subset = df[df["Sentiment"] == sentiment][TEXT_COLUMN]
        if subset.empty:
            st.write("No reviews available.")
        else:
            samples = subset.sample(
                n=min(3, len(subset)),
                random_state=42
            )
            for review in samples:
                st.write("-", truncate(review))

    # ---------------- KEYWORD EXTRACTION ----------------
    st.markdown("## ❌ Common Issues from Negative Reviews")

    negative_reviews = df[df["Sentiment"] == "Negative"][TEXT_COLUMN]

    words = []
    for review in negative_reviews:
        tokens = re.findall(r"\b[a-z]{4,}\b", review.lower())
        words.extend(tokens)

    common_words = Counter(words).most_common(10)

    if common_words:
        keyword_df = pd.DataFrame(common_words, columns=["Keyword", "Frequency"])
        st.table(keyword_df)
    else:
        st.write("No common negative keywords found.")

    # ---------------- INFO ----------------
    st.markdown("## ⚙️ Model Info")
    st.info(
        "This system uses a TF-IDF + Logistic Regression model with class balancing "
        "and confidence-aware prediction to handle large-scale review datasets."
    )

else:
    st.info("👈 Upload a CSV file to begin analysis.")
