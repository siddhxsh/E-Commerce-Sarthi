import os
import joblib
import re
import pandas as pd

# ----------------------------
# CONFIG (MUST MATCH train.py)
# ----------------------------
TEXT_COLUMN = "text"
FILE_NAME = "flipkart_product_cleaned.csv"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ----------------------------
# LOAD MODEL & VECTORIZER
# ----------------------------
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
model = joblib.load(os.path.join(MODEL_DIR, "logistic_model.joblib"))

# ----------------------------
# TEXT PREPROCESSING FUNCTION
# ----------------------------
def clean_text(text: str) -> str:
    """
    Must match training preprocessing
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text)      # remove URLs
    text = re.sub(r"<.*?>", "", text)         # remove HTML
    text = re.sub(r"[^a-z\s]", "", text)      # remove special chars
    text = re.sub(r"\s+", " ", text).strip()  # normalize spaces
    return text

# ----------------------------
# SINGLE PREDICTION PIPELINE
# ----------------------------
def predict_sentiment(text: str):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])   # ⚠️ transform ONLY
    prediction = model.predict(vector)[0]
    confidence = model.predict_proba(vector).max()
    return prediction, confidence

# ----------------------------
# BATCH PREDICTION PIPELINE
# ----------------------------
def predict_batch(texts):
    cleaned = [clean_text(t) for t in texts]
    vectors = tfidf.transform(cleaned)
    preds = model.predict(vectors)
    probs = model.predict_proba(vectors).max(axis=1)
    return list(zip(preds, probs))

# ----------------------------
# DEMO (RUN THIS FILE DIRECTLY)
# ----------------------------
if __name__ == "__main__":
    samples = [
        "This product is amazing, totally worth the price",
        "Worst quality, very disappointed",
        "It is okay, not great but not bad"
    ]

    for s in samples:
        sentiment, conf = predict_sentiment(s)
        print(f"Text: {s}")
        print(f"Prediction: {sentiment}, Confidence: {conf:.2f}")
        print("-" * 50)
