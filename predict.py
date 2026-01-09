import os
import joblib
import re

# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ----------------------------
# LOAD MODEL
# ----------------------------
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
model = joblib.load(os.path.join(MODEL_DIR, "logistic_model.joblib"))

# ----------------------------
# CLEAN TEXT
# ----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------------
# PREDICTION
# ----------------------------
def predict_sentiment(text):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])

    probs = model.predict_proba(vector)[0]
    classes = model.classes_

    best_idx = probs.argmax()
    prediction = classes[best_idx]
    confidence = probs[best_idx]

    negative_phrases = [
        "used before",
        "already used",
        "second hand",
        "damaged",
        "broken",
        "defective",
        "missing"
    ]

    for phrase in negative_phrases:
        if phrase in cleaned:
            return "Negative", confidence

    if confidence < 0.55:
        return "Negative", confidence

    return prediction, confidence

# ----------------------------
# DEMO
# ----------------------------
if __name__ == "__main__":
    samples = [
        "great product excellent quality",
        "very bad product waste of money",
        "okay product nothing special",
        "for this proce the product is okay",
        "the price is affordable at the expense of quality",
        "the packaging was not damaged, but the product is used before delivery"
    ]

    for s in samples:
        sentiment, conf = predict_sentiment(s)app
        print(f"Text: {s}")
        print(f"Prediction: {sentiment}, Confidence: {conf:.2f}")
        print("-" * 50)
