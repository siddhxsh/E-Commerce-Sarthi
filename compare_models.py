import os
import pandas as pd
import joblib
import nltk

from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ----------------------------
# DOWNLOAD VADER (FIRST RUN)
# ----------------------------
#nltk.download("vader_lexicon")

# ----------------------------
# CONFIG (MUST MATCH OTHERS)
# ----------------------------
TEXT_COLUMN = "text"
LABEL_COLUMN = "sentiment"
FILE_NAME = "flipkart_product_cleaned.csv"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", FILE_NAME)
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(DATA_PATH)
df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)

X = df[TEXT_COLUMN]
y = df[LABEL_COLUMN]

# SAME SPLIT FOR BOTH MODELS
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================================
# 1️⃣ VADER BASELINE
# ==========================================================
sia = SentimentIntensityAnalyzer()

def vader_predict(text):
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

y_pred_vader = X_test.apply(vader_predict)

vader_acc = accuracy_score(y_test, y_pred_vader)
vader_prec, vader_rec, vader_f1, _ = precision_recall_fscore_support(
    y_test, y_pred_vader, average="weighted"
)

# ==========================================================
# 2️⃣ ML MODEL (TF-IDF + Logistic Regression)
# ==========================================================
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
model = joblib.load(os.path.join(MODEL_DIR, "logistic_model.joblib"))

X_test_tfidf = tfidf.transform(X_test)
y_pred_ml = model.predict(X_test_tfidf)

ml_acc = accuracy_score(y_test, y_pred_ml)
ml_prec, ml_rec, ml_f1, _ = precision_recall_fscore_support(
    y_test, y_pred_ml, average="weighted"
)

# ==========================================================
# 3️⃣ COMPARISON TABLE
# ==========================================================
comparison = pd.DataFrame({
    "Model": ["VADER (Baseline)", "TF-IDF + Logistic Regression"],
    "Accuracy": [vader_acc, ml_acc],
    "Precision": [vader_prec, ml_prec],
    "Recall": [vader_rec, ml_rec],
    "F1-score": [vader_f1, ml_f1]
})

print("\nMODEL COMPARISON\n")
print(comparison)

#Accuracy Bar Plot

import matplotlib.pyplot as plt

models = ["VADER", "TF-IDF + LR"]
accuracies = [vader_acc, ml_acc]

plt.figure(figsize=(6,4))
plt.bar(models, accuracies)
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.show()

# F1-Score Bar Plot

f1_scores = [vader_f1, ml_f1]

plt.figure(figsize=(6,4))
plt.bar(models, f1_scores)
plt.ylim(0, 1)
plt.ylabel("F1-score")
plt.title("Model F1-score Comparison")
plt.show()
