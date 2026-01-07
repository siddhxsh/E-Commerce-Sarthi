# This file:
# Uses the same dataset
# Uses the same train/test split
# Produces same metrics as ML model

import os
import pandas as pd
import nltk

from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------
# DOWNLOAD VADER (FIRST TIME)
# ----------------------------
nltk.download("vader_lexicon")

# ----------------------------
# CONFIG
# ----------------------------
TEXT_COLUMN = "text"        # SAME as ML model
LABEL_COLUMN = "sentiment"
FILE_NAME = "flipkart_product_cleaned.csv"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", FILE_NAME)

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(DATA_PATH)

# Clean text (minimal)
df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)

X = df[TEXT_COLUMN]
y = df[LABEL_COLUMN]

# ----------------------------
# SAME TRAIN–TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# VADER SENTIMENT ANALYZER
# ----------------------------
sia = SentimentIntensityAnalyzer()

def vader_predict(text):
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# ----------------------------
# PREDICT ON TEST SET
# ----------------------------
y_pred = X_test.apply(vader_predict)

# ----------------------------
# EVALUATION
# ----------------------------
print("\nVADER BASELINE RESULTS\n")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
