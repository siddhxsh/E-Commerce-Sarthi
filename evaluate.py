import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------
# CONFIG
# ----------------------------
TEXT_COLUMN = "text"
LABEL_COLUMN = "sentiment"
FILE_NAME = "flipkart_product_cleaned.csv"   # 🔴 same file as train.py

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

# ----------------------------
# SAME SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# LOAD MODEL
# ----------------------------
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
model = joblib.load(os.path.join(MODEL_DIR, "logistic_model.joblib"))

# ----------------------------
# PREDICT
# ----------------------------
X_test_tfidf = tfidf.transform(X_test)
y_pred = model.predict(X_test_tfidf)

# ----------------------------
# METRICS
# ----------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------
# CONFUSION MATRIX
# ----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
