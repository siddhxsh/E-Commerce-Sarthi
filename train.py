import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ----------------------------
# CONFIG
# ----------------------------
TEXT_COLUMN = "text"        # CHANGE ONLY IF NEEDED
LABEL_COLUMN = "sentiment"
FILE_NAME = "flipkart_product_cleaned.csv"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", FILE_NAME)
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print(df[LABEL_COLUMN].value_counts())

# ----------------------------
# FORCE CLEAN TEXT COLUMN (CRITICAL)
# ----------------------------
print("NaN count BEFORE cleaning:", df[TEXT_COLUMN].isna().sum())

df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)

print("NaN count AFTER cleaning:", df[TEXT_COLUMN].isna().sum())

# ----------------------------
# FEATURE / LABEL SPLIT
# ----------------------------
X = df[TEXT_COLUMN]
y = df[LABEL_COLUMN]

# ----------------------------
# TRAIN / TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# TF-IDF VECTORIZATION
# ----------------------------
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=40000,
    min_df=5,
    max_df=0.9,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)

print("TF-IDF Train Shape:", X_train_tfidf.shape)

# ----------------------------
# MODEL TRAINING
# ----------------------------
model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    n_jobs=-1,
    solver="lbfgs"
)

model.fit(X_train_tfidf, y_train)

# ----------------------------
# SAVE MODEL & VECTORIZER
# ----------------------------
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
joblib.dump(model, os.path.join(MODEL_DIR, "logistic_model.joblib"))

print("Training completed successfully.")
