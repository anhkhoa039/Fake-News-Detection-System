import os
import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_CSV = os.path.join(
    PROJECT_ROOT, "dataset", "fake_news_detection(FakeNewsNet)", "fnn_train.csv"
)
TEST_CSV = os.path.join(
    PROJECT_ROOT, "dataset", "fake_news_detection(FakeNewsNet)", "fnn_test.csv"
)
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

TEXT_COL = "fullText_based_content"
LABEL_COL = "label_fnn"

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_label(value) -> int:
    s = str(value).strip().lower()
    if s in {"1", "fake", "false", "f"}:
        return 1
    if s in {"0", "real", "true", "t"}:
        return 0
    # Try numeric fallback
    try:
        return int(float(s))
    except Exception:
        raise ValueError(f"Unrecognized label value: {value}")

# Load data
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# Keep only necessary columns and drop missing
train_df = train_df[[TEXT_COL, LABEL_COL]].dropna()
test_df = test_df[[TEXT_COL, LABEL_COL]].dropna()

# Basic cleaning
train_df[TEXT_COL] = train_df[TEXT_COL].apply(clean_text)
test_df[TEXT_COL] = test_df[TEXT_COL].apply(clean_text)

# Ensure labels are integers (0/1), handling 'fake'/'real' strings
train_df[LABEL_COL] = train_df[LABEL_COL].apply(normalize_label).astype(int)
test_df[LABEL_COL] = test_df[LABEL_COL].apply(normalize_label).astype(int)

# Feature extraction (TF-IDF)
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_df[TEXT_COL])
y_train = train_df[LABEL_COL].values

X_test = vectorizer.transform(test_df[TEXT_COL])
y_test = test_df[LABEL_COL].values

# Train model
model = LogisticRegression(max_iter=2000, n_jobs=-1)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer
with open(os.path.join(MODEL_DIR, 'new_data_logistic_model.pkl'), 'wb') as f:
    pickle.dump(model, f)

with open(os.path.join(MODEL_DIR, 'new_data_tfidf_vectorizer.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model (new_data_logistic_model.pkl) and vectorizer (new_data_tfidf_vectorizer.pkl) saved to 'model/'")
