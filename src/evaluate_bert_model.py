import os
import json
from typing import List

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification


# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_CSV = os.path.join(
    PROJECT_ROOT, "dataset", "fake_news_detection(FakeNewsNet)", "fnn_test.csv"
)
MODEL_DIR = os.path.join(PROJECT_ROOT, "bert_fake_news_model")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(LOG_DIR, exist_ok=True)

TEXT_COL = "fullText_based_content"
LABEL_COL = "label_fnn"


def normalize_label(value) -> int:
    s = str(value).strip().lower()
    if s in {"1", "fake", "false", "f"}:
        return 1
    if s in {"0", "real", "true", "t"}:
        return 0
    try:
        return int(float(s))
    except Exception:
        raise ValueError(f"Unrecognized label value: {value}")


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=[TEXT_COL, LABEL_COL]).dropna()
    df[LABEL_COL] = df[LABEL_COL].apply(normalize_label).astype(int)
    return df


def tokenize_texts(tokenizer: BertTokenizer, texts: List[str]):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )


@torch.no_grad()
def predict(model: BertForSequenceClassification, inputs) -> List[int]:
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = 32
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    preds: List[int] = []
    for start in range(0, input_ids.size(0), batch_size):
        end = start + batch_size
        batch = {
            "input_ids": input_ids[start:end].to(device),
            "attention_mask": attention_mask[start:end].to(device),
        }
        outputs = model(**batch)
        logits = outputs.logits
        batch_preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
        preds.extend(batch_preds)
    return preds


def main() -> None:
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}. Train the model first."
        )

    print("Loading tokenizer and model ...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)

    print("Loading test data ...")
    test_df = load_data(TEST_CSV)
    texts = test_df[TEXT_COL].tolist()
    labels = test_df[LABEL_COL].tolist()

    print("Tokenizing ...")
    encodings = tokenize_texts(tokenizer, texts)

    print("Running inference ...")
    y_pred = predict(model, encodings)

    print("Computing metrics ...")
    acc = accuracy_score(labels, y_pred)
    f1 = f1_score(labels, y_pred, average="macro")
    prec = precision_score(labels, y_pred, average="macro", zero_division=0)
    rec = recall_score(labels, y_pred, average="macro", zero_division=0)
    report = classification_report(labels, y_pred, digits=4)

    metrics = {
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": prec,
        "recall_macro": rec,
    }

    print("\n===== BERT Test Metrics =====")
    print(json.dumps(metrics, indent=2))
    print("\nClassification Report:\n")
    print(report)

    # Save metrics
    metrics_path = os.path.join(LOG_DIR, "bert_test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()


