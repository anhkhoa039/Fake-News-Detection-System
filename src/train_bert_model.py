import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_CSV = os.path.join(
    PROJECT_ROOT, "dataset", "fake_news_detection(FakeNewsNet)", "fnn_train.csv"
)
MODEL_DIR = os.path.join(PROJECT_ROOT, "bert_fake_news_model")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TEXT_COL = "fullText_based_content"
LABEL_COL = "label_fnn"

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

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

# Load and prepare data
df = pd.read_csv(TRAIN_CSV, usecols=[TEXT_COL, LABEL_COL]).dropna()
df[LABEL_COL] = df[LABEL_COL].apply(normalize_label).astype(int)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df[TEXT_COL].tolist(),
    df[LABEL_COL].tolist(),
    test_size=0.1,
    random_state=42,
    stratify=df[LABEL_COL].tolist()
)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors="pt")

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    warmup_steps=0,
    weight_decay=0.01,
    logging_dir=LOG_DIR,
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print("✅ Training completed and model saved to 'bert_fake_news_model'")
