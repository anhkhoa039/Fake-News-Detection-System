from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split

# Sample dataset
texts = [
    "The government launched a new healthcare scheme for rural families.",
    "India signed a trade agreement with the EU to boost exports.",
    "NASA launched a new satellite to study climate change.",
    "Vaccination drives have reached over 10 million children.",
    "New education policy aims to make coding compulsory in schools.",
    "Aliens have arrived to colonize Earth disguised as tourists.",
    "The moon is expected to explode by next week, NASA confirms.",
    "Scientists create immortality pills from ancient crystals.",
    "Time travelers spotted leaving a black hole in Antarctica.",
    "New species of dragons discovered under the Himalayan glaciers."
]

labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Real, 1 = Fake

# Tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

# Train/test split
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

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

# Model
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Compatible training arguments (no evaluation_strategy)
training_args = TrainingArguments(
    output_dir="./bert_fake_news_model",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=1000  # prevents unnecessary checkpoint saving
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train
trainer.train()

# Save model
model.save_pretrained("./bert_fake_news_model")
tokenizer.save_pretrained("./bert_fake_news_model")

print("✅ Training completed and model saved to './bert_fake_news_model'")
