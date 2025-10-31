from transformers import BertTokenizer, BertForSequenceClassification
import torch

def predict_news(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return "Real News" if predicted_class == 0 else "Fake News"

def main():
    print("🔍 Fake News Detector (BERT-based)")
    print("Type your news article below. Type 'exit' to quit.\n")

    model_path = "./bert_fake_news_model"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    while True:
        text = input("Enter a news article text: ")
        if text.lower() == "exit":
            print("Exiting prediction tool.")
            break
        prediction = predict_news(text, model, tokenizer)
        print(f"Prediction: {prediction}\n")

if __name__ == "__main__":
    main()
