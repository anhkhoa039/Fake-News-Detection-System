import nltk
from nltk.corpus import stopwords
import re
import pickle

# Download stopwords if needed
nltk.download('stopwords')

# Load the saved model and vectorizer
with open('fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-z\s]', '', text)
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def predict_news(text):
    processed = preprocess_text(text)
    vectorized = vectorizer.transform([processed]).toarray()
    prediction = model.predict(vectorized)[0]
    return "Real News" if prediction == 1 else "Fake News"

# Example usage:
if __name__ == "__main__":
    sample_text = input("Enter a news article text: ")
    result = predict_news(sample_text)
    print(f"Prediction: {result}")
