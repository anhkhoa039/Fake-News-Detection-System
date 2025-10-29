import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords once (you can comment this line after first run)
nltk.download('stopwords')

# Load CSV files
fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")

# Add labels: 0 for fake, 1 for real
fake_df['label'] = 0
real_df['label'] = 1

# Combine datasets
news_df = pd.concat([fake_df, real_df], ignore_index=True)

# Prepare stopwords list
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # lowercase
    text = re.sub('[^a-z\s]', '', text)  # remove punctuation/digits
    tokens = text.split()  # tokenize
    filtered_tokens = [word for word in tokens if word not in stop_words]  # remove stopwords
    return ' '.join(filtered_tokens)

# Apply preprocessing on 'text' column
news_df['processed_text'] = news_df['text'].apply(preprocess_text)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the processed text
X = vectorizer.fit_transform(news_df['processed_text']).toarray()

# Labels
y = news_df['label'].values

# Print shapes
print("Feature matrix shape:", X.shape)  # (samples, features)
print("Labels shape:", y.shape)           # (samples,)
