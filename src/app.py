from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import os
from datetime import datetime
import csv
import pickle
import nltk
from nltk.corpus import stopwords
import re
from textblob import TextBlob
import requests
import time
import torch
from transformers import BertTokenizer, BertForSequenceClassification
try:
    from sklearn.exceptions import NotFittedError
except Exception:
    NotFittedError = Exception
import random
import random as pyrandom
import json
import pandas as pd
import ast

# Resolve project root (one level up from src)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, 'templates')
STATIC_DIR = os.path.join(PROJECT_ROOT, 'static')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
BERT_MODEL_DIR = os.path.join(PROJECT_ROOT, 'bert_fake_news_model')

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.secret_key = "your_secret_key_here"  # Replace with a secure key

# Download stopwords if needed
nltk.download('stopwords', quiet=True)

# Helper to open a file from model dir or project root fallback

def open_model_file(filename: str, mode: str = 'rb'):
    model_path = os.path.join(MODEL_DIR, filename)
    root_path = os.path.join(PROJECT_ROOT, filename)
    if os.path.exists(model_path):
        return open(model_path, mode)
    return open(root_path, mode)

# Load the saved model and vectorizer
try:
    with open_model_file('fake_news_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open_model_file('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("✅ Model and vectorizer loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading model files: {e}")
    print("🔄 Using fallback prediction method...")
    model = None
    vectorizer = None

stop_words = set(stopwords.words('english'))

# Keywords to highlight
fake_keywords = ['fake', 'hoax', 'conspiracy', 'false', 'rumor', 'scam']

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def analyze_sentiment(text):
    """Analyze sentiment of the text"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.1:
        sentiment = "Positive"
        emoji = "😊"
    elif polarity < -0.1:
        sentiment = "Negative"
        emoji = "😞"
    else:
        sentiment = "Neutral"
        emoji = "😐"

    return {
        'sentiment': sentiment,
        'emoji': emoji,
        'polarity': round(polarity, 3),
        'subjectivity': round(subjectivity, 3)
    }

def detect_news_source(text):
    """Detect potential news source patterns"""
    sources = {
        'CNN': ['cnn', 'cnn.com'],
        'BBC': ['bbc', 'bbc.com', 'bbc news'],
        'Reuters': ['reuters', 'reuters.com'],
        'Associated Press': ['ap', 'associated press', 'ap.org'],
        'Fox News': ['fox news', 'foxnews.com'],
        'New York Times': ['new york times', 'nytimes.com', 'ny times'],
        'Washington Post': ['washington post', 'wapo.com', 'washpost'],
        'Guardian': ['guardian', 'theguardian.com'],
        'Unknown': []
    }

    text_lower = text.lower()
    detected_sources = []

    for source, keywords in sources.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_sources.append(source)

    return detected_sources if detected_sources else ['Unknown']

def analyze_text_features(text):
    """Analyze various text features"""
    words = text.split()
    sentences = text.split('.')

    # Calculate readability metrics
    avg_words_per_sentence = len(words) / len(sentences) if sentences else 0
    avg_chars_per_word = sum(len(word) for word in words) / len(words) if words else 0

    # Detect potential clickbait patterns
    clickbait_patterns = [
        r'you won\'t believe',
        r'shocking',
        r'amazing',
        r'incredible',
        r'must read',
        r'breaking',
        r'exclusive',
        r'revealed',
        r'secret',
        r'truth about'
    ]

    clickbait_score = sum(1 for pattern in clickbait_patterns
                         if re.search(pattern, text.lower()))

    return {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_words_per_sentence': round(avg_words_per_sentence, 1),
        'avg_chars_per_word': round(avg_chars_per_word, 1),
        'clickbait_score': clickbait_score,
        'readability': 'Easy' if avg_words_per_sentence < 15 else 'Moderate' if avg_words_per_sentence < 25 else 'Complex'
    }

def generate_insights(prediction, confidence, sentiment, features):
    """Generate interesting insights about the analysis"""
    insights = []

    if prediction == "Fake News":
        if confidence > 0.8:
            insights.append("🚨 High confidence fake news detected! This article shows strong indicators of misinformation.")
        if features['clickbait_score'] > 2:
            insights.append("⚠️ Contains multiple clickbait patterns commonly found in fake news.")
        if sentiment['subjectivity'] > 0.7:
            insights.append("📝 Highly subjective language detected, which is common in biased or fake content.")
    else:
        if confidence > 0.8:
            insights.append("✅ High confidence real news! This article appears to be from a reliable source.")
        if features['clickbait_score'] == 0:
            insights.append("📰 Professional writing style with no clickbait patterns detected.")
        if sentiment['subjectivity'] < 0.3:
            insights.append("🎯 Objective reporting style typical of credible news sources.")

    if features['readability'] == 'Complex':
        insights.append("📚 Complex sentence structure detected - may require careful reading.")
    elif features['readability'] == 'Easy':
        insights.append("📖 Simple, accessible writing style.")

    return insights

def validate_user_input(text: str):
    """Validate user-provided news text to reduce spam/unacceptable input.

    Returns (is_valid: bool, message: Optional[str])
    """
    if text is None:
        return False, "Please provide some text."

    original = text
    text = text.strip()
    if not text:
        return False, "Input is empty."

    # Minimal content requirements
    # if len(text) < 3:
    #     return False, "Please provide a longer snippet (≥ 20 characters)."

    # # Hard cap to avoid abuse
    # if len(text) > 4000:
    #     return False, "Text too long (≤ 4000 characters)."

    # Basic word count
    words = re.findall(r"[A-Za-z']+", text)
    if len(words) < 3:
        return False, "Please include more context (≥ 5 words)."

    # Excessive repetition
    if re.search(r"(.)\1{7,}", text):
        return False, "Input has excessive character repetition."

    # Too many links
    url_occurrences = len(re.findall(r"https?://|www\\.", text, flags=re.IGNORECASE))
    if url_occurrences >= 3:
        return False, "Too many links in the input."

    # Profanity / unacceptable words (basic list)
    banned_words = {
        'fuck', 'shit', 'bitch', 'asshole', 'bastard', 'cunt', 'nigger', 'faggot', 'slut', 'whore'
    }
    lowered = text.lower()
    if any(bad in lowered for bad in banned_words):
        return False, "Input contains unacceptable language."

    # Script or HTML tags that could be dangerous in logs/templates
    if re.search(r"<\s*script|onerror\s*=|onload\s*=", lowered):
        return False, "HTML/script content is not allowed."

    return True, None

def calculate_user_score(session_history):
    """Calculate user's truth detection score"""
    if not session_history:
        return 0

    total_analyses = len(session_history)
    high_confidence_count = sum(1 for item in session_history if item.get('confidence', 0) > 80)
    fake_detected = sum(1 for item in session_history if item.get('prediction') == 'Fake News')

    # Base score from analyses
    base_score = total_analyses * 10

    # Bonus for high confidence predictions
    confidence_bonus = high_confidence_count * 5

    # Bonus for detecting fake news (more challenging)
    fake_detection_bonus = fake_detected * 15

    return base_score + confidence_bonus + fake_detection_bonus

def get_user_achievements(session_history):
    """Get user achievements based on their activity"""
    achievements = []

    if not session_history:
        return achievements

    total_analyses = len(session_history)
    high_confidence_count = sum(1 for item in session_history if item.get('confidence', 0) > 80)
    fake_detected = sum(1 for item in session_history if item.get('prediction') == 'Fake News')

    # Analysis milestones
    if total_analyses >= 1:
        achievements.append({"name": "🔍 First Analysis", "description": "Analyzed your first news article!"})
    if total_analyses >= 5:
        achievements.append({"name": "📰 News Detective", "description": "Analyzed 5 news articles!"})
    if total_analyses >= 10:
        achievements.append({"name": "🕵️ Truth Seeker", "description": "Analyzed 10 news articles!"})
    if total_analyses >= 25:
        achievements.append({"name": "🎯 Master Analyst", "description": "Analyzed 25 news articles!"})

    # Confidence achievements
    if high_confidence_count >= 3:
        achievements.append({"name": "🎯 Confident Detective", "description": "Made 3 high-confidence predictions!"})
    if high_confidence_count >= 10:
        achievements.append({"name": "💎 Precision Master", "description": "Made 10 high-confidence predictions!"})

    # Fake news detection achievements
    if fake_detected >= 1:
        achievements.append({"name": "🚨 Fake News Hunter", "description": "Detected your first fake news!"})
    if fake_detected >= 5:
        achievements.append({"name": "🛡️ Truth Guardian", "description": "Detected 5 fake news articles!"})
    if fake_detected >= 10:
        achievements.append({"name": "⚔️ Misinformation Slayer", "description": "Detected 10 fake news articles!"})

    return achievements

def highlight_keywords(text):
    for word in fake_keywords:
        text = text.replace(word, f"<mark>{word}</mark>")
        text = text.replace(word.capitalize(), f"<mark>{word.capitalize()}</mark>")
    return text

# -------- BERT inference (lazy-loaded) --------
_bert_tokenizer = None
_bert_model = None

def load_bert_model():
    global _bert_tokenizer, _bert_model
    if _bert_tokenizer is None or _bert_model is None:
        if not os.path.isdir(BERT_MODEL_DIR):
            raise FileNotFoundError(f"BERT model not found at {BERT_MODEL_DIR}")
        _bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_DIR)
        _bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
        _bert_model.eval()
    return _bert_model, _bert_tokenizer

@torch.no_grad()
def predict_news_bert(text: str):
    model, tokenizer = load_bert_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().tolist()
    pred_idx = int(torch.argmax(logits, dim=-1).item())
    prediction = "Real News" if pred_idx == 0 else "Fake News"
    confidence = float(max(probs))

    sentiment = analyze_sentiment(text)
    sources = detect_news_source(text)
    features = analyze_text_features(text)
    insights = generate_insights(prediction, confidence, sentiment, features)

    return {
        'prediction': prediction,
        'confidence': confidence,
        'sentiment': sentiment,
        'sources': sources,
        'features': features,
        'insights': insights
    }

def predict_news(text):
    # Check if model is available
    if model is None or vectorizer is None:
        # Fallback prediction based on heuristics
        return fallback_prediction(text)

    processed = preprocess_text(text)
    try:
        vectorized = vectorizer.transform([processed]).toarray()
    except NotFittedError:
        # Vectorizer not fitted – use fallback to avoid crashing
        return fallback_prediction(text)
    prediction_proba = model.predict_proba(vectorized)[0]
    predicted_class = model.predict(vectorized)[0]
    confidence_score = max(prediction_proba)
    prediction = "Real News" if predicted_class == 1 else "Fake News"

    # Additional analysis
    sentiment = analyze_sentiment(text)
    sources = detect_news_source(text)
    features = analyze_text_features(text)
    insights = generate_insights(prediction, confidence_score, sentiment, features)

    return {
        'prediction': prediction,
        'confidence': confidence_score,
        'sentiment': sentiment,
        'sources': sources,
        'features': features,
        'insights': insights
    }

def fallback_prediction(text):
    """Fallback prediction method when ML model is not available"""
    # Simple heuristic-based prediction
    fake_indicators = ['fake', 'hoax', 'conspiracy', 'false', 'rumor', 'scam', 'misleading', 'deceptive']
    clickbait_words = ['shocking', 'amazing', 'incredible', 'you won\'t believe', 'breaking', 'exclusive']

    text_lower = text.lower()

    # Count fake indicators
    fake_score = sum(1 for word in fake_indicators if word in text_lower)
    clickbait_score = sum(1 for word in clickbait_words if word in text_lower)

    # Simple scoring system
    total_score = fake_score * 2 + clickbait_score

    if total_score >= 3:
        prediction = "Fake News"
        confidence = min(0.85, 0.5 + (total_score * 0.1))
    else:
        prediction = "Real News"
        confidence = min(0.85, 0.6 + (max(0, 3 - total_score) * 0.05))

    # Additional analysis
    sentiment = analyze_sentiment(text)
    sources = detect_news_source(text)
    features = analyze_text_features(text)
    insights = generate_insights(prediction, confidence, sentiment, features)

    # Add fallback insight
    insights.append("🤖 Using heuristic analysis (ML model unavailable)")

    return {
        'prediction': prediction,
        'confidence': confidence,
        'sentiment': sentiment,
        'sources': sources,
        'features': features,
        'insights': insights
    }

# LLaMA (Ollama) based classifier

def predict_news_llama_ollama(text, model_name: str = "llama3.2:3b", host: str = None):
    """Classify news text using a local Ollama LLaMA model.

    Returns same structure as predict_news():
    { 'prediction': 'Real News'|'Fake News', 'confidence': float[0,1], 'sentiment': {...}, 'sources': [...], 'features': {...}, 'insights': [...] }
    """
    if not host:
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    system_prompt = (
        "You are a strict classifier for news authenticity. "
        "Classify the given text as either 'Real News' or 'Fake News'. "
        "Return ONLY a compact JSON object with keys 'label' and 'confidence'. "
        "'label' must be 'Real News' or 'Fake News'. 'confidence' must be a number between 0 and 1."
    )

    payload = {
        "model": model_name,
        "format": "json",  # ask Ollama to emit valid JSON
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Text to classify:\n{text}"},
        ],
        "stream": False,
        "options": {"temperature": 0.2}
    }

    try:
        resp = requests.post(f"{host}/api/chat", json=payload, timeout=45)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "{}")

        # content should be JSON per format, but parse defensively
        try:
            parsed = json.loads(content)
        except Exception:
            # Some models may wrap JSON in extra text; try to extract braces
            start = content.find("{")
            end = content.rfind("}")
            parsed = json.loads(content[start:end+1]) if start != -1 and end != -1 else {}

        label = str(parsed.get("label", "Real News")).strip()
        confidence = parsed.get("confidence", 0.6)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.6
        confidence = max(0.0, min(1.0, confidence))

        # Normalize label
        normalized_label = label.lower()
        if "fake" in normalized_label:
            prediction = "Fake News"
        elif "real" in normalized_label:
            prediction = "Real News"
        else:
            # If uncertain, decide based on a simple threshold
            prediction = "Fake News" if confidence >= 0.55 else "Real News"

        # Additional analysis for consistency with the app
        sentiment = analyze_sentiment(text)
        sources = detect_news_source(text)
        features = analyze_text_features(text)
        insights = generate_insights(prediction, confidence, sentiment, features)
        # insights.append("🧠 Classified using LLaMA via Ollama")

        return {
            'prediction': prediction,
            'confidence': confidence,
            'sentiment': sentiment,
            'sources': sources,
            'features': features,
            'insights': insights
        }
    except Exception:
        # On any failure (no server/model), gracefully fallback
        return fallback_prediction(text)

# DB check

def check_user(username, password):
    db_path = os.path.join(PROJECT_ROOT, 'users.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()
    return user

# Register user

def register_user(username, password):
    db_path = os.path.join(PROJECT_ROOT, 'users.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, 'user')", (username, password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

@app.route("/", methods=["GET", "POST"])
def home():
    if "username" not in session:
        return redirect(url_for('login'))
    if request.method == "GET":
        session['visited_home'] = True
        session.modified = True

    if "history" not in session:
        session["history"] = []

    analysis_result = None
    highlighted_text = None

    if request.method == "POST":
        news_text = request.form.get("news_text")
        if news_text:
            # Simple per-session cooldown (2 seconds)
            now_ts = time.time()
            last_ts = session.get("last_submit_ts", 0)
            if now_ts - last_ts < 2:
                error = "You're submitting too quickly. Please wait a moment."
                return render_template("index.html",
                    analysis_result=None,
                    highlighted_text=None,
                    history=session.get("history"),
                    user_score=calculate_user_score(session.get("history", [])),
                    achievements=get_user_achievements(session.get("history", [])),
                    error=error
                )

            is_valid, error_msg = validate_user_input(news_text)
            if not is_valid:
                return render_template("index.html",
                    analysis_result=None,
                    highlighted_text=highlight_keywords(news_text or ""),
                    history=session.get("history"),
                    user_score=calculate_user_score(session.get("history", [])),
                    achievements=get_user_achievements(session.get("history", [])),
                    error=error_msg
                )

            # Route by length: <50 words → Ollama; otherwise → BERT
            word_count = len(re.findall(r"[A-Za-z']+", news_text))
            if word_count < 50:
                res = predict_news_llama_ollama(news_text)
            else:
                res = predict_news_bert(news_text)
            analysis_result = {
                'prediction': res['prediction'],
                'confidence': res['confidence'],
                'sentiment': res['sentiment'],
                'sources': res['sources'],
                'features': res['features'],
                'insights': res['insights']
            }
            highlighted_text = highlight_keywords(news_text)

            session["history"].append({
                "text": news_text,
                "prediction": analysis_result['prediction'],
                "confidence": round(analysis_result['confidence'] * 100, 2),
                "sentiment": analysis_result['sentiment']['sentiment'],
                "sources": analysis_result['sources']
            })
            if len(session["history"]) > 10:
                session["history"].pop(0)
            session.modified = True
            session["last_submit_ts"] = now_ts

            with open(os.path.join(PROJECT_ROOT, "prediction_history.csv"), "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([session["username"], news_text, analysis_result['prediction'], 
                               round(analysis_result['confidence'], 4), datetime.now()])

    # Calculate user stats for gamification
    user_score = calculate_user_score(session.get("history", []))
    achievements = get_user_achievements(session.get("history", []))

    return render_template("index.html",
        analysis_result=analysis_result,
        highlighted_text=highlighted_text,
        history=session.get("history"),
        user_score=user_score,
        achievements=achievements
    )

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = check_user(request.form['username'], request.form['password'])
        if user:
            session['username'] = user[1]
            session['role'] = user[3]
            return redirect(url_for('home'))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        success = register_user(request.form['username'], request.form['password'])
        if success:
            return redirect(url_for('login'))
        else:
            return render_template("register.html", error="Username already exists")
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route("/feedback", methods=["POST"])
def feedback():
    news_text = request.form['text']
    prediction = request.form['prediction']
    feedback = request.form['feedback']
    comment = request.form.get('comment', '')
    username = session.get("username", "guest")
    with open(os.path.join(PROJECT_ROOT, "feedback.csv"), "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([username, news_text, prediction, feedback, comment, datetime.now()])
    return redirect(url_for('home'))

@app.route("/admin")
def admin():
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    feedbacks = []
    feedback_file = os.path.join(PROJECT_ROOT, "feedback.csv")
    if os.path.exists(feedback_file):
        with open(feedback_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            feedbacks = list(reader)
    return render_template("dashboard.html", feedbacks=feedbacks)

@app.route("/visualization")
def visualization():
    fake_count = 0
    real_count = 0
    correct_count = 0
    incorrect_count = 0

    hist_file = os.path.join(PROJECT_ROOT, "prediction_history.csv")
    if os.path.exists(hist_file):
        with open(hist_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    pred = row[2]
                    if pred == "Fake News":
                        fake_count += 1
                    elif pred == "Real News":
                        real_count += 1

    feedback_file = os.path.join(PROJECT_ROOT, "feedback.csv")
    if os.path.exists(feedback_file):
        with open(feedback_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 4:
                    fb = row[3]
                    if fb.lower() == "yes":
                        correct_count += 1
                    elif fb.lower() == "no":
                        incorrect_count += 1

    total_preds = fake_count + real_count
    total_feedback = correct_count + incorrect_count

    fake_percent = round((fake_count / total_preds) * 100, 2) if total_preds else 0
    real_percent = round((real_count / total_preds) * 100, 2) if total_preds else 0
    correct_percent = round((correct_count / total_feedback) * 100, 2) if total_feedback else 0
    incorrect_percent = round((incorrect_count / total_feedback) * 100, 2) if total_feedback else 0

    return render_template("visualization.html",
        fake_count=fake_count,
        real_count=real_count,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        fake_percent=fake_percent,
        real_percent=real_percent,
        correct_percent=correct_percent,
        incorrect_percent=incorrect_percent
    )

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/history")
def history():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("history.html", history=session.get("history", []))

@app.route("/clear_history", methods=["POST"])
def clear_history():
    if "username" not in session:
        return redirect(url_for("login"))
    session["history"] = []
    session.modified = True
    return redirect(url_for("history"))

@app.route("/game")
def game():
    if "username" not in session:
        return redirect(url_for('login'))
    if not session.get('visited_home', False):
        return redirect(url_for('home'))
    # Use pandas for robust CSV loading
    path = os.path.join(PROJECT_ROOT, "dataset/fake_news_detection(FakeNewsNet)/fnn_test.csv")
    df = pd.read_csv(path)
    questions = []
    for _, row in df.iterrows():
        pcontent = row["paragraph_based_content"]
        plist = []
        if pd.notnull(pcontent):
            if isinstance(pcontent, list):
                plist = pcontent
            elif isinstance(pcontent, str):
                try:
                    plist = json.loads(pcontent)
                except Exception:
                    try:
                        plist = ast.literal_eval(pcontent)
                    except Exception:
                        plist = []
        label = str(row.get("label_fnn", "real")).lower()
        is_real = label == "real"
        if plist and isinstance(plist[0], str) and plist[0].strip():
            headline = plist[0].strip()
            if len(headline) > 320:
                headline = headline[:320].rsplit(" ", 1)[0] + "..."
            questions.append({"id": row["id"], "headline": headline, "isReal": is_real})
    print("Total rows in dataframe:", len(df))
    print("Valid quiz questions found:", len(questions))
    sample = pyrandom.sample(questions, 10) if len(questions) >= 10 else questions
    print("Final quiz questions passed to template:", len(sample))
    return render_template("game.html", news_data=sample)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)
