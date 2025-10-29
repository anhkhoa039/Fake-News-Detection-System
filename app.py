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
import random

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with a secure key

# Download stopwords if needed
nltk.download('stopwords', quiet=True)

# Load the saved model and vectorizer
try:
    with open('fake_news_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('tfidf_vectorizer.pkl', 'rb') as f:
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
    text = re.sub('[^a-z\s]', '', text)
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

def predict_news(text):
    # Check if model is available
    if model is None or vectorizer is None:
        # Fallback prediction based on heuristics
        return fallback_prediction(text)
    
    processed = preprocess_text(text)
    vectorized = vectorizer.transform([processed]).toarray()
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

# DB check
def check_user(username, password):
    db_path = os.path.join(os.path.dirname(__file__), 'users.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()
    return user

# Register user
def register_user(username, password):
    db_path = os.path.join(os.path.dirname(__file__), 'users.db')
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

    if "history" not in session:
        session["history"] = []

    analysis_result = None
    highlighted_text = None

    if request.method == "POST":
        news_text = request.form.get("news_text")
        if news_text:
            analysis_result = predict_news(news_text)
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

            with open("prediction_history.csv", "a", newline="", encoding="utf-8") as f:
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
    with open("feedback.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([username, news_text, prediction, feedback, comment, datetime.now()])
    return redirect(url_for('home'))

@app.route("/admin")
def admin():
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    feedbacks = []
    if os.path.exists("feedback.csv"):
        with open("feedback.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            feedbacks = list(reader)
    return render_template("dashboard.html", feedbacks=feedbacks)

@app.route("/visualization")
def visualization():
    fake_count = 0
    real_count = 0
    correct_count = 0
    incorrect_count = 0

    if os.path.exists("prediction_history.csv"):
        with open("prediction_history.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    pred = row[2]
                    if pred == "Fake News":
                        fake_count += 1
                    elif pred == "Real News":
                        real_count += 1

    if os.path.exists("feedback.csv"):
        with open("feedback.csv", "r", encoding="utf-8") as f:
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

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)
