# 🛡️ Fake News Detection - AI-Powered Fake News Detection System

A comprehensive web application that uses machine learning to detect fake news articles with advanced analysis features, user management, and gamification elements.

## 🌟 Features

### 🔍 **Core Detection Features**
- **AI-Powered Analysis** - Uses machine learning models to analyze news articles
- **Confidence Scoring** - Provides confidence percentages for predictions
- **Real-time Processing** - Instant analysis of news text
- **Fallback System** - Heuristic-based detection when ML models are unavailable

### 📊 **Advanced Analytics**
- **Sentiment Analysis** - Analyzes emotional tone of articles
- **Source Detection** - Identifies potential news sources mentioned
- **Text Feature Analysis** - Evaluates readability, clickbait potential, and writing quality
- **AI Insights** - Provides intelligent explanations for predictions

### 🎮 **Gamification System**
- **Truth Score** - User scoring system based on analysis accuracy
- **Achievement System** - Unlockable achievements for different milestones
- **Progress Tracking** - Visual progress indicators and statistics
- **User Engagement** - Interactive elements to encourage usage

### 👥 **User Management**
- **User Authentication** - Secure login and registration system
- **Role-based Access** - Admin and regular user roles
- **Session Management** - Persistent user sessions
- **User Profiles** - Individual user statistics and history

### 📈 **Data Visualization**
- **Analytics Dashboard** - Comprehensive data visualization
- **Prediction History** - Track all previous analyses
- **Statistics Overview** - Real-time statistics and metrics
- **Feedback System** - User feedback collection and analysis

### 🎨 **Modern UI/UX**
- **Responsive Design** - Works on desktop, tablet, and mobile
- **Glassmorphism Design** - Modern translucent UI elements
- **Purple Gradient Theme** - Consistent branding throughout
- **Smooth Animations** - Enhanced user experience
- **Dark/Light Contrast** - Optimized readability

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd FAKE-NEWS-DETECTION
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the database**
   ```bash
   python create_users_db.py
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   - Local: `http://localhost:5001`
   - Network: `http://YOUR_IP:5001`

## 🔧 Configuration

### Default Login Credentials
- **Username:** `admin`
- **Password:** `admin123`

### Model Files
The application expects these model files in the root directory:
- `fake_news_model.pkl` - Trained ML model
- `tfidf_vectorizer.pkl` - TF-IDF vectorizer

If these files are missing, the application will use fallback heuristic detection.

## 📱 Usage Guide

### 1. **Login**
- Navigate to the application URL
- Enter your credentials
- Click "Login"

### 2. **Analyze News**
- Paste news article text in the input field
- Click "Analyze News"
- View detailed results including:
  - Prediction (Real/Fake)
  - Confidence score
  - Sentiment analysis
  - Source detection
  - Text features
  - AI insights

### 3. **View History**
- Click "History" in navigation
- Browse previous analyses
- View detailed results for each entry

### 4. **Analytics Dashboard**
- Click "Analytics" in navigation
- View comprehensive statistics
- Monitor system performance
- Review user feedback

### 5. **Admin Panel** (Admin users only)
- Click "Dashboard" in navigation
- Manage user feedback
- View system statistics
- Monitor application performance

## 🌐 External Access

### Local Network Access
```bash
python app.py
# Access via: http://YOUR_IP:5001
```

### Internet Access (using Cloudflare Tunnel)
```bash
# Terminal 1: Start Flask app
python app.py

# Terminal 2: Create tunnel
cloudflared tunnel --url http://localhost:5001

# Share the https://xxxxx.trycloudflare.com URL (or your custom domain if configured)
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET/POST | Main analysis page |
| `/login` | GET/POST | User authentication |
| `/register` | GET/POST | User registration |
| `/history` | GET | View analysis history |
| `/visualization` | GET | Analytics dashboard |
| `/admin` | GET | Admin panel |
| `/logout` | GET | User logout |
| `/clear_history` | POST | Clear user history |

## 🛠️ Technical Details

### Technology Stack
- **Backend:** Flask (Python)
- **Frontend:** HTML5, CSS3, JavaScript
- **Database:** SQLite3
- **ML Models:** scikit-learn, TF-IDF
- **Text Processing:** NLTK, TextBlob
- **Styling:** Custom CSS with glassmorphism

### Dependencies
```
Flask==2.3.3
nltk==3.8.1
textblob==0.17.1
scikit-learn==1.1.3
pandas==1.4.4
numpy==1.21.6
```

### File Structure
```
FAKE-NEWS-DETECTION/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── create_users_db.py     # Database setup
├── users.db              # SQLite database
├── prediction_history.csv # Analysis history
├── feedback.csv          # User feedback
├── templates/            # HTML templates
│   ├── index.html        # Main page
│   ├── login.html        # Login page
│   ├── register.html     # Registration page
│   ├── history.html     # History page
│   ├── visualization.html # Analytics page
│   └── dashboard.html    # Admin dashboard
└── README.md            # This file
```

## 🎯 Features Breakdown

### Analysis Features
- **Text Preprocessing** - Removes stopwords, normalizes text
- **Keyword Highlighting** - Highlights suspicious keywords
- **Confidence Scoring** - Provides reliability metrics
- **Multi-factor Analysis** - Combines multiple detection methods

### User Experience
- **Responsive Design** - Mobile-friendly interface
- **Real-time Feedback** - Instant analysis results
- **Progress Indicators** - Visual loading states
- **Error Handling** - Graceful error management

### Admin Features
- **User Management** - View and manage users
- **Feedback Analysis** - Collect and analyze user feedback
- **System Monitoring** - Track application performance
- **Data Export** - Export analysis data

## 🔒 Security Features

- **Session Management** - Secure user sessions
- **Input Validation** - Sanitized user inputs
- **SQL Injection Protection** - Parameterized queries
- **XSS Protection** - Escaped output rendering

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Change port in app.py
   app.run(debug=True, host='0.0.0.0', port=5002)
   ```

2. **Model Files Missing**
   - Application will use fallback detection
   - Place model files in root directory

3. **Database Errors**
   ```bash
   python create_users_db.py
   ```

4. **Dependencies Issues**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

## 📈 Performance

- **Response Time:** < 2 seconds for analysis
- **Concurrent Users:** Supports multiple simultaneous users
- **Memory Usage:** Optimized for efficient resource usage
- **Scalability:** Can be deployed on cloud platforms

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation

## 🎉 Acknowledgments

- Flask community for the excellent web framework
- scikit-learn team for machine learning tools
- NLTK contributors for natural language processing
- All users who provided feedback and suggestions

---

**Made with ❤️ for truth and accuracy in news**