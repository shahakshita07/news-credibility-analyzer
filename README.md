# Credify – AI News Credibility Analyzer

AI-powered web application that detects fake or misleading news using machine learning and NLP.

## Features
- **Fake news detection**: High-accuracy classification of news articles.
- **Credibility scoring**: Real-time percentage-based reliability indicator.
- **Sentiment analysis**: Visualizing polarity and subjectivity bias.
- **Sensational language detection**: Identifying clickbait and high-risk language.
- **Live news monitoring**: Scan trending global headlines instantly.
- **Modern interactive dashboard**: Professional single-page AI dashboard UI.

## Tech Stack
- **Python**
- **Streamlit** (Web UI)
- **Scikit-learn** (Machine Learning Engine)
- **NLP** (NLTK, TextBlob, Newspaper3k)
- **Plotly** (Interactive Analytics)
- **Google News API** (Live Feed)

## Project Structure
```text
credify-news-analyzer/
│
├── app/                # Streamlit UI Layer
├── api/                # External API Integrations
├── data/               # Datasets and raw data
├── model/              # ML Pipeline (Train/Predict)
├── utils/              # Helper modules (NLP/Scrapers)
├── notebooks/          # Research & Development
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup and Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/credify-news-analyzer.git
   cd credify-news-analyzer
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the initial model:**
   Since `.pkl` files are ignored, generate the model artifacts locally:
   ```bash
   python model/train_model.py
   ```

5. **Run the application:**
   ```bash
   streamlit run app/app.py
   ```

---
Built with Python, Scikit-learn, and Machine Learning.