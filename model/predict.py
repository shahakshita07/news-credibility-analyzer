import sys
import os

# Add root directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_loader import ModelLoader
from utils.preprocessing import clean_text

def predict_news_credibility(text):
    """
    Accepts article text, preprocesses it, and predicts credibility.
    """
    if not text or len(text.strip()) < 10:
        return None

    model = ModelLoader.get_model()
    vectorizer = ModelLoader.get_vectorizer()

    if model is None or vectorizer is None:
        return {"error": "Model or vectorizer not loaded. Please train the model first."}

    # Preprocess
    cleaned_data = clean_text(text)
    
    # Vectorize
    vectorized_data = vectorizer.transform([cleaned_data])
    
    # Predict
    # Many models support predict_proba, but LinearSVC does not by default.
    # We use decision_function for SVM or just prediction for simplicity if proba is unavailable.
    try:
        probabilities = model.predict_proba(vectorized_data)[0]
        fake_prob = probabilities[0]
        real_prob = probabilities[1]
    except AttributeError:
        # Fallback for models without predict_proba (like LinearSVC)
        prediction = model.predict(vectorized_data)[0]
        real_prob = 1.0 if prediction == 1 else 0.0
        fake_prob = 1.0 - real_prob

    # Score
    credibility_score = round(real_prob * 100, 2)
    
    # Classification
    status = ""
    if credibility_score <= 40:
        status = "Likely Fake"
    elif credibility_score <= 70:
        status = "Possibly Misleading"
    else:
        status = "Likely Reliable"

    return {
        "real_probability": round(real_prob * 100, 2),
        "fake_probability": round(fake_prob * 100, 2),
        "credibility_score": credibility_score,
        "status": status
    }
