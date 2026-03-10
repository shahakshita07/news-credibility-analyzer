from textblob import TextBlob

def analyze_sentiment(text):
    """
    Analyzes sentiment of the text using TextBlob.
    Returns polarity and subjectivity.
    """
    if not text:
        return {"polarity": 0.0, "subjectivity": 0.0}
    
    analysis = TextBlob(text)
    return {
        "polarity": round(analysis.sentiment.polarity, 4),
        "subjectivity": round(analysis.sentiment.subjectivity, 4)
    }
