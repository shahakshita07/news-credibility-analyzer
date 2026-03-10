def detect_sensational_keywords(text):
    """
    Detects sensational/misleading keywords in the text.
    Returns a list of detected keywords.
    """
    if not text:
        return []

    sensational_keywords = [
        "shocking", "unbelievable", "secret", "breaking", 
        "you won't believe", "exposed", "scandal", "alert", 
        "must see", "truth revealed", "bombshell"
    ]
    
    detected = []
    text_lower = text.lower()
    
    for word in sensational_keywords:
        if word in text_lower:
            detected.append(word)
            
    return detected
