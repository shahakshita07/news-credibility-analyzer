import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK data is downloaded
def download_nltk_resources():
    resources = ['stopwords', 'punkt', 'wordnet', 'omw-1.4', 'punkt_tab']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Error downloading {resource}: {e}")

download_nltk_resources()

def clean_text(text):
    """
    Performs full text preprocessing:
    - Lowercase conversion
    - Remove punctuation
    - Remove numbers
    - Tokenization
    - Stopword removal
    - Lemmatization
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword removal and Lemmatization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    cleaned_tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token not in stop_words and len(token) > 2
    ]

    return " ".join(cleaned_tokens)
