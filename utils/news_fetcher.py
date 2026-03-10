from api.google_news_api import GoogleNewsAPI

def get_latest_news():
    """
    Fetches latest trending news.
    """
    return GoogleNewsAPI.fetch_top_news()

def fetch_and_analyze(news_item, analyzer_func):
    """
    Takes a news item, extracts text, and runs the analyzer function.
    """
    # This will be used in the Streamlit app to bridge fetching and prediction
    pass
