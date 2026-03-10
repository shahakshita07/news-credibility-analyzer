import requests
from bs4 import BeautifulSoup
from newspaper import Article
from urllib.parse import urlparse

def extract_article_details(url):
    """
    Robust 3-step extraction pipeline to bypass blocks and ensure content retrieval.
    """
    domain = urlparse(url).netloc
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # Step 1: Standard Newspaper3k extraction
    try:
        article = Article(url)
        article.download()
        article.parse()
        if article.text.strip():
            return {
                "title": article.title,
                "text": article.text,
                "source": domain,
                "success": True
            }
    except Exception:
        pass

    # Step 2: Requests with browser headers + Newspaper3k
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            article = Article(url)
            article.set_html(response.text)
            article.parse()
            if article.text.strip():
                return {
                    "title": article.title,
                    "text": article.text,
                    "source": domain,
                    "success": True
                }
    except Exception:
        pass

    # Step 3: BeautifulSoup Fallback
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract title
            title = ""
            title_tag = soup.find("h1") or soup.find("title")
            if title_tag:
                title = title_tag.get_text().strip()
            
            # Extract paragraphs
            paragraphs = soup.find_all("p")
            text = " ".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            
            if text.strip():
                return {
                    "title": title,
                    "text": text,
                    "source": domain,
                    "success": True
                }
    except Exception:
        pass

    # If all fail
    return {
        "title": "",
        "text": "",
        "source": domain,
        "success": False,
        "error": "Unable to extract article from this website. Please paste the article text manually."
    }

# Alias for specific user request while maintaining UI compatibility
extract_article = extract_article_details
