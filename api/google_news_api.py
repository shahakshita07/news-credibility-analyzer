import requests
import xml.etree.ElementTree as ET
from datetime import datetime

class GoogleNewsAPI:
    """
    Utility class to fetch news from Google News RSS feed.
    """
    RSS_URL = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"

    @staticmethod
    def fetch_top_news():
        """
        Fetches top trending news headlines from Google News RSS.
        Returns a list of dictionaries containing title, source, url, and date.
        """
        try:
            response = requests.get(GoogleNewsAPI.RSS_URL)
            if response.status_code != 200:
                return []

            root = ET.fromstring(response.content)
            news_items = []

            for item in root.findall('.//item'):
                title = item.find('title').text
                link = item.find('link').text
                pub_date = item.find('pubDate').text
                source = item.find('source').text if item.find('source') is not None else "Unknown"

                # Parse date
                try:
                    date_obj = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
                    formatted_date = date_obj.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    formatted_date = pub_date

                news_items.append({
                    "title": title,
                    "source": source,
                    "url": link,
                    "published_date": formatted_date
                })

            return news_items
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []
