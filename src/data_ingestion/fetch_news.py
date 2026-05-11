"""
Fetch news articles from Google News RSS feed.
Uses stdlib xml.etree.ElementTree to avoid feedparser's sgmllib dependency.
"""

import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import json
from datetime import datetime
from pathlib import Path
import time
import logging
import hashlib

logger = logging.getLogger(__name__)


class NewsFetcher:
    def __init__(self, data_dir="data/news/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })

    def _parse_rss_xml(self, xml_content):
        """Parse RSS XML and return list of (title, link, published, summary) tuples."""
        items = []
        try:
            root = ET.fromstring(xml_content)
            # RSS namespace map
            ns = {"media": "http://search.yahoo.com/mrss/"}

            channel = root.find("channel")
            if channel is None:
                return items

            for item in channel.findall("item"):
                title = item.findtext("title", default="")
                link = item.findtext("link", default="")
                pub_date = item.findtext("pubDate", default=datetime.now().isoformat())
                description = item.findtext("description", default="")
                source_el = item.find("source")
                source = source_el.text if source_el is not None else "Unknown"

                # Clean HTML from description
                description = BeautifulSoup(description, "html.parser").get_text()

                items.append({
                    "title": title,
                    "link": link,
                    "published": pub_date,
                    "source": source,
                    "summary": description,
                })
        except ET.ParseError as e:
            logger.warning(f"RSS XML parse error: {e}")
        return items

    def fetch_google_news_rss(self, query, max_results=20):
        """Fetch news from Google News RSS with deduplication."""
        rss_url = (
            f"https://news.google.com/rss/search"
            f"?q={query.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
        )

        try:
            response = self.session.get(rss_url, timeout=15)
            response.raise_for_status()
            raw_items = self._parse_rss_xml(response.content)

            articles = []
            for item in raw_items[:max_results]:
                if not item["link"]:
                    continue
                article_id = hashlib.md5(item["link"].encode()).hexdigest()
                articles.append({
                    "id": article_id,
                    "title": item["title"],
                    "url": item["link"],
                    "published": item["published"],
                    "source": item["source"],
                    "summary": item["summary"],
                    "query": query,
                    "fetched_at": datetime.now().isoformat(),
                    "content": "",
                })

            logger.info(f"Fetched {len(articles)} articles for query: '{query}'")
            return articles

        except Exception as e:
            logger.error(f"Failed to fetch RSS for '{query}': {e}")
            return []

    def _extract_article_text(self, url):
        """Extract article text with timeout and fallback."""
        try:
            response = self.session.get(url, timeout=8)
            soup = BeautifulSoup(response.content, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            for selector in ["article", ".article-body", ".story-content", "main"]:
                elements = soup.select(selector)
                if elements:
                    text = " ".join(el.get_text(separator=" ") for el in elements)
                    return text.strip()[:3000]

            # Fallback: grab all paragraphs
            paragraphs = soup.find_all("p")
            if paragraphs:
                return " ".join(p.get_text() for p in paragraphs)[:3000]

            return ""
        except Exception as e:
            logger.debug(f"Could not extract text from {url}: {e}")
            return ""

    def save_articles(self, articles, stock_symbol):
        """Save articles to JSON, merging with any existing ones."""
        if not articles:
            return

        filename = self.data_dir / f"{stock_symbol}_{datetime.now().strftime('%Y%m%d')}.json"

        existing_articles = []
        if filename.exists():
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    existing_articles = json.load(f)
            except Exception:
                existing_articles = []

        all_articles = {a["id"]: a for a in existing_articles}
        for a in articles:
            all_articles[a["id"]] = a

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(list(all_articles.values()), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(all_articles)} unique articles for {stock_symbol}")

    def load_all_articles(self, stock_symbol):
        """Load all stored articles for a stock across all dates."""
        articles = []
        for json_file in sorted(self.data_dir.glob(f"{stock_symbol}_*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    articles.extend(json.load(f))
            except Exception as e:
                logger.warning(f"Could not read {json_file}: {e}")
        # Deduplicate
        seen = {}
        for a in articles:
            seen[a["id"]] = a
        return list(seen.values())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetcher = NewsFetcher()
    stocks = {
        "LT.NS": "Larsen Turbo L&T stock India",
        "COALINDIA.NS": "Coal India COALINDIA stock NSE",
        "TATACOMM.NS": "Tata Communications stock India",
        "ADANIENT.NS": "Adani Enterprises stock India",
        "INFY.NS": "Infosys stock India",
        "BBOX.NS": "Black Box BBOX stock India",
    }
    for symbol, query in stocks.items():
        articles = fetcher.fetch_google_news_rss(query)
        fetcher.save_articles(articles, symbol)
        time.sleep(2)
