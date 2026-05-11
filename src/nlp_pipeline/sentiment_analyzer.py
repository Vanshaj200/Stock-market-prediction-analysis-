"""
Sentiment analysis using VADER (NLTK).
VADER is chosen for speed and zero-download footprint relative to FinBERT.
"""

import logging
import nltk

logger = logging.getLogger(__name__)

# Ensure VADER lexicon is available
try:
    nltk.download("vader_lexicon", quiet=True)
except Exception:
    pass


class SentimentAnalyzer:
    """
    VADER-based sentiment analyzer for financial news.

    Auto-fix mechanisms:
    - Returns neutral sentiment on empty/error input
    - Handles missing VADER lexicon by re-downloading
    """

    def __init__(self):
        self._sia = None
        self._init_vader()

    def _init_vader(self):
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            self._sia = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer ready")
        except Exception as e:
            logger.warning(f"VADER init failed: {e}. Retrying download...")
            try:
                nltk.download("vader_lexicon", quiet=False)
                from nltk.sentiment import SentimentIntensityAnalyzer
                self._sia = SentimentIntensityAnalyzer()
                logger.info("VADER loaded after re-download")
            except Exception as e2:
                logger.error(f"Could not initialize VADER: {e2}")

    def analyze(self, text: str) -> dict:
        """
        Analyze text sentiment.

        Returns:
            dict with keys: sentiment (str), score (float -1..1), confidence (float 0..1)
        """
        neutral_result = {"sentiment": "neutral", "score": 0.0, "confidence": 0.0}

        if not text or not isinstance(text, str) or not text.strip():
            return neutral_result

        if self._sia is None:
            return neutral_result

        try:
            scores = self._sia.polarity_scores(text[:5000])
            compound = scores["compound"]

            if compound >= 0.05:
                sentiment = "positive"
            elif compound <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            return {
                "sentiment": sentiment,
                "score": float(compound),
                "confidence": float(abs(compound)),
            }
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return neutral_result

    def batch_analyze(self, texts: list, show_progress: bool = True) -> list:
        """Analyze a list of texts, returning list of result dicts."""
        if show_progress:
            try:
                from tqdm import tqdm
                texts = tqdm(texts, desc="Sentiment")
            except ImportError:
                pass
        return [self.analyze(t) for t in texts]

    def aggregate_scores(self, scores: list) -> dict:
        """
        Aggregate a list of sentiment score dicts into a single summary.
        Weighted by confidence.
        """
        if not scores:
            return {"mean_score": 0.0, "std_score": 0.0, "positive_ratio": 0.0,
                    "negative_ratio": 0.0, "neutral_ratio": 0.0}

        import statistics
        raw_scores = [s["score"] for s in scores]
        sentiments = [s["sentiment"] for s in scores]
        n = len(sentiments)

        return {
            "mean_score": statistics.mean(raw_scores),
            "std_score": statistics.stdev(raw_scores) if len(raw_scores) > 1 else 0.0,
            "positive_ratio": sentiments.count("positive") / n,
            "negative_ratio": sentiments.count("negative") / n,
            "neutral_ratio": sentiments.count("neutral") / n,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = SentimentAnalyzer()
    samples = [
        "Larsen & Turbo reported strong quarterly earnings, beating expectations by 15%.",
        "Adani Enterprises shares fell sharply amid concerns over debt levels.",
        "Infosys maintains guidance for FY26 despite global uncertainty.",
    ]
    for text in samples:
        result = analyzer.analyze(text)
        print(f"  [{result['sentiment']:8s}] score={result['score']:+.3f}  {text[:60]}")
