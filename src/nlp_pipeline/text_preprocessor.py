"""
Basic text preprocessing for news articles.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Download NLTK data lazily
_NLTK_READY = False


def _ensure_nltk():
    global _NLTK_READY
    if _NLTK_READY:
        return
    try:
        import nltk
        for resource in ("stopwords", "punkt", "wordnet"):
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass
        _NLTK_READY = True
    except Exception as e:
        logger.warning(f"NLTK setup failed: {e}")


class TextPreprocessor:
    """Clean and normalize news article text."""

    # Financial noise patterns to strip
    _NOISE = re.compile(
        r"(subscribe|advertisement|click here|read more|follow us"
        r"|sign up|newsletter|cookies|privacy policy)",
        re.IGNORECASE,
    )

    @staticmethod
    def clean(text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        # Remove HTML entities and tags
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&\w+;", " ", text)
        # Remove URLs
        text = re.sub(r"https?://\S+", " ", text)
        # Remove special chars but keep sentence structure
        text = re.sub(r"[^\w\s.,!?'\-]", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def truncate(text: str, max_chars: int = 2000) -> str:
        return text[:max_chars] if len(text) > max_chars else text

    @classmethod
    def prepare(cls, text: str, max_chars: int = 2000) -> str:
        return cls.truncate(cls.clean(text), max_chars)
