"""Sentiment analysis: Google News RSS + Reddit + FMP News.

Uses FinBERT (ProsusAI/finbert) for financial-domain sentiment analysis,
with VADER as a fallback if FinBERT is unavailable (missing torch/transformers).
"""

import logging

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import config
from utils.data_fetch import get_reddit_posts

logger = logging.getLogger(__name__)

# VADER fallback — always available
_vader_analyzer = SentimentIntensityAnalyzer()

# FinBERT singleton — lazy-loaded on first use
_finbert = None


def _get_finbert():
    """Lazy-load FinBERT pipeline. Returns pipeline or None if unavailable."""
    global _finbert
    if _finbert is None:
        try:
            import os
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            # Suppress noisy HF Hub warnings
            import warnings
            warnings.filterwarnings("ignore", message=".*unauthenticated.*")
            warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
            from transformers import logging as hf_logging
            hf_logging.set_verbosity_error()

            from transformers import pipeline
            _finbert = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                top_k=None,
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT loaded successfully")
        except Exception as e:
            logger.warning("FinBERT unavailable, falling back to VADER: %s", e)
            _finbert = "unavailable"
    return _finbert if _finbert != "unavailable" else None


def analyse(ticker: str, company_name: str = "") -> dict:
    """Fetch recent news + Reddit + FMP news and score sentiment. Returns score from -1 to 1."""
    # Google News sentiment
    news_headlines = _fetch_headlines(ticker, company_name)
    news_score, news_details = _score_texts(news_headlines)

    # Reddit sentiment
    reddit_posts = get_reddit_posts(ticker)
    reddit_titles = [p["title"] for p in reddit_posts if p.get("title")]
    reddit_score, reddit_details = _score_texts(reddit_titles)

    # Tag reddit details with source and engagement
    for i, detail in enumerate(reddit_details):
        detail["source"] = "reddit"
        if i < len(reddit_posts):
            detail["subreddit"] = reddit_posts[i].get("subreddit", "")
            detail["upvotes"] = reddit_posts[i].get("score", 0)

    for detail in news_details:
        detail["source"] = "news"

    # FMP News sentiment (3rd source)
    fmp_headlines, fmp_score, fmp_details = _fetch_fmp_news(ticker)
    fmp_active = len(fmp_details) > 0

    # Weighted combination — FMP is primary (curated, ticker-specific)
    if fmp_active:
        # 3-source weights: FMP primary, Google News secondary, Reddit tertiary
        fmp_weight = 0.45
        news_weight = 0.30
        reddit_weight = 0.25
    else:
        # Fallback to 2-source weights when FMP unavailable
        news_weight = config.SENTIMENT_WEIGHTS["news"]
        reddit_weight = config.SENTIMENT_WEIGHTS["reddit"]
        fmp_weight = 0.0

    if not news_details and not reddit_details and not fmp_details:
        return {
            "score": 0.0,
            "reasons": ["no recent news or social media found"],
            "headlines": [],
            "reddit_headlines": [],
            "fmp_headlines": [],
            "avg_sentiment": 0.0,
            "news_score": 0.0,
            "reddit_score": 0.0,
            "fmp_news_score": 0.0,
        }

    # Calculate combined score — if a source is missing, redistribute weight
    sources = []
    if news_details:
        sources.append((news_score, news_weight))
    if reddit_details:
        sources.append((reddit_score, reddit_weight))
    if fmp_details:
        sources.append((fmp_score, fmp_weight))

    if sources:
        total_weight = sum(w for _, w in sources)
        combined_score = sum(s * w for s, w in sources) / total_weight if total_weight > 0 else 0.0
    else:
        combined_score = 0.0

    reasons = []
    if combined_score > 0.2:
        reasons.append("positive sentiment")
    elif combined_score < -0.2:
        reasons.append("negative sentiment")
    else:
        reasons.append("neutral sentiment")

    # Add source-specific notes
    if reddit_details:
        if reddit_score > 0.2:
            reasons.append("bullish Reddit buzz")
        elif reddit_score < -0.2:
            reasons.append("bearish Reddit sentiment")

    if fmp_details:
        if fmp_score > 0.2:
            reasons.append("positive FMP news")
        elif fmp_score < -0.2:
            reasons.append("negative FMP news")

    return {
        "score": max(-1.0, min(1.0, combined_score)),
        "reasons": reasons,
        "headlines": news_details,
        "reddit_headlines": reddit_details,
        "fmp_headlines": fmp_details,
        "avg_sentiment": combined_score,
        "news_score": news_score,
        "reddit_score": reddit_score,
        "fmp_news_score": fmp_score,
    }


# ---------------------------------------------------------------------------
# Scoring — FinBERT primary, VADER fallback
# ---------------------------------------------------------------------------

def _score_texts(texts: list[str]) -> tuple[float, list[dict]]:
    """Score texts using FinBERT (preferred) or VADER (fallback).

    Returns (avg_score, details) where score is in [-1, 1].
    """
    if not texts:
        return 0.0, []

    fb = _get_finbert()
    if fb is not None:
        return _score_texts_finbert(texts, fb)
    return _score_texts_vader(texts)


def _score_texts_finbert(texts: list[str], fb) -> tuple[float, list[dict]]:
    """Run FinBERT on a list of texts. Returns (avg_score, details).

    FinBERT outputs labels: positive, negative, neutral with confidence scores.
    Maps to: positive → +confidence, negative → -confidence, neutral → 0.
    """
    scores = []
    details = []

    for text in texts[:20]:  # Limit batch size for memory
        try:
            result = fb(text[:512])  # top_k=None returns all labels per input
            # result is a list of dicts: [{"label": "positive", "score": 0.9}, ...]
            if isinstance(result[0], list):
                # top_k=None returns nested list
                label_scores = {r["label"]: r["score"] for r in result[0]}
            else:
                label_scores = {r["label"]: r["score"] for r in result}

            # Net sentiment: positive confidence minus negative confidence
            score = label_scores.get("positive", 0) - label_scores.get("negative", 0)
            scores.append(score)
            details.append({"title": text, "sentiment": round(score, 4)})
        except Exception:
            # If a single text fails, use VADER for just that text
            vs = _vader_analyzer.polarity_scores(text)
            scores.append(vs["compound"])
            details.append({"title": text, "sentiment": vs["compound"]})

    avg = sum(scores) / len(scores) if scores else 0.0
    return avg, details


def _score_texts_vader(texts: list[str]) -> tuple[float, list[dict]]:
    """Fallback: Run VADER on a list of texts. Returns (avg_score, details)."""
    scores = []
    details = []
    for text in texts[:config.NEWS_HEADLINE_COUNT]:
        vs = _vader_analyzer.polarity_scores(text)
        compound = vs["compound"]
        scores.append(compound)
        details.append({"title": text, "sentiment": compound})

    avg = sum(scores) / len(scores) if scores else 0.0
    return avg, details


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _fetch_headlines(ticker: str, company_name: str) -> list[str]:
    """Fetch headlines from Google News RSS."""
    query = company_name if company_name else ticker
    query = query.replace(".L", "").strip()

    url = f"https://news.google.com/rss/search?q={query}+stock&hl=en&gl=US&ceid=US:en"
    try:
        feed = feedparser.parse(url)
        return [entry.title for entry in feed.entries[:config.NEWS_HEADLINE_COUNT]]
    except Exception:
        return []


def _fetch_fmp_news(ticker: str) -> tuple[list[str], float, list[dict]]:
    """Fetch FMP stock news and score. Returns (texts, avg_score, details)."""
    try:
        from utils.fmp_client import get_stock_news, is_available
        if not is_available():
            return [], 0.0, []
        articles = get_stock_news(ticker, limit=20)
        if not articles or not isinstance(articles, list):
            return [], 0.0, []
        texts = [a.get("title", "") for a in articles if a.get("title")]
        if not texts:
            return [], 0.0, []
        score, details = _score_texts(texts)
        for d in details:
            d["source"] = "fmp_news"
        return texts, score, details
    except Exception:
        return [], 0.0, []
