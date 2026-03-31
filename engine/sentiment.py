"""Sentiment analysis: Google News RSS + Reddit + FMP News.

Uses FinBERT (ProsusAI/finbert) for financial-domain sentiment analysis,
with VADER as a fallback if FinBERT is unavailable (missing torch/transformers).
"""

import calendar
import logging
import math
import time
from datetime import datetime

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import config
from utils.analysis_cache import PersistentAnalysisCache
from utils.data_fetch import get_reddit_posts

logger = logging.getLogger(__name__)

# VADER fallback — always available
_vader_analyzer = SentimentIntensityAnalyzer()

# Per-ticker sentiment cache (TTL-based, survives within a single process)
_sentiment_cache: dict[str, tuple[dict, float]] = {}  # {ticker: (result, timestamp)}
SENTIMENT_CACHE_TTL = getattr(config, "SENTIMENT_CACHE_TTL", 3600)  # 1 hour default
_PERSISTENT_SENTIMENT_TTL = getattr(config, "SENTIMENT_PERSISTENT_CACHE_TTL", 21600)
_persistent_cache = PersistentAnalysisCache("sentiment")

# Throttle between Google News RSS fetches to avoid rate-limiting
_NEWS_FETCH_DELAY = 2.0  # seconds between RSS calls (raised from 1.0 to reduce rate-limiting)
_last_news_fetch_time: float = 0.0

# FinBERT singleton — lazy-loaded on first use
_finbert = None


def _cache_key(ticker: str, company_name: str) -> str:
    normalized_name = (company_name or "").strip().lower()
    return f"{ticker.upper()}|{normalized_name}"


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
    """Fetch recent news + Reddit + FMP news and score sentiment. Returns score from -1 to 1.

    Includes sentiment_confidence (0-1) based on data quality:
    - Number of articles found across all sources
    - Number of active sources (Google News, Reddit, FMP)
    """
    cache_key = _cache_key(ticker, company_name)

    # --- TTL Cache: return cached result if fresh enough ---
    if cache_key in _sentiment_cache:
        cached_result, cached_ts = _sentiment_cache[cache_key]
        if time.time() - cached_ts < SENTIMENT_CACHE_TTL:
            logger.debug("Sentiment cache hit for %s (%.0fs old)", ticker,
                         time.time() - cached_ts)
            return cached_result

    persistent_result = _persistent_cache.get(cache_key, _PERSISTENT_SENTIMENT_TTL)
    if persistent_result is not None:
        _sentiment_cache[cache_key] = (persistent_result, time.time())
        logger.debug("Sentiment persistent cache hit for %s", ticker)
        return persistent_result

    # Google News sentiment (with throttle to avoid rate-limiting)
    global _last_news_fetch_time
    elapsed = time.time() - _last_news_fetch_time
    if elapsed < _NEWS_FETCH_DELAY:
        time.sleep(_NEWS_FETCH_DELAY - elapsed)

    news_items = _fetch_headlines(ticker, company_name)
    _last_news_fetch_time = time.time()

    news_texts = [item[0] for item in news_items]
    news_timestamps = [item[1] for item in news_items]
    news_score, news_details = _score_texts(news_texts, timestamps=news_timestamps)

    # Reddit sentiment (no timestamps available — treated as fresh)
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

    # --- Data quality metrics ---
    news_count = len(news_details)
    reddit_count = len(reddit_details)
    fmp_count = len(fmp_details)
    total_articles = news_count + reddit_count + fmp_count
    active_sources = sum([news_count > 0, reddit_count > 0, fmp_count > 0])

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
        result = {
            "score": 0.0,
            "reasons": ["no recent news or social media found"],
            "headlines": [],
            "reddit_headlines": [],
            "fmp_headlines": [],
            "avg_sentiment": 0.0,
            "news_score": 0.0,
            "reddit_score": 0.0,
            "fmp_news_score": 0.0,
            "sentiment_confidence": 0.0,
            "article_count": 0,
            "active_sources": 0,
        }
        _sentiment_cache[cache_key] = (result, time.time())
        _persistent_cache.put(cache_key, result)
        _persistent_cache.save()
        return result

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

    # --- Sentiment confidence score ---
    # Based on: number of articles (more = higher) and source diversity
    # Max confidence at 10+ articles from 3 sources
    article_confidence = min(total_articles / 10.0, 1.0)
    source_confidence = active_sources / 3.0
    sentiment_confidence = round(0.6 * article_confidence + 0.4 * source_confidence, 3)

    if sentiment_confidence < 0.3:
        reasons.append("low data confidence")
        logger.info("%s: low sentiment confidence %.2f (%d articles, %d sources)",
                    ticker, sentiment_confidence, total_articles, active_sources)

    result = {
        "score": max(-1.0, min(1.0, combined_score)),
        "reasons": reasons,
        "headlines": news_details,
        "reddit_headlines": reddit_details,
        "fmp_headlines": fmp_details,
        "avg_sentiment": combined_score,
        "news_score": news_score,
        "reddit_score": reddit_score,
        "fmp_news_score": fmp_score,
        "sentiment_confidence": sentiment_confidence,
        "article_count": total_articles,
        "active_sources": active_sources,
    }
    _sentiment_cache[cache_key] = (result, time.time())
    _persistent_cache.put(cache_key, result)
    _persistent_cache.save()
    return result


# ---------------------------------------------------------------------------
# Scoring — FinBERT primary, VADER fallback
# ---------------------------------------------------------------------------

def _score_texts(
    texts: list[str],
    timestamps: list[float | None] | None = None,
) -> tuple[float, list[dict]]:
    """Score texts using FinBERT (preferred) or VADER (fallback).

    Returns (avg_score, details) where score is in [-1, 1].
    When timestamps are provided and SENTIMENT_RECENCY_DECAY is enabled,
    applies exponential decay weighting (newer articles count more).
    """
    if not texts:
        return 0.0, []

    fb = _get_finbert()
    if fb is not None:
        return _score_texts_finbert(texts, fb, timestamps=timestamps)
    return _score_texts_vader(texts, timestamps=timestamps)


def _decay_weighted_avg(
    scores: list[float],
    timestamps: list[float | None] | None,
) -> float:
    """Compute decay-weighted average of scores. Falls back to simple average."""
    if not scores:
        return 0.0
    use_decay = (
        timestamps is not None
        and getattr(config, "SENTIMENT_RECENCY_DECAY", False)
    )
    if use_decay:
        weights = [_compute_decay_weight(timestamps[i] if i < len(timestamps) else None)
                   for i in range(len(scores))]
        total_w = sum(weights)
        if total_w > 0:
            return sum(s * w for s, w in zip(scores, weights)) / total_w
    return sum(scores) / len(scores)


def _score_texts_finbert(
    texts: list[str],
    fb,
    timestamps: list[float | None] | None = None,
) -> tuple[float, list[dict]]:
    """Run FinBERT on a list of texts. Returns (avg_score, details).

    FinBERT outputs labels: positive, negative, neutral with confidence scores.
    Maps to: positive → +confidence, negative → -confidence, neutral → 0.
    """
    scores = []
    details = []

    for i, raw_text in enumerate(texts[:20]):  # Limit batch size for memory
        text = raw_text[:2000]  # Hard truncate before tokenizer to prevent OOM
        _ts = timestamps[i] if timestamps and i < len(timestamps) else None
        _pub = datetime.utcfromtimestamp(_ts).isoformat() if _ts else None
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
            details.append({"title": text, "sentiment": round(score, 4), "published": _pub})
        except Exception:
            # If a single text fails, use VADER for just that text
            vs = _vader_analyzer.polarity_scores(text)
            scores.append(vs["compound"])
            details.append({"title": text, "sentiment": vs["compound"], "published": _pub})

    avg = _decay_weighted_avg(scores, timestamps)
    return avg, details


def _score_texts_vader(
    texts: list[str],
    timestamps: list[float | None] | None = None,
) -> tuple[float, list[dict]]:
    """Fallback: Run VADER on a list of texts. Returns (avg_score, details)."""
    scores = []
    details = []
    for i, text in enumerate(texts[:config.NEWS_HEADLINE_COUNT]):
        vs = _vader_analyzer.polarity_scores(text)
        compound = vs["compound"]
        scores.append(compound)
        _ts = timestamps[i] if timestamps and i < len(timestamps) else None
        _pub = datetime.utcfromtimestamp(_ts).isoformat() if _ts else None
        details.append({"title": text, "sentiment": compound, "published": _pub})

    avg = _decay_weighted_avg(scores, timestamps)
    return avg, details


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _compute_decay_weight(pub_timestamp: float | None) -> float:
    """Exponential decay weight based on article age. Returns 1.0 for unknown age."""
    if pub_timestamp is None:
        return 1.0
    half_life = getattr(config, "SENTIMENT_DECAY_HALF_LIFE_HOURS", 48.0)
    lam = 0.693147 / max(half_life, 1.0)  # ln(2) / half_life
    age_hours = max(0.0, (time.time() - pub_timestamp) / 3600.0)
    return math.exp(-lam * age_hours)


def _fetch_headlines(ticker: str, company_name: str) -> list[tuple[str, float | None]]:
    """Fetch headlines from Google News RSS with publication timestamps."""
    query = company_name if company_name else ticker
    query = query.replace(".L", "").strip()

    url = f"https://news.google.com/rss/search?q={query}+stock&hl=en&gl=US&ceid=US:en"
    try:
        feed = feedparser.parse(url)
        results = []
        for entry in feed.entries[:config.NEWS_HEADLINE_COUNT]:
            pub_ts = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    pub_ts = float(calendar.timegm(entry.published_parsed))
                except Exception:
                    pass
            results.append((entry.title, pub_ts))
        return results
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
        texts = []
        timestamps = []
        for a in articles:
            title = a.get("title", "")
            if not title:
                continue
            texts.append(title)
            pub_ts = None
            pd_str = a.get("publishedDate", "") or ""
            if pd_str:
                try:
                    pub_ts = datetime.strptime(pd_str[:19], "%Y-%m-%d %H:%M:%S").timestamp()
                except (ValueError, OSError):
                    pass
            timestamps.append(pub_ts)
        if not texts:
            return [], 0.0, []
        score, details = _score_texts(texts, timestamps=timestamps)
        for d in details:
            d["source"] = "fmp_news"
        return texts, score, details
    except Exception:
        return [], 0.0, []
