"""Financial Modeling Prep (FMP) API client with rate limiting and tiered caching.

FMP is the PRIMARY data source (Starter plan: 300 calls/minute).
If FMP_API_KEY is empty, every function returns None and the system
falls back to yfinance-only behaviour.
"""

import logging
import time
from collections import deque
from datetime import date
from typing import Any

import requests

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Starter plan: non-US ticker detection
# ---------------------------------------------------------------------------
# FMP Starter plan returns 402 for non-US tickers on most endpoints
# (fundamentals, technicals, grades). Only /news/stock and /profile work.
# Suffixes that indicate non-US exchanges:
_NON_US_SUFFIXES = (
    ".L", ".DE", ".MC", ".PA", ".AS", ".MI", ".BR", ".SW", ".HK",
    ".TO", ".AX", ".T", ".SI", ".KS", ".BO", ".NS", ".SA",
    ".F", ".VX", ".ST", ".HE", ".CO", ".OL", ".WA", ".LS",
    ".VI", ".IS",
)


def _is_non_us_ticker(ticker: str) -> bool:
    """Check if ticker has a non-US exchange suffix."""
    return any(ticker.upper().endswith(s) for s in _NON_US_SUFFIXES)


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# Sliding window rate limiter: track timestamps of recent API calls
_call_timestamps: deque[float] = deque()
_cache: dict[str, tuple[Any, float, float]] = {}  # key -> (data, timestamp, ttl)
_total_calls_today: int = 0  # For UI display only
_calls_today_date: str = ""


# ---------------------------------------------------------------------------
# Rate limiting — sliding window (300 calls per 60-second window)
# ---------------------------------------------------------------------------

def _prune_old_timestamps() -> None:
    """Remove timestamps older than 60 seconds from the sliding window."""
    cutoff = time.time() - 60.0
    while _call_timestamps and _call_timestamps[0] < cutoff:
        _call_timestamps.popleft()


def _record_call() -> None:
    """Record a new API call in the sliding window and daily counter."""
    global _total_calls_today, _calls_today_date
    now = time.time()
    _call_timestamps.append(now)
    # Daily counter for UI display
    today = date.today().isoformat()
    if _calls_today_date != today:
        _total_calls_today = 0
        _calls_today_date = today
    _total_calls_today += 1


def _wait_if_rate_limited() -> bool:
    """If at rate limit, sleep briefly and retry. Returns True if OK to proceed."""
    _prune_old_timestamps()
    limit = getattr(config, "FMP_RATE_LIMIT_PER_MIN", 300)
    if len(_call_timestamps) < limit:
        return True
    # At limit — wait for oldest call to expire from window
    wait_time = _call_timestamps[0] + 60.0 - time.time() + 0.1
    if wait_time > 0:
        logger.info("FMP rate limit reached, waiting %.1fs", wait_time)
        time.sleep(min(wait_time, 5.0))  # Cap wait at 5s
    _prune_old_timestamps()
    return len(_call_timestamps) < limit


def get_remaining_budget() -> int:
    """Return remaining FMP API calls in the current 60-second window."""
    _prune_old_timestamps()
    limit = getattr(config, "FMP_RATE_LIMIT_PER_MIN", 300)
    return max(0, limit - len(_call_timestamps))


def get_calls_today() -> int:
    """Return total FMP calls made today (for UI display)."""
    global _total_calls_today, _calls_today_date
    today = date.today().isoformat()
    if _calls_today_date != today:
        _total_calls_today = 0
        _calls_today_date = today
    return _total_calls_today


def is_available() -> bool:
    """Check if FMP is configured and has rate budget remaining."""
    key = getattr(config, "FMP_API_KEY", "")
    if not key:
        return False
    _prune_old_timestamps()
    limit = getattr(config, "FMP_RATE_LIMIT_PER_MIN", 300)
    return len(_call_timestamps) < limit


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _cache_key(path: str, params: dict) -> str:
    sorted_params = sorted((k, v) for k, v in params.items() if k != "apikey")
    return f"{path}|{sorted_params}"


def _cache_get(key: str) -> Any | None:
    if key in _cache:
        data, ts, ttl = _cache[key]
        if time.time() - ts < ttl:
            return data
        del _cache[key]
    return None


def _cache_set(key: str, data: Any, ttl: float) -> None:
    _cache[key] = (data, time.time(), ttl)


def clear_cache() -> None:
    """Clear all cached FMP data."""
    _cache.clear()


# ---------------------------------------------------------------------------
# Core HTTP wrapper
# ---------------------------------------------------------------------------

def _fmp_get(path: str, params: dict | None = None, ttl: float = 3600) -> Any | None:
    """Fetch from FMP API with caching + sliding-window rate limiting.

    Returns parsed JSON (dict or list) or None on any failure.
    """
    api_key = getattr(config, "FMP_API_KEY", "")
    if not api_key:
        return None

    if params is None:
        params = {}

    # Check cache first (no rate limit cost)
    key = _cache_key(path, params)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    # Check / wait for rate limit
    if not _wait_if_rate_limited():
        logger.warning("FMP rate limit still exceeded after wait")
        return None

    # Make request
    url = f"{config.FMP_BASE_URL}{path}"
    params["apikey"] = api_key

    # Retry with exponential backoff for transient failures
    _MAX_RETRIES = 3
    _BACKOFF_BASE = 2.0  # 2s, 4s, 8s

    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=15)
            _record_call()

            if resp.status_code != 200:
                # Suppress noise for known Starter plan limitations
                _known_unavailable = ("/earnings-surprises", "/sector-pe-ratio")
                if path in _known_unavailable and resp.status_code in (402, 404):
                    logger.debug("FMP %s not available on Starter plan (status %d)", path, resp.status_code)
                    return None  # Permanent — don't retry
                if resp.status_code == 429:
                    # Rate-limited — backoff and retry
                    wait = _BACKOFF_BASE ** (attempt + 1)
                    logger.debug("FMP rate limited on %s, backing off %.1fs", path, wait)
                    time.sleep(wait)
                    continue
                if resp.status_code >= 500:
                    # Server error — transient, retry
                    wait = _BACKOFF_BASE ** (attempt + 1)
                    logger.debug("FMP server error %d on %s, retry %d/%d",
                                 resp.status_code, path, attempt + 1, _MAX_RETRIES)
                    time.sleep(wait)
                    continue
                logger.warning("FMP %s returned status %d", path, resp.status_code)
                return None  # Other client errors — permanent

            data = resp.json()

            # FMP returns error messages as dicts with "Error Message" key
            if isinstance(data, dict) and "Error Message" in data:
                logger.warning("FMP error for %s: %s", path, data["Error Message"])
                return None

            _cache_set(key, data, ttl)
            return data

        except (requests.ConnectionError, requests.Timeout) as e:
            # Transient network error — retry with backoff
            wait = _BACKOFF_BASE ** (attempt + 1)
            logger.debug("FMP transient error on %s (attempt %d/%d): %s",
                         path, attempt + 1, _MAX_RETRIES, e)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(wait)
                continue
            logger.warning("FMP request failed for %s after %d retries: %s",
                           path, _MAX_RETRIES, e)
            return None
        except requests.RequestException as e:
            logger.warning("FMP request failed for %s: %s", path, e)
            return None
        except ValueError:
            logger.warning("FMP returned invalid JSON for %s", path)
            return None

    return None


# ---------------------------------------------------------------------------
# Endpoint wrappers — Fundamental data
# ---------------------------------------------------------------------------

def get_earnings_surprises(ticker: str, limit: int = 8) -> list[dict] | None:
    """Actual vs estimated EPS for recent quarters.

    NOTE: /earnings-surprises returns 404 on the Starter plan.
    We synthesize surprise data from the income statement (actual EPS)
    and analyst estimates (estimated EPS) instead.  Returns None if
    the required data is unavailable.  Non-US tickers are not supported
    on the Starter plan for fundamentals.
    """
    if _is_non_us_ticker(ticker):
        return None

    # Try the native endpoint first (may work on higher-tier plans)
    data = _fmp_get(
        "/earnings-surprises",
        {"symbol": ticker, "limit": limit},
        ttl=config.FMP_CACHE_TTL_QUARTERLY,
    )
    if isinstance(data, list) and data:
        return data

    # Fallback: synthesize from income statement + analyst estimates
    income = get_income_statement(ticker, limit=limit)
    estimates = get_analyst_estimates(ticker, limit=limit)
    if not income:
        return None

    # Build a lookup of estimated EPS by date (annual periods)
    est_by_date = {}
    if estimates:
        for e in estimates:
            est_by_date[e.get("date", "")] = e.get("estimatedEpsAvg", e.get("epsEstimated"))

    surprises = []
    for stmt in income:
        actual_eps = stmt.get("eps")
        stmt_date = stmt.get("date", "")
        estimated_eps = est_by_date.get(stmt_date)
        if actual_eps is not None:
            surprises.append({
                "symbol": ticker,
                "date": stmt_date,
                "actualEarningResult": actual_eps,
                "estimatedEarning": estimated_eps,
            })
    return surprises if surprises else None


def get_analyst_estimates(
    ticker: str, period: str = "annual", limit: int = 4,
) -> list[dict] | None:
    """Forward EPS/revenue estimates from analysts.

    NOTE: Starter plan only supports period='annual' and US tickers only.
    """
    if _is_non_us_ticker(ticker):
        return None
    # Force annual on Starter plan — quarterly is premium-only
    safe_period = "annual" if period in ("quarter", "quarterly") else period
    data = _fmp_get(
        "/analyst-estimates",
        {"symbol": ticker, "period": safe_period, "limit": limit},
        ttl=config.FMP_CACHE_TTL_QUARTERLY,
    )
    return data if isinstance(data, list) else None


def get_upgrades_downgrades(ticker: str) -> list[dict] | None:
    """Analyst rating consensus (buy/hold/sell counts).

    NOTE: /upgrades-downgrades returns 404 on Starter plan.
    Replaced with /grades-consensus which returns aggregated counts.
    Non-US tickers not supported on Starter plan.
    """
    if _is_non_us_ticker(ticker):
        return None
    consensus = _fmp_get(
        "/grades-consensus",
        {"symbol": ticker},
        ttl=config.FMP_CACHE_TTL_QUARTERLY,
    )

    # Also fetch price targets for additional context
    price_targets = _fmp_get(
        "/price-target-consensus",
        {"symbol": ticker},
        ttl=config.FMP_CACHE_TTL_QUARTERLY,
    )

    if isinstance(consensus, list) and consensus:
        result = consensus[0]
        # Merge price target data if available
        if isinstance(price_targets, list) and price_targets:
            result["priceTarget"] = price_targets[0]
        return [result]

    return None


def get_key_metrics(
    ticker: str, period: str = "annual", limit: int = 8,
) -> list[dict] | None:
    """Key financial metrics (PEG, ROE, margins, etc.).

    NOTE: Starter plan: annual only, US tickers only.
    """
    if _is_non_us_ticker(ticker):
        return None
    # Starter plan: omit period param entirely (defaults to annual)
    # or explicitly set 'annual'. 'quarter' is premium-only.
    params: dict[str, Any] = {"symbol": ticker, "limit": limit}
    if period not in ("quarter", "quarterly"):
        params["period"] = period
    # else: omit period → defaults to annual on Starter plan

    data = _fmp_get(
        "/key-metrics",
        params,
        ttl=config.FMP_CACHE_TTL_QUARTERLY,
    )
    return data if isinstance(data, list) else None


def get_income_statement(
    ticker: str, period: str = "annual", limit: int = 8,
) -> list[dict] | None:
    """Income statement with EPS, revenue, margins.

    NOTE: US tickers only on Starter plan.
    """
    if _is_non_us_ticker(ticker):
        return None
    data = _fmp_get(
        "/income-statement",
        {"symbol": ticker, "period": period, "limit": limit},
        ttl=config.FMP_CACHE_TTL_QUARTERLY,
    )
    return data if isinstance(data, list) else None


def get_company_profile(ticker: str) -> dict | None:
    """Company overview: sector, industry, market cap, description."""
    data = _fmp_get(
        "/profile",
        {"symbol": ticker},
        ttl=config.FMP_CACHE_TTL_QUARTERLY,
    )
    # Profile returns a list with one item
    if isinstance(data, list) and data:
        return data[0]
    return data if isinstance(data, dict) else None


def get_sector_pe(sector: str) -> float | None:
    """Average P/E ratio for a sector (used for relative valuation).

    NOTE: /sector-pe-ratio returns 404 on Starter plan.
    Returns None — callers should fall back to yfinance sector P/E
    or skip relative valuation when unavailable.
    """
    if not sector:
        return None
    # Try the endpoint (works on higher-tier plans)
    data = _fmp_get(
        "/sector-pe-ratio",
        {"date": date.today().isoformat(), "exchange": "NYSE"},
        ttl=config.FMP_CACHE_TTL_QUARTERLY,
    )
    if not isinstance(data, list) or not data:
        return None
    for entry in data:
        if entry.get("sector", "").lower() == sector.lower():
            try:
                return float(entry["pe"])
            except (KeyError, ValueError, TypeError):
                pass
    return None


def get_financial_ratios(
    ticker: str, period: str = "annual", limit: int = 4,
) -> list[dict] | None:
    """Financial ratios (P/E, P/B, ROE, margins, etc.).

    NOTE: Starter plan: annual only, US tickers only.
    """
    if _is_non_us_ticker(ticker):
        return None
    # Force annual on Starter plan — quarterly is premium-only
    params: dict[str, Any] = {"symbol": ticker, "limit": limit}
    if period not in ("quarter", "quarterly"):
        params["period"] = period
    # else: omit period → defaults to annual

    data = _fmp_get(
        "/ratios",
        params,
        ttl=config.FMP_CACHE_TTL_QUARTERLY,
    )
    return data if isinstance(data, list) else None


# ---------------------------------------------------------------------------
# Endpoint wrappers — News & Calendar
# ---------------------------------------------------------------------------

def get_stock_news(ticker: str, limit: int = 20) -> list[dict] | None:
    """Recent news articles for a ticker (default 20 for FinBERT accuracy).

    NOTE: /stock-news returns 404 on Starter plan.
    Correct path is /news/stock with 'tickers' parameter.
    """
    data = _fmp_get(
        "/news/stock",
        {"tickers": ticker, "limit": limit},
        ttl=config.FMP_CACHE_TTL_DAILY,
    )
    return data if isinstance(data, list) else None


def get_earnings_calendar(ticker: str) -> list[dict] | None:
    """Upcoming and recent earnings dates for a specific ticker.

    The FMP /earnings-calendar endpoint returns a global calendar
    (ignores the symbol param on Starter plan), so we must filter
    the results to entries matching the requested ticker.
    """
    # Normalize ticker for matching (FMP strips exchange suffixes)
    base_symbol = ticker.split(".")[0].upper()

    data = _fmp_get(
        "/earnings-calendar",
        {},  # symbol param is ignored on Starter — don't waste it
        ttl=config.FMP_CACHE_TTL_CALENDAR,
    )
    if not isinstance(data, list):
        return None

    # Filter to entries matching this ticker
    filtered = [
        e for e in data
        if e.get("symbol", "").upper() == base_symbol
        or e.get("symbol", "").upper() == ticker.upper()
    ]
    return filtered if filtered else None


# ---------------------------------------------------------------------------
# Endpoint wrappers — Technical indicators
# ---------------------------------------------------------------------------

def get_technical_indicator(
    ticker: str, indicator: str, period: int = 14,
) -> list[dict] | None:
    """Fetch a pre-computed technical indicator from FMP.

    Supported indicators: sma, ema, rsi, adx, williams, dema, tema, wma.
    Returns list of dicts with date + indicator value, newest first.
    NOTE: US tickers only on Starter plan.
    """
    if _is_non_us_ticker(ticker):
        return None
    data = _fmp_get(
        f"/technical-indicators/{indicator}",
        {
            "symbol": ticker,
            "periodLength": period,
            "timeframe": "1day",
        },
        ttl=config.FMP_CACHE_TTL_DAILY,
    )
    return data if isinstance(data, list) else None


# ---------------------------------------------------------------------------
# Endpoint wrappers — Screener & Discovery
# ---------------------------------------------------------------------------

def screen_stocks(
    exchange: str,
    market_cap_min: int | None = None,
    market_cap_max: int | None = None,
    volume_min: int | None = None,
    limit: int = 200,
) -> list[dict] | None:
    """Screen stocks on a single exchange using FMP company screener.

    Returns list of dicts with: symbol, companyName, price, marketCap, beta,
    volume, lastAnnualDividend, sector, industry, country, exchangeShortName.
    """
    params: dict[str, Any] = {
        "exchange": exchange,
        "isEtf": "false",
        "isFund": "false",
        "isActivelyTrading": "true",
        "limit": limit,
    }
    if market_cap_min is not None:
        params["marketCapMoreThan"] = market_cap_min
    if market_cap_max is not None:
        params["marketCapLowerThan"] = market_cap_max
    if volume_min is not None:
        params["volumeMoreThan"] = volume_min

    data = _fmp_get(
        "/company-screener",
        params,
        ttl=config.FMP_CACHE_TTL_DAILY,  # 1h cache — screener results change intraday
    )
    return data if isinstance(data, list) else None


def get_batch_quotes(tickers: list[str]) -> list[dict] | None:
    """Fetch real-time quotes for multiple tickers in one call.

    FMP supports comma-separated symbols in the /quote endpoint.
    Returns list of quote dicts with: symbol, price, volume, marketCap, etc.
    """
    if not tickers:
        return None
    # FMP batch quote supports up to ~50 symbols per call
    symbol_str = ",".join(tickers[:50])
    data = _fmp_get(
        "/quote",
        {"symbol": symbol_str},
        ttl=300,  # 5-minute cache for quotes
    )
    return data if isinstance(data, list) else None
