"""Post-scoring risk overlay — adjusts scores and adds metadata flags.

Applied AFTER core pillar scoring, BEFORE final ranking / optimizer.
Three components:

1. Parabolic move penalty  — directly reduces aggregate score when a stock
   has experienced an extreme, likely-unsustainable price run-up.

2. Earnings proximity flag — metadata-only; signals elevated uncertainty
   around imminent earnings.  Does NOT change the score directionally.

3. Market cap confidence tier — metadata-only; classifies stocks into
   size tiers that downstream consumers (optimizer, discovery) use for
   position sizing and hurdle adjustments.

Public API:
    apply_risk_overlay(result, ticker) -> RiskOverlay
"""

import logging
from dataclasses import dataclass

import config
from utils.data_fetch import get_price_history, get_ticker_info

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Parabolic move thresholds — 90-day return that triggers penalty
PARABOLIC_THRESHOLD_90D = 1.00   # +100% in 90 days
PARABOLIC_THRESHOLD_30D = 0.60   # +60% in 30 days
PARABOLIC_MAX_PENALTY = 0.40     # Maximum score reduction

# Earnings proximity
EARNINGS_NEAR_DAYS = 14          # Flag when earnings within N days
EARNINGS_IMMINENT_DAYS = 5       # Stronger flag when very close
POST_EARNINGS_RECENCY_DAYS = 21  # Flag stocks that reported within N days

# 52-week high proximity
HIGH_52W_PROXIMITY_PCT = 0.05    # Within 5% of 52-week high

# Market cap tiers (USD)
MCAP_MEGA = 200_000_000_000     # >$200B
MCAP_LARGE = 10_000_000_000     # >$10B
MCAP_MID = 2_000_000_000        # >$2B
MCAP_SMALL = 300_000_000        # >$300M
# Below $300M = micro


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class RiskOverlay:
    """Risk overlay output — score adjustment + metadata flags."""

    # Parabolic move
    parabolic_penalty: float = 0.0     # Score reduction (0.0 = no penalty)
    return_90d: float | None = None
    return_30d: float | None = None
    is_parabolic: bool = False

    # Earnings proximity
    earnings_near: bool = False        # Within EARNINGS_NEAR_DAYS
    earnings_imminent: bool = False    # Within EARNINGS_IMMINENT_DAYS
    earnings_days: int | None = None   # Days until next earnings

    # Market cap tier
    market_cap: float | None = None
    cap_tier: str = "unknown"          # mega / large / mid / small / micro / unknown
    confidence_discount: float = 1.0   # 1.0 = full confidence, <1.0 = reduced
    max_weight_scale: float = 1.0      # Multiplier on MAX_POSITION_WEIGHT for optimizer

    # Post-earnings recency (reported within last N days)
    post_earnings_recent: bool = False
    post_earnings_days: int | None = None  # Days since last earnings
    earnings_miss: bool = False            # Actual < Expected
    earnings_miss_pct: float | None = None # (actual - expected) / |expected| as %

    # 52-week high proximity
    near_52w_high: bool = False
    pct_from_52w_high: float | None = None  # e.g. 0.03 = 3% below high


# ---------------------------------------------------------------------------
# Parabolic move detection
# ---------------------------------------------------------------------------

def _compute_parabolic_penalty(ticker: str, df=None) -> tuple[float, float | None, float | None, bool]:
    """Detect parabolic price moves and return a score penalty.

    Returns (penalty, return_90d, return_30d, is_parabolic).
    """
    try:
        if df is None:
            df = get_price_history(ticker)
        if df is None or df.empty or len(df) < 30:
            return 0.0, None, None, False

        close = df["Close"]
        current = float(close.iloc[-1])

        # 90-day return
        ret_90d = None
        if len(close) >= 63:
            price_90d_ago = float(close.iloc[-63])
            if price_90d_ago > 0:
                ret_90d = (current / price_90d_ago) - 1.0

        # 30-day return
        ret_30d = None
        if len(close) >= 21:
            price_30d_ago = float(close.iloc[-21])
            if price_30d_ago > 0:
                ret_30d = (current / price_30d_ago) - 1.0

        # Compute penalty — scaled linearly from threshold to 2x threshold
        penalty = 0.0
        is_parabolic = False

        if ret_90d is not None and ret_90d >= PARABOLIC_THRESHOLD_90D:
            is_parabolic = True
            # Linear scale: at threshold = 0.10, at 2x threshold = MAX_PENALTY
            excess = (ret_90d - PARABOLIC_THRESHOLD_90D) / PARABOLIC_THRESHOLD_90D
            penalty = max(penalty, 0.10 + min(excess, 1.0) * (PARABOLIC_MAX_PENALTY - 0.10))

        if ret_30d is not None and ret_30d >= PARABOLIC_THRESHOLD_30D:
            is_parabolic = True
            excess = (ret_30d - PARABOLIC_THRESHOLD_30D) / PARABOLIC_THRESHOLD_30D
            p30 = 0.10 + min(excess, 1.0) * (PARABOLIC_MAX_PENALTY - 0.10)
            penalty = max(penalty, p30)

        # Additional check: price far above analyst target (if available)
        # A stock trading at 2x+ analyst target is a strong reversion signal
        try:
            info = get_ticker_info(ticker)
            if info:
                try:
                    target = float(info.get("targetMeanPrice") or 0)
                    price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
                    num_analysts = int(info.get("numberOfAnalystOpinions") or 0)
                except (TypeError, ValueError):
                    target, price, num_analysts = 0, 0, 0
                if target and price and target > 0 and num_analysts >= 3:
                    overshoot = price / target
                    if overshoot >= 2.0:
                        is_parabolic = True
                        p_target = 0.15 + min((overshoot - 2.0) / 2.0, 1.0) * 0.15
                        penalty = max(penalty, p_target)
        except Exception:
            pass

        penalty = min(penalty, PARABOLIC_MAX_PENALTY)
        return penalty, ret_90d, ret_30d, is_parabolic

    except Exception as e:
        logger.warning("Parabolic check failed for %s: %s", ticker, e)
        return 0.0, None, None, False


# ---------------------------------------------------------------------------
# Earnings proximity detection
# ---------------------------------------------------------------------------

def _check_earnings_proximity(result: dict) -> tuple[bool, bool, int | None]:
    """Check if earnings are imminent using data already in the result dict.

    Returns (earnings_near, earnings_imminent, days_until).
    """
    days = result.get("earnings_proximity_days")
    if days is None:
        return False, False, None

    near = days <= EARNINGS_NEAR_DAYS
    imminent = days <= EARNINGS_IMMINENT_DAYS
    return near, imminent, days


# ---------------------------------------------------------------------------
# Post-earnings recency + earnings miss detection
# ---------------------------------------------------------------------------

def _check_post_earnings(ticker: str) -> tuple[bool, int | None, bool, float | None]:
    """Check if earnings were reported recently and whether they missed.

    Uses yfinance earnings data to detect:
    - Whether earnings were reported within POST_EARNINGS_RECENCY_DAYS
    - Whether actual EPS missed analyst estimates

    Returns (post_earnings_recent, days_since, is_miss, miss_pct).
    """
    try:
        import yfinance as yf
        from datetime import datetime, timedelta

        t = yf.Ticker(ticker)
        today = datetime.now().date()

        # Try to get earnings dates from calendar
        cal = None
        try:
            cal = t.calendar
        except Exception:
            pass

        # Try earnings_dates for historical data
        earnings_dates = None
        try:
            earnings_dates = t.earnings_dates
        except Exception:
            pass

        # Find most recent past earnings date
        recent_date = None
        days_since = None

        if earnings_dates is not None and not earnings_dates.empty:
            past_dates = earnings_dates.index[earnings_dates.index.date <= today]
            if len(past_dates) > 0:
                recent_date = past_dates[0]  # Most recent
                days_since = (today - recent_date.date()).days

        # Check if within recency window
        post_recent = days_since is not None and days_since <= POST_EARNINGS_RECENCY_DAYS

        # Check for earnings miss using earnings_dates columns
        is_miss = False
        miss_pct = None

        if earnings_dates is not None and not earnings_dates.empty and recent_date is not None:
            try:
                row = earnings_dates.loc[recent_date]
                # Column names vary: "Reported EPS" / "EPS Estimate"
                actual = None
                estimate = None
                for col in earnings_dates.columns:
                    col_lower = col.lower()
                    if "reported" in col_lower or "actual" in col_lower:
                        actual = row[col]
                    elif "estimate" in col_lower or "expected" in col_lower:
                        estimate = row[col]

                if actual is not None and estimate is not None:
                    import math
                    if not (math.isnan(actual) or math.isnan(estimate)):
                        if abs(estimate) > 0.001:
                            miss_pct = (actual - estimate) / abs(estimate) * 100
                            is_miss = actual < estimate
                        elif actual < estimate:
                            is_miss = True
                            miss_pct = -100.0  # Sentinel for near-zero denominator
            except Exception:
                pass

        return post_recent, days_since, is_miss, miss_pct

    except Exception as e:
        logger.warning("Post-earnings check failed for %s: %s", ticker, e)
        return False, None, False, None


# ---------------------------------------------------------------------------
# 52-week high proximity detection
# ---------------------------------------------------------------------------

def _check_52w_high_proximity(ticker: str, df=None) -> tuple[bool, float | None]:
    """Check if current price is near 52-week high.

    Returns (near_high, pct_from_high).
    """
    try:
        import yfinance as yf

        if df is None or df.empty:
            t = yf.Ticker(ticker)
            df = t.history(period="1y")

        if df is None or df.empty or len(df) < 20:
            return False, None

        close_col = "Close"
        if close_col not in df.columns:
            return False, None

        current = float(df[close_col].iloc[-1])
        high_52w = float(df[close_col].max())

        if high_52w <= 0:
            return False, None

        pct_from_high = (high_52w - current) / high_52w
        near_high = pct_from_high <= HIGH_52W_PROXIMITY_PCT

        return near_high, round(pct_from_high, 4)

    except Exception as e:
        logger.warning("52w high check failed for %s: %s", ticker, e)
        return False, None


# ---------------------------------------------------------------------------
# Market cap confidence tier
# ---------------------------------------------------------------------------

def _classify_market_cap(market_cap: float | None) -> tuple[str, float, float]:
    """Classify market cap into tiers and return confidence/sizing adjustments.

    Returns (tier_name, confidence_discount, max_weight_scale).
    """
    if market_cap is None:
        return "unknown", 0.80, 0.60
    try:
        market_cap = float(market_cap)
    except (TypeError, ValueError):
        return "unknown", 0.80, 0.60

    if market_cap >= MCAP_MEGA:
        return "mega", 1.00, 1.00
    elif market_cap >= MCAP_LARGE:
        return "large", 1.00, 1.00
    elif market_cap >= MCAP_MID:
        return "mid", 0.95, 0.90
    elif market_cap >= MCAP_SMALL:
        return "small", 0.85, 0.70
    else:
        return "micro", 0.70, 0.50


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_risk_overlay(result: dict, ticker: str, df=None) -> RiskOverlay:
    """Apply the full risk overlay to a scored result.

    Args:
        result: Dict from analyse_holding() or discovery scoring —
                expected to contain 'earnings_proximity_days', '_market_cap'
                or 'market_cap', etc.
        ticker: Stock ticker symbol.
        df: Optional pre-loaded price DataFrame (avoids re-fetch).

    Returns:
        RiskOverlay with all flags and adjustments computed.
    """
    overlay = RiskOverlay()

    # 1. Parabolic move
    penalty, ret_90d, ret_30d, is_parabolic = _compute_parabolic_penalty(ticker, df=df)
    overlay.parabolic_penalty = round(penalty, 4)
    overlay.return_90d = round(ret_90d, 4) if ret_90d is not None else None
    overlay.return_30d = round(ret_30d, 4) if ret_30d is not None else None
    overlay.is_parabolic = is_parabolic

    # 2. Earnings proximity
    near, imminent, days = _check_earnings_proximity(result)
    overlay.earnings_near = near
    overlay.earnings_imminent = imminent
    overlay.earnings_days = days

    # 3. Market cap tier
    mcap = result.get("_market_cap") or result.get("market_cap")
    tier, confidence, weight_scale = _classify_market_cap(mcap)
    overlay.market_cap = mcap
    overlay.cap_tier = tier
    overlay.confidence_discount = confidence
    overlay.max_weight_scale = weight_scale

    # 4. Post-earnings recency + earnings miss
    post_recent, days_since, is_miss, miss_pct = _check_post_earnings(ticker)
    overlay.post_earnings_recent = post_recent
    overlay.post_earnings_days = days_since
    overlay.earnings_miss = is_miss
    overlay.earnings_miss_pct = round(miss_pct, 1) if miss_pct is not None else None

    # 5. 52-week high proximity
    near_high, pct_from_high = _check_52w_high_proximity(ticker, df=df)
    overlay.near_52w_high = near_high
    overlay.pct_from_52w_high = pct_from_high

    # --- Logging ---
    if is_parabolic:
        logger.info(
            "%s: parabolic move detected (90d: %s, 30d: %s) — penalty %.2f",
            ticker,
            f"{ret_90d:+.0%}" if ret_90d is not None else "N/A",
            f"{ret_30d:+.0%}" if ret_30d is not None else "N/A",
            penalty,
        )
    if imminent:
        logger.info("%s: earnings imminent (%d days)", ticker, days)
    if post_recent:
        miss_str = f", MISS {miss_pct:+.1f}%" if is_miss and miss_pct is not None else ""
        logger.info("%s: reported earnings %d days ago%s", ticker, days_since, miss_str)
    if near_high:
        logger.info("%s: within %.1f%% of 52-week high", ticker, (pct_from_high or 0) * 100)
    if tier in ("small", "micro"):
        logger.info("%s: %s cap (confidence %.0f%%, max weight scale %.0f%%)",
                    ticker, tier, confidence * 100, weight_scale * 100)

    return overlay
