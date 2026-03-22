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
PARABOLIC_THRESHOLD_90D = 2.00   # +200% in 90 days
PARABOLIC_THRESHOLD_30D = 1.00   # +100% in 30 days
PARABOLIC_MAX_PENALTY = 0.40     # Maximum score reduction

# Earnings proximity
EARNINGS_NEAR_DAYS = 14          # Flag when earnings within N days
EARNINGS_IMMINENT_DAYS = 5       # Stronger flag when very close

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
                target = info.get("targetMeanPrice")
                price = info.get("currentPrice") or info.get("regularMarketPrice")
                num_analysts = info.get("numberOfAnalystOpinions", 0) or 0
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
# Market cap confidence tier
# ---------------------------------------------------------------------------

def _classify_market_cap(market_cap: float | None) -> tuple[str, float, float]:
    """Classify market cap into tiers and return confidence/sizing adjustments.

    Returns (tier_name, confidence_discount, max_weight_scale).
    """
    if market_cap is None:
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
    if tier in ("small", "micro"):
        logger.info("%s: %s cap (confidence %.0f%%, max weight scale %.0f%%)",
                    ticker, tier, confidence * 100, weight_scale * 100)

    return overlay
