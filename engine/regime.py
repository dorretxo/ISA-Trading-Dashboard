"""VIX-based market regime detection and dynamic weight tilting.

Uses VIX percentile rank over 1-year history to classify the market as
BULL (low fear), NEUTRAL, or BEAR (high fear), then tilts pillar weights
accordingly.  In bull regimes, momentum/forecast factors outperform; in
bear regimes, fundamental/value factors offer a safety premium.
"""

import logging

import numpy as np
import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)

# Module-level cache — VIX history fetched once per session
_vix_cache: pd.Series | None = None


def _get_vix_history() -> pd.Series | None:
    """Fetch VIX close prices for the last VIX_HISTORY_DAYS days.

    Uses a separate cache from get_macro_data() which only fetches 90 days.
    """
    global _vix_cache
    if _vix_cache is not None:
        return _vix_cache

    try:
        days = getattr(config, "VIX_HISTORY_DAYS", 365)
        df = yf.download("^VIX", period=f"{days}d", progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            closes = df["Close"].dropna()
            # Handle MultiIndex columns from yfinance
            if hasattr(closes, "columns"):
                closes = closes.iloc[:, 0]
            _vix_cache = closes
            return _vix_cache
    except Exception as e:
        logger.warning("Failed to fetch VIX history: %s", e)

    return None


def get_vix_regime() -> dict:
    """Detect current market regime from VIX percentile rank.

    Returns dict with:
        vix_level: float — current VIX close
        vix_percentile: float — percentile rank (0-100) vs 1-year history
        regime_label: str — "BULL", "NEUTRAL", or "BEAR"
    """
    vix = _get_vix_history()
    if vix is None or len(vix) < 20:
        return {"vix_level": 0.0, "vix_percentile": 50.0, "regime_label": "NEUTRAL"}

    current_vix = float(vix.iloc[-1])
    vix_values = vix.values.astype(float)

    # Percentile: fraction of history where VIX was <= current level
    percentile = float(np.sum(vix_values <= current_vix) / len(vix_values) * 100)

    bull_threshold = getattr(config, "VIX_PERCENTILE_BULL", 25)
    bear_threshold = getattr(config, "VIX_PERCENTILE_BEAR", 75)

    if percentile < bull_threshold:
        label = "BULL"
    elif percentile > bear_threshold:
        label = "BEAR"
    else:
        label = "NEUTRAL"

    return {
        "vix_level": round(current_vix, 2),
        "vix_percentile": round(percentile, 1),
        "regime_label": label,
    }


def get_regime_adjusted_weights(base_weights: dict[str, float]) -> dict[str, float]:
    """Apply regime-based tilt to pillar weights.

    BULL regime:  +tilt to technical/forecast, -tilt to fundamental/sentiment
    BEAR regime:  +tilt to fundamental/sentiment, -tilt to technical/forecast
    NEUTRAL:      no change

    Respects WEIGHT_MIN_FLOOR and re-normalizes to sum=1.0.
    """
    try:
        regime = get_vix_regime()
    except Exception:
        return dict(base_weights)

    label = regime["regime_label"]
    if label == "NEUTRAL":
        return dict(base_weights)

    tilt = getattr(config, "REGIME_TILT_PCT", 0.05)
    min_floor = getattr(config, "WEIGHT_MIN_FLOOR", 0.10)

    adjusted = dict(base_weights)

    if label == "BULL":
        # Momentum/forecast outperform in low-vol regimes
        adjusted["technical"] = adjusted.get("technical", 0.25) + tilt
        adjusted["forecast"] = adjusted.get("forecast", 0.25) + tilt
        adjusted["fundamental"] = adjusted.get("fundamental", 0.25) - tilt
        adjusted["sentiment"] = adjusted.get("sentiment", 0.25) - tilt
    elif label == "BEAR":
        # Value/fundamental outperform in high-vol regimes
        adjusted["fundamental"] = adjusted.get("fundamental", 0.25) + tilt
        adjusted["sentiment"] = adjusted.get("sentiment", 0.25) + tilt
        adjusted["technical"] = adjusted.get("technical", 0.25) - tilt
        adjusted["forecast"] = adjusted.get("forecast", 0.25) - tilt

    # Clamp to minimum floor
    for key in adjusted:
        adjusted[key] = max(adjusted[key], min_floor)

    # Re-normalize to sum = 1.0
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: round(v / total, 4) for k, v in adjusted.items()}

    return adjusted
