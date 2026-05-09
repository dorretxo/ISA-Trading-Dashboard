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
    """Fetch VIX close prices, reusing the shared macro cache when possible."""
    global _vix_cache
    if _vix_cache is not None:
        return _vix_cache

    # Try shared macro cache first (same 365-day lookback)
    try:
        from utils.data_fetch import get_macro_data
        macro = get_macro_data()
        if "vix" in macro and not macro["vix"].empty:
            closes = macro["vix"]["Close"].dropna()
            if hasattr(closes, "columns"):
                closes = closes.iloc[:, 0]
            if len(closes) >= 20:
                _vix_cache = closes
                return _vix_cache
    except Exception:
        pass

    # Fallback: direct download
    try:
        days = getattr(config, "VIX_HISTORY_DAYS", 365)
        df = yf.download("^VIX", period=f"{days}d", progress=False,
                         auto_adjust=True, timeout=30)
        if df is not None and not df.empty:
            closes = df["Close"].dropna()
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

    try:
        from engine.pillar_weighting import apply_weight_guardrails
        adjusted = apply_weight_guardrails(adjusted, horizon="90d")
    except Exception:
        pass

    return adjusted


# ---------------------------------------------------------------------------
# Multi-Macro Regime Model (Arnott, Harvey, Kalesnik & Linnainmaa 2021)
# ---------------------------------------------------------------------------

def get_multi_macro_regime() -> dict:
    """4-state macro regime model using term spread, credit spread, and market trend.

    States:
    - RISK_ON: term spread positive + credit tight + SPY uptrend → momentum
    - RISK_OFF: inverted yield curve OR wide credit spread + SPY downtrend → quality + low-vol
    - TRANSITION_UP: improving macro (narrowing credit, steepening curve) → value + momentum
    - TRANSITION_DOWN: deteriorating macro → quality + reversal

    Returns dict with regime_label, factor_tilts, and component signals.
    """
    try:
        from utils.data_fetch import get_macro_data
        macro = get_macro_data()
    except Exception as e:
        logger.debug("Multi-macro regime: macro data unavailable: %s", e)
        return _default_regime()

    # --- Extract macro signals ---
    term_spread = _compute_term_spread(macro)
    credit_spread = _compute_credit_spread(macro)
    market_trend = _compute_market_trend(macro)

    # --- 4-state classification ---
    term_expansion = getattr(config, "MACRO_TERM_SPREAD_EXPANSION", 1.0)
    term_contraction = getattr(config, "MACRO_TERM_SPREAD_CONTRACTION", 0.0)
    credit_tight = getattr(config, "MACRO_CREDIT_SPREAD_TIGHT", 0.02)
    credit_wide = getattr(config, "MACRO_CREDIT_SPREAD_WIDE", 0.05)

    risk_on_signals = 0
    risk_off_signals = 0

    if term_spread is not None:
        if term_spread > term_expansion:
            risk_on_signals += 1
        elif term_spread < term_contraction:
            risk_off_signals += 1

    if credit_spread is not None:
        if credit_spread < credit_tight:
            risk_on_signals += 1
        elif credit_spread > credit_wide:
            risk_off_signals += 1

    if market_trend is not None:
        if market_trend > 0.05:  # >5% trailing return
            risk_on_signals += 1
        elif market_trend < -0.05:
            risk_off_signals += 1

    # Determine regime
    if risk_on_signals >= 2 and risk_off_signals == 0:
        regime = "RISK_ON"
    elif risk_off_signals >= 2 and risk_on_signals == 0:
        regime = "RISK_OFF"
    elif risk_on_signals > risk_off_signals:
        regime = "TRANSITION_UP"
    elif risk_off_signals > risk_on_signals:
        regime = "TRANSITION_DOWN"
    else:
        regime = "NEUTRAL"

    # --- Factor tilts per regime (Arnott et al. 2021) ---
    tilts = _REGIME_FACTOR_TILTS.get(regime, {})

    result = {
        "regime_label": regime,
        "factor_tilts": tilts,
        "term_spread": term_spread,
        "credit_spread": credit_spread,
        "market_trend": market_trend,
        "risk_on_signals": risk_on_signals,
        "risk_off_signals": risk_off_signals,
    }
    logger.info("Multi-macro regime: %s (on=%d, off=%d, term=%.3f, credit=%.3f, mkt=%.3f)",
                regime, risk_on_signals, risk_off_signals,
                term_spread or 0, credit_spread or 0, market_trend or 0)
    return result


# Regime-conditional factor tilts (percentage adjustments)
_REGIME_FACTOR_TILTS = {
    "RISK_ON": {
        "momentum_tilt": 0.15,
        "quality_tilt": -0.05,
        "investment_tilt": 0.0,
        "volatility_tilt": -0.05,
        "reversal_tilt": -0.05,
    },
    "RISK_OFF": {
        "momentum_tilt": -0.15,
        "quality_tilt": 0.15,
        "investment_tilt": 0.05,
        "volatility_tilt": 0.10,
        "reversal_tilt": 0.05,
    },
    "TRANSITION_UP": {
        "momentum_tilt": 0.05,
        "quality_tilt": 0.0,
        "investment_tilt": 0.10,
        "volatility_tilt": 0.0,
        "reversal_tilt": 0.0,
    },
    "TRANSITION_DOWN": {
        "momentum_tilt": -0.05,
        "quality_tilt": 0.10,
        "investment_tilt": 0.0,
        "volatility_tilt": 0.05,
        "reversal_tilt": 0.05,
    },
    "NEUTRAL": {
        "momentum_tilt": 0.0,
        "quality_tilt": 0.0,
        "investment_tilt": 0.0,
        "volatility_tilt": 0.0,
        "reversal_tilt": 0.0,
    },
}


def _default_regime() -> dict:
    """Return neutral regime when data is unavailable."""
    return {
        "regime_label": "NEUTRAL",
        "factor_tilts": _REGIME_FACTOR_TILTS["NEUTRAL"],
        "term_spread": None,
        "credit_spread": None,
        "market_trend": None,
        "risk_on_signals": 0,
        "risk_off_signals": 0,
    }


def _compute_term_spread(macro: dict) -> float | None:
    """10Y yield - 2Y yield proxy (term structure slope)."""
    try:
        tnx = macro.get("bonds_10y")
        irx = macro.get("bonds_2y")
        if tnx is not None and not tnx.empty and irx is not None and not irx.empty:
            tnx_close = tnx["Close"].dropna()
            irx_close = irx["Close"].dropna()
            if hasattr(tnx_close, "columns"):
                tnx_close = tnx_close.iloc[:, 0]
            if hasattr(irx_close, "columns"):
                irx_close = irx_close.iloc[:, 0]
            if len(tnx_close) > 0 and len(irx_close) > 0:
                # TNX is in percentage points, IRX is 13-week T-bill rate
                return float(tnx_close.iloc[-1] - irx_close.iloc[-1])
    except Exception:
        pass
    return None


def _compute_credit_spread(macro: dict) -> float | None:
    """HY - IG spread proxy (credit risk appetite)."""
    try:
        hy = macro.get("hy_spread")
        ig = macro.get("ig_spread")
        if hy is not None and not hy.empty and ig is not None and not ig.empty:
            hy_close = hy["Close"].dropna()
            ig_close = ig["Close"].dropna()
            if hasattr(hy_close, "columns"):
                hy_close = hy_close.iloc[:, 0]
            if hasattr(ig_close, "columns"):
                ig_close = ig_close.iloc[:, 0]
            if len(hy_close) > 0 and len(ig_close) > 0:
                # Proxy: YTD return difference (negative = HY underperforming = wider spread)
                hy_ret = float(hy_close.iloc[-1] / hy_close.iloc[0] - 1) if len(hy_close) > 60 else 0
                ig_ret = float(ig_close.iloc[-1] / ig_close.iloc[0] - 1) if len(ig_close) > 60 else 0
                return abs(ig_ret - hy_ret)  # Wider gap = more risk aversion
    except Exception:
        pass
    return None


def _compute_market_trend(macro: dict) -> float | None:
    """SPY trailing 12-1 month return (market trend signal)."""
    try:
        spy = macro.get("spy")
        if spy is not None and not spy.empty:
            closes = spy["Close"].dropna()
            if hasattr(closes, "columns"):
                closes = closes.iloc[:, 0]
            if len(closes) >= 252:
                # 12-month return minus most recent month (skip short-term reversal)
                ret_12m = float(closes.iloc[-1] / closes.iloc[-252] - 1)
                ret_1m = float(closes.iloc[-1] / closes.iloc[-21] - 1)
                return ret_12m - ret_1m
            elif len(closes) >= 63:
                return float(closes.iloc[-1] / closes.iloc[-63] - 1)
    except Exception:
        pass
    return None
