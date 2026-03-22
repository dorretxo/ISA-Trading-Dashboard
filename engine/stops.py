"""Stop-loss and take-profit calculators with volatility-adjusted parameters."""

import numpy as np
import pandas as pd

import config
from utils.data_fetch import get_price_history


# ---------------------------------------------------------------------------
# Volatility estimation
# ---------------------------------------------------------------------------

def _realized_volatility(ticker: str) -> tuple[float | None, float | None]:
    """Calculate realized annualized volatility and its percentile rank.

    Returns (annualized_vol, vol_percentile) where:
        annualized_vol: float — annualized volatility as decimal (e.g., 0.35 = 35%)
        vol_percentile: float — percentile of current vol vs its own 1-year rolling history (0-100)
    """
    df = get_price_history(ticker)
    if df.empty or len(df) < config.VOL_LOOKBACK + 5:
        return None, None

    closes = df["Close"].values.astype(float)
    log_returns = np.diff(np.log(closes))

    # Current realized vol (annualized)
    recent_returns = log_returns[-config.VOL_LOOKBACK:]
    current_vol = float(np.std(recent_returns)) * np.sqrt(252)

    # Rolling vol history for percentile ranking (1-year of rolling windows)
    lookback = config.VOL_LOOKBACK
    n = len(log_returns)
    if n < lookback + 60:
        # Not enough history for percentile — use 50th as default
        return current_vol, 50.0

    vol_history = []
    for i in range(lookback, min(n, lookback + 252)):
        window = log_returns[i - lookback:i]
        vol_history.append(float(np.std(window)) * np.sqrt(252))

    # Percentile rank of current vol within its history
    percentile = float(np.sum(np.array(vol_history) <= current_vol) / len(vol_history) * 100)

    return current_vol, percentile


def _dynamic_atr_multiplier(vol_percentile: float | None) -> float:
    """Interpolate ATR multiplier based on volatility percentile.

    Low vol → tighter stop (1.5x ATR), high vol → wider stop (3.0x ATR).
    """
    if vol_percentile is None:
        return config.ATR_MULTIPLIER  # Fallback to fixed default

    return float(np.interp(
        vol_percentile,
        [20, 80],
        [config.ATR_MULT_LOW_VOL, config.ATR_MULT_HIGH_VOL],
    ))


def _dynamic_trailing_pct(vol_percentile: float | None) -> float:
    """Interpolate trailing stop percentage based on volatility percentile.

    Low vol → tighter trail (6%), high vol → wider trail (12%).
    """
    if vol_percentile is None:
        return config.TRAILING_STOP_PCT  # Fallback to fixed default

    return float(np.interp(
        vol_percentile,
        [20, 80],
        [config.TRAIL_PCT_LOW_VOL, config.TRAIL_PCT_HIGH_VOL],
    ))


# ---------------------------------------------------------------------------
# Stop-loss
# ---------------------------------------------------------------------------

def calculate_stop_loss(
    ticker: str, atr: float | None, current_price: float | None,
    sma_200: float | None = None,
) -> dict:
    """Calculate stop-loss using volatility-adjusted ATR, trailing, and SMA-200 methods.

    The stop-loss is always below current_price. The tightest valid stop
    (highest value that is still below current price) wins.
    """
    if current_price is None or current_price <= 0:
        return {"stop_loss": None, "method": "N/A"}

    df = get_price_history(ticker)

    # Compute volatility for dynamic parameters
    vol, vol_pct = _realized_volatility(ticker)
    atr_mult = _dynamic_atr_multiplier(vol_pct)
    trail_pct = _dynamic_trailing_pct(vol_pct)

    candidates = {}

    # Method 1: Volatility-adjusted ATR stop (always relative to current price)
    if atr is not None and atr > 0:
        atr_stop = current_price - (atr * atr_mult)
        candidates["ATR"] = atr_stop

    # Method 2: Trailing stop from recent high — but cap the reference price
    # If the stock has fallen far from the peak, use current_price as the
    # anchor instead so the stop stays below the current level.
    if not df.empty:
        recent_high = float(df["High"].tail(63).max())
        # Cap reference at current_price so the stop can never exceed it
        reference = min(recent_high, current_price)
        trailing_stop = reference * (1 - trail_pct)
        candidates["trailing"] = trailing_stop

    # Method 3: SMA-200 as structural support (only if meaningfully below price)
    if sma_200 is not None and 0 < sma_200 < current_price and sma_200 > current_price * 0.85:
        candidates["SMA-200"] = sma_200

    # Hard rule: discard any stop at or above current price
    stops = {k: v for k, v in candidates.items() if 0 < v < current_price}

    if not stops:
        # Fallback: fixed percentage stop (ensures we always return something)
        fallback_stop = current_price * (1 - trail_pct)
        stops["pct_fallback"] = fallback_stop

    # Pick the tightest (highest) valid stop — closest to price = most protective
    method = max(stops, key=stops.get)
    return {
        "stop_loss": round(stops[method], 2),
        "method": f"{method} stop",
        "all_stops": {k: round(v, 2) for k, v in stops.items()},
        "raw_candidates": {k: round(v, 2) for k, v in candidates.items()},
        "volatility": round(vol, 4) if vol is not None else None,
        "vol_percentile": round(vol_pct, 1) if vol_pct is not None else None,
        "dynamic_params": {
            "atr_multiplier": round(atr_mult, 2),
            "trailing_pct": round(trail_pct, 3),
        },
    }


# ---------------------------------------------------------------------------
# Take-profit
# ---------------------------------------------------------------------------

def calculate_take_profit(
    ticker: str, current_price: float | None, stop_loss: float | None
) -> dict:
    """Calculate take-profit target using risk/reward ratio and resistance levels."""
    if current_price is None:
        return {"take_profit": None, "method": "N/A"}

    targets = {}

    # Method 1: Risk/Reward ratio from stop-loss
    if stop_loss is not None and stop_loss < current_price:
        risk = current_price - stop_loss
        reward = risk * config.RISK_REWARD_RATIO
        rr_target = current_price + reward
        targets["R/R ratio"] = rr_target

    # Method 2: Historical resistance (recent highs)
    df = get_price_history(ticker)
    if not df.empty and len(df) > 20:
        # Find resistance as the highest close in the last 6 months
        high_6m = float(df["High"].tail(126).max())
        if high_6m > current_price:
            targets["resistance"] = high_6m

        # Also check 52-week high
        high_52w = float(df["High"].tail(252).max()) if len(df) >= 252 else high_6m
        if high_52w > current_price and high_52w != high_6m:
            targets["52w high"] = high_52w

    if not targets:
        # Fallback: 15% upside target
        targets["default 15%"] = current_price * 1.15

    # Use the nearest realistic target
    method = min(targets, key=targets.get)
    return {
        "take_profit": round(targets[method], 2),
        "method": method,
        "all_targets": {k: round(v, 2) for k, v in targets.items()},
    }
