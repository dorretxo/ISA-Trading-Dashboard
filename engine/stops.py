"""Stop-loss, take-profit, entry price, and position sizing calculators.

Stop-loss: Support confluence model (Kaminski & Lo 2014; Kestner 2003).
Entry: Limit-order placement with thesis-aware pullback depth.
Position sizing: fixed-fraction stop-based sizing with optional Kelly cap.
"""

import numpy as np
import pandas as pd

import config
from utils.data_fetch import get_macro_data, get_price_history


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
    n_windows = n - lookback + 1
    trailing_windows = min(n_windows, 252)
    start_idx = n - trailing_windows + 1
    for end_idx in range(start_idx, n + 1):
        window = log_returns[end_idx - lookback:end_idx]
        vol_history.append(float(np.std(window)) * np.sqrt(252))

    # Percentile rank of current vol within its history
    percentile = float(np.sum(np.array(vol_history) <= current_vol) / len(vol_history) * 100)

    return current_vol, percentile


def _get_vix_percentile() -> float:
    """Get VIX percentile rank over trailing 1 year. Returns 50 if unavailable."""
    try:
        vix = get_macro_data().get("vix")
        if vix is not None and len(vix) >= 30:
            closes = vix["Close"]
            if isinstance(closes, pd.DataFrame):
                closes = closes.iloc[:, 0]
            closes = closes.dropna().values.flatten().astype(float)
            current = closes[-1]
            return float(np.sum(closes <= current) / len(closes) * 100)
    except Exception:
        pass
    return 50.0


def _swing_low(df: pd.DataFrame, window: int = 20) -> float | None:
    """Find the lowest low in the last `window` trading days."""
    if df.empty or len(df) < window:
        return None
    return float(df["Low"].tail(window).min())


# ---------------------------------------------------------------------------
# Support confluence stop-loss (Phase 1A)
# ---------------------------------------------------------------------------

def calculate_stop_loss(
    ticker: str, atr: float | None, current_price: float | None,
    sma_200: float | None = None, sma_50: float | None = None,
    bb_lower: float | None = None,
) -> dict:
    """Calculate stop-loss using support confluence model.

    Academic basis:
    - ATR-based: Kestner (2003) — ATR-scaled stops for equities
    - Regime adaptation: Kaminski & Lo (2014) — wider stops in stressed regimes
    - Structural confluence: combine long-term trend support, swing lows, and volatility

    The model scores each support level by heuristic reliability weight and proximity,
    then picks the tightest valid candidate at or below the weighted average.
    """
    if current_price is None or current_price <= 0:
        return {"stop_loss": None, "method": "N/A", "all_stops": {},
                "support_levels": {}, "regime": {}}

    df = get_price_history(ticker)

    # Compute volatility for dynamic parameters
    vol, vol_pct = _realized_volatility(ticker)
    vix_pct = _get_vix_percentile()

    # Regime-adaptive ATR multiplier (Kaminski & Lo 2014)
    # Base 2.5× per Kestner (2003); scale 1.0× in calm → 1.5× in crisis
    base_mult = 2.5
    regime_factor = 1.0 + 0.5 * (vix_pct / 100)  # 1.0 at VIX_pct=0, 1.5 at VIX_pct=100
    atr_mult = base_mult * regime_factor

    # Trailing % also regime-scaled
    vol_pct_safe = vol_pct if vol_pct is not None else 50.0
    trail_pct = float(np.interp(vol_pct_safe, [20, 80],
                                [config.TRAIL_PCT_LOW_VOL, config.TRAIL_PCT_HIGH_VOL]))

    # Swing low from price data
    swing_low_20d = _swing_low(df) if not df.empty else None

    # --- Support confluence candidates ---
    # Each: (price_level, reliability_weight)
    # Heuristic weights: long-term supports > medium-term > swing/mean-reversion
    candidates = {}

    # Level 1: Regime-adaptive ATR stop (weight 0.35)
    if atr is not None and atr > 0:
        atr_stop = current_price - atr * atr_mult
        if 0 < atr_stop < current_price:
            candidates["atr"] = (atr_stop, 0.35)

    # Level 2: SMA-200 support — strongest structural support
    if sma_200 is not None and 0.85 * current_price < sma_200 < current_price:
        # 3% cushion below SMA-200 to avoid exact-level whipsaw
        candidates["sma_200"] = (sma_200 * 0.97, 0.25)

    # Level 3: SMA-50 support — medium-term trend support
    if sma_50 is not None and 0.90 * current_price < sma_50 < current_price:
        # 2% cushion below SMA-50
        candidates["sma_50"] = (sma_50 * 0.98, 0.20)

    # Level 4: 20-day swing low — local structure
    if swing_low_20d is not None and 0.85 * current_price < swing_low_20d < current_price:
        # 1% cushion below swing low
        candidates["swing_low"] = (swing_low_20d * 0.99, 0.15)

    # Level 5: Bollinger Band lower — mean-reversion floor (supplementary)
    if bb_lower is not None and 0 < bb_lower < current_price:
        candidates["bb_lower"] = (bb_lower, 0.05)

    # --- Confluence resolution ---
    if candidates:
        total_weight = sum(w for _, w in candidates.values())
        weighted_avg = sum(p * w for p, w in candidates.values()) / total_weight

        # Pick the tightest (highest) candidate at or below weighted average + 2% tolerance
        valid = {k: (p, w) for k, (p, w) in candidates.items()
                 if p <= weighted_avg * 1.02}
        if valid:
            best_key = max(valid, key=lambda k: valid[k][0])
            stop_price = valid[best_key][0]
            method = f"{best_key} confluence"
        else:
            stop_price = weighted_avg
            method = "weighted confluence"
    else:
        # Fallback: trailing percentage stop
        if not df.empty:
            recent_high = float(df["High"].tail(63).max())
            reference = min(recent_high, current_price)
            stop_price = reference * (1 - trail_pct)
        else:
            stop_price = current_price * (1 - trail_pct)
        method = "pct_fallback"

    # Hard rule: stop must be below current price
    if stop_price >= current_price:
        stop_price = current_price * (1 - trail_pct)
        method = "pct_fallback"

    stop_price = round(stop_price, 2)
    stop_distance_pct = round((current_price - stop_price) / current_price * 100, 2)

    # Build support levels detail for UI
    support_levels = {}
    for k, (p, w) in candidates.items():
        support_levels[k] = {
            "price": round(p, 2),
            "weight": w,
            "distance_pct": round((current_price - p) / current_price * 100, 2),
        }

    return {
        "stop_loss": stop_price,
        "method": method,
        "stop_distance_pct": stop_distance_pct,
        "all_stops": {k: round(p, 2) for k, (p, _) in candidates.items()},
        "support_levels": support_levels,
        "regime": {
            "vix_percentile": round(vix_pct, 1),
            "atr_multiplier": round(atr_mult, 2),
            "regime_factor": round(regime_factor, 2),
            "trailing_pct": round(trail_pct, 3),
        },
        "volatility": round(vol, 4) if vol is not None else None,
        "vol_percentile": round(vol_pct, 1) if vol_pct is not None else None,
    }


# ---------------------------------------------------------------------------
# Take-profit
# ---------------------------------------------------------------------------

def calculate_take_profit(
    ticker: str,
    current_price: float | None,
    stop_loss: float | None,
    entry_price: float | None = None,
    entry_lens: str | None = None,
) -> dict:
    """Calculate take-profit target using entry-consistent R/R and resistance.

    For discovery candidates, `entry_price` should be the planned limit price.
    For existing holdings, it can be left as None to anchor targets to spot.
    """
    reference_price = entry_price if entry_price is not None and entry_price > 0 else current_price
    if reference_price is None:
        return {"take_profit": None, "method": "N/A"}

    targets = {}

    # Method 1: Risk/Reward ratio from stop-loss
    if entry_lens == "momentum":
        rr_multiple = 2.5
    elif entry_lens in ("quality", "value"):
        rr_multiple = 2.0
    else:
        rr_multiple = config.RISK_REWARD_RATIO
    if stop_loss is not None and stop_loss < reference_price:
        risk = reference_price - stop_loss
        reward = risk * rr_multiple
        rr_target = reference_price + reward
        targets["R/R ratio"] = rr_target

    # Method 2: Historical resistance (recent highs)
    df = get_price_history(ticker)
    resistance = None
    if not df.empty and len(df) > 20:
        # Find resistance as the highest close in the last 6 months
        high_6m = float(df["High"].tail(126).max())
        if high_6m > reference_price:
            targets["resistance"] = high_6m
            resistance = high_6m

        # Also check 52-week high
        high_52w = float(df["High"].tail(252).max()) if len(df) >= 252 else high_6m
        if high_52w > reference_price and high_52w != high_6m:
            targets["52w high"] = high_52w
            if resistance is None:
                resistance = high_52w

    if not targets:
        # Fallback: 15% upside target
        targets["default 15%"] = reference_price * 1.15

    if "R/R ratio" in targets and resistance is not None and entry_lens in ("momentum", "quality", "value"):
        if entry_lens == "value":
            final_target = min(targets["R/R ratio"], resistance)
            method = "value rr/resistance"
        else:
            final_target = max(targets["R/R ratio"], resistance)
            method = f"{entry_lens} rr/resistance"
    else:
        # Fallback for portfolio holdings (no entry_lens): pick the nearest
        # (lowest) target — conservative. Discovery candidates with entry_lens
        # go through the thesis-aware branches above.
        method = min(targets, key=targets.get)
        final_target = targets[method]

    return {
        "take_profit": round(final_target, 2),
        "method": method,
        "all_targets": {k: round(v, 2) for k, v in targets.items()},
    }


# ---------------------------------------------------------------------------
# Entry price calculator (Phase 2A)
# ---------------------------------------------------------------------------

def calculate_entry_strategy(
    current_price: float,
    atr: float | None,
    sma_50: float | None = None,
    bb_lower: float | None = None,
    vol_percentile: float | None = None,
    entry_lens: str = "momentum",
) -> dict:
    """Calculate thesis-aware limit buy price and entry zone for candidates."""
    if current_price is None or current_price <= 0:
        return {"entry_price": None, "entry_method": "N/A", "entry_zone": (None, None),
                "fill_probability": None, "all_levels": {}, "discount_pct": 0}

    vol_pct = vol_percentile if vol_percentile is not None else 50.0
    entry_levels = {}

    # Common limit anchor: 0.5% in calm markets to 1.5% in volatile markets.
    offset_pct = 0.005 + 0.01 * (vol_pct / 100)
    entry_levels["limit_offset"] = current_price * (1 - offset_pct)

    # Lens-aware pullbacks: momentum entries stay tighter; value waits deeper.
    if sma_50 is not None and 0 < sma_50 < current_price:
        pullback_pct = (current_price - sma_50) / current_price
        if pullback_pct < 0.10:
            entry_levels["sma_50_support"] = sma_50

    if atr is not None and atr > 0:
        if entry_lens == "momentum":
            atr_entry = current_price - 0.5 * atr
            label = "atr_half_dip"
        elif entry_lens == "quality":
            atr_entry = current_price - 0.75 * atr
            label = "atr_quality_dip"
        else:
            atr_entry = current_price - 1.0 * atr
            label = "atr_value_dip"
        if 0 < atr_entry < current_price:
            entry_levels[label] = atr_entry

    if bb_lower is not None and bb_lower < current_price:
        bb_dist = (current_price - bb_lower) / current_price
        if bb_dist < 0.10:
            entry_levels["bb_lower"] = bb_lower

    if entry_lens == "value":
        deep_offset = current_price * (1 - min(offset_pct + 0.005, 0.03))
        entry_levels["value_limit_offset"] = deep_offset

    if entry_levels:
        if entry_lens == "value":
            ordered = sorted(
                (price, method) for method, price in entry_levels.items()
                if 0 < price < current_price
            )
            if ordered:
                mid_idx = len(ordered) // 2
                entry_price, best_method = ordered[mid_idx]
            else:
                entry_price = current_price * 0.99
                best_method = "value_fallback"
        else:
            best_method = max(entry_levels, key=entry_levels.get)
            entry_price = entry_levels[best_method]
    else:
        entry_price = current_price * 0.995  # 0.5% default
        best_method = "market_discount"

    entry_zone = (round(entry_price, 2), round(current_price, 2))

    # Expected fill probability (Angel et al. 2015 calibration)
    discount_pct = (current_price - entry_price) / current_price * 100
    if discount_pct < 0.5:
        fill_probability = 0.90
    elif discount_pct < 1.0:
        fill_probability = 0.80
    elif discount_pct < 2.0:
        fill_probability = 0.65
    else:
        fill_probability = 0.50

    return {
        "entry_price": round(entry_price, 2),
        "entry_method": best_method,
        "entry_zone": entry_zone,
        "fill_probability": round(fill_probability, 2),
        "all_levels": {k: round(v, 2) for k, v in entry_levels.items()},
        "discount_pct": round(discount_pct, 2),
        "entry_lens": entry_lens,
    }


# ---------------------------------------------------------------------------
# Position sizing — fixed-fraction stop sizing with optional Kelly cap
# ---------------------------------------------------------------------------

def calculate_position_size(
    portfolio_value: float, entry_price: float, stop_loss: float,
    take_profit: float | None = None,
    risk_per_trade_pct: float = 0.01,
    kelly_cap_fraction: float | None = None,
) -> dict:
    """Size positions from stop risk, optionally capped by empirical Kelly.

    Base rule: risk a fixed fraction of portfolio capital if the stop is hit.
    Optional cap: if the backtest has enough evidence for the action tier,
    also cap notional exposure by the empirical half-Kelly fraction.
    """
    if (entry_price <= 0 or stop_loss <= 0 or stop_loss >= entry_price
            or portfolio_value <= 0):
        return {"shares": 0, "position_value": 0, "position_weight": 0,
                "risk_amount": 0, "risk_per_share": 0, "stop_distance_pct": 0,
                "r_r_ratio": None, "sizing_method": "invalid", "kelly_cap_fraction": None}

    risk_per_share = entry_price - stop_loss
    stop_distance_pct = risk_per_share / entry_price

    risk_budget = portfolio_value * risk_per_trade_pct
    shares = int(risk_budget / risk_per_share)
    sizing_method = "fixed_fraction_stop"

    # Risk/reward ratio
    r_r_ratio = None
    if take_profit is not None and take_profit > entry_price:
        reward_per_share = take_profit - entry_price
        r_r_ratio = round(reward_per_share / risk_per_share, 2)

    if kelly_cap_fraction is not None and kelly_cap_fraction > 0:
        kelly_shares = int((portfolio_value * kelly_cap_fraction) / entry_price)
        if kelly_shares > 0:
            shares = min(shares, kelly_shares)
            sizing_method = "fixed_fraction_stop + half_kelly_cap"

    # Cap at max position weight from config
    max_weight = getattr(config, "MAX_POSITION_WEIGHT", 0.25)
    max_weight_shares = int((portfolio_value * max_weight) / entry_price)
    if max_weight_shares > 0:
        shares = min(shares, max_weight_shares)

    position_value = shares * entry_price
    position_weight = position_value / portfolio_value if portfolio_value > 0 else 0
    risk_amount = shares * risk_per_share

    return {
        "shares": shares,
        "position_value": round(position_value, 2),
        "position_weight": round(position_weight, 4),
        "risk_amount": round(risk_amount, 2),
        "risk_per_share": round(risk_per_share, 2),
        "stop_distance_pct": round(stop_distance_pct * 100, 2),
        "r_r_ratio": r_r_ratio,
        "sizing_method": sizing_method,
        "kelly_cap_fraction": round(kelly_cap_fraction, 4) if kelly_cap_fraction else None,
    }
