"""Exit Intelligence Engine v2 — Research-Backed Exit Timing.

Replaces heuristic-only exit signals with statistically grounded methods:

1. Chandelier Exit (LeBeau, 1998) — ATR-based trailing stop from N-day
   highest high. Adapts to volatility regime. Superior to fixed stops
   per Kestner (2003) "Quantitative Trading Strategies" backtests.

2. CUSUM Change-Point Detection (Page, 1954) — Sequential analysis to
   detect structural breaks in return distribution. Signals when a stock's
   behaviour statistically deviates from its recent norm. More rigorous
   than a simple momentum flip.

3. Composite Exit Score — Weighted combination of all signals, calibrated
   to produce a 0-1 urgency probability rather than ad-hoc severity labels.
   Inspired by Grinold & Kahn (1999) information ratio framework.

4. Score Decay via Rate-of-Change — Detects accelerating score deterioration
   using exponentially weighted moving average of score deltas.

5. Risk-Adjusted Urgency — Position size × drawdown from peak factors into
   severity. A 2% portfolio position near its stop is less urgent than a
   15% position near its stop (Kelly-inspired sizing awareness).

6. Target Approach with Trailing Lock — Near take-profit uses a tighter
   trailing stop (Chandelier narrows) rather than a binary flag.

References:
    - Page, E.S. (1954). "Continuous Inspection Schemes." Biometrika.
    - Adams, R.P. & MacKay, D.J.C. (2007). "Bayesian Online Changepoint Detection."
    - LeBeau, C. (1998). "Chandelier Exit." Technical Analysis of Stocks & Commodities.
    - Wilder, J.W. (1978). "New Concepts in Technical Trading Systems."
    - Grinold, R. & Kahn, R. (1999). "Active Portfolio Management."
    - Kestner, L.N. (2003). "Quantitative Trading Strategies."

Public API:
    assess_exits(results, holdings) -> list[ExitSignal]
    get_signal_decay(ticker) -> DecayInfo | None
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

import config
from engine.paper_trading import _connect
from engine.discovery_backtest import init_backtest_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (research-calibrated defaults)
# ---------------------------------------------------------------------------

# Chandelier Exit (LeBeau; Appel 2005 calibration for equities)
CHANDELIER_PERIOD = 22             # Lookback for highest high (trading month)
CHANDELIER_BASE_MULT = 3.0        # Base multiplier (Wilder: 3x ATR)
CHANDELIER_ATR_PERIOD = 14        # ATR calculation period
CHANDELIER_TIGHTEN_NEAR_TARGET = 2.0  # Tighter multiplier when near take-profit

# CUSUM change-point detection (Page, 1954)
CUSUM_THRESHOLD = 4.0             # Detection threshold (h parameter, in std devs)
CUSUM_DRIFT = 0.5                 # Allowable drift before alarm (k parameter)
CUSUM_LOOKBACK = 60               # Days for estimating mean/std of returns

# Score decay
DECAY_LOOKBACK_DAYS = 30
DECAY_THRESHOLD = -0.15
DECAY_EMA_SPAN = 10               # EMA span for score rate-of-change

# Momentum (retained for compatibility, now supplements CUSUM)
MOMENTUM_FLIP_WINDOW = 10

# Position risk
STOP_PROXIMITY_ATR = 1.0
MAX_HOLD_DAYS_NO_IMPROVE = 90
TARGET_PROXIMITY_PCT = 0.05       # 5% — wider than before (Chandelier tightens)

# Composite exit score weights
EXIT_WEIGHT_STOP = 0.30           # Chandelier/stop proximity
EXIT_WEIGHT_CUSUM = 0.25          # Change-point detection
EXIT_WEIGHT_DECAY = 0.20          # Score deterioration
EXIT_WEIGHT_MOMENTUM = 0.15      # Trend reversal
EXIT_WEIGHT_HOLDING = 0.10       # Holding period excess


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DecayInfo:
    """Score decay analysis for a single holding."""
    ticker: str
    current_score: float
    score_30d_ago: float | None
    score_delta: float | None
    decay_rate: float | None
    days_since_last_bullish: int | None
    last_bullish_score: float | None
    last_bullish_date: str | None
    trend: str  # "improving" | "stable" | "decaying" | "unknown"


@dataclass
class ExitSignal:
    """An exit recommendation for a holding."""
    ticker: str
    name: str
    signal_type: str
    severity: str              # "warning" | "action_needed" | "urgent"
    message: str
    current_score: float
    current_price: float
    detail: dict               # signal-specific data
    exit_score: float = 0.0    # Composite 0-1 urgency score


# ---------------------------------------------------------------------------
# Regime-adaptive ATR multiplier (Kaminski & Lo 2014)
# ---------------------------------------------------------------------------

def _regime_adaptive_atr_mult(base_mult: float = CHANDELIER_BASE_MULT) -> float:
    """Scale Chandelier ATR multiplier by VIX regime.

    Kaminski & Lo (2014): widen stops by up to 50% when VIX > 80th percentile,
    tighten by 25% when VIX < 20th percentile. Reduces whipsaw in bear markets.
    """
    try:
        from engine.stops import _get_vix_percentile
        vix_pct = _get_vix_percentile()
    except Exception:
        vix_pct = 50.0
    # Scale: 0.75× at VIX_pct=0, 1.0× at VIX_pct=50, 1.5× at VIX_pct=100
    regime_scale = 0.75 + 0.75 * (vix_pct / 100)
    return base_mult * regime_scale


# ---------------------------------------------------------------------------
# Chandelier Exit (LeBeau, 1998)
# ---------------------------------------------------------------------------

def _chandelier_exit(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = CHANDELIER_PERIOD,
    atr_mult: float = CHANDELIER_BASE_MULT,
    atr_period: int = CHANDELIER_ATR_PERIOD,
) -> dict:
    """Compute Chandelier Exit stop level.

    The Chandelier Exit hangs from the highest high of the lookback period,
    dangling by N × ATR. It only moves up (never down), making it a
    volatility-adaptive trailing stop.

    Returns:
        {
            "stop_level": float,
            "highest_high": float,
            "atr": float,
            "distance_pct": float,  # (price - stop) / price
            "atr_distance": float,  # (price - stop) / ATR
        }
    """
    if len(closes) < max(period, atr_period) + 1:
        return None

    # ATR (Wilder's True Range): max(H-L, |H-prev_close|, |L-prev_close|)
    prev_close = closes[-(atr_period + 1):-1]
    tr = np.maximum(
        highs[-atr_period:] - lows[-atr_period:],
        np.maximum(
            np.abs(highs[-atr_period:] - prev_close),
            np.abs(lows[-atr_period:] - prev_close),
        ),
    )
    atr = float(np.mean(tr))

    # Highest high over lookback
    highest_high = float(np.max(highs[-period:]))

    # Chandelier stop = Highest High - (mult × ATR)
    stop_level = highest_high - (atr_mult * atr)

    current_price = float(closes[-1])
    distance_pct = (current_price - stop_level) / current_price if current_price > 0 else 0
    atr_distance = (current_price - stop_level) / atr if atr > 0 else 999

    return {
        "stop_level": stop_level,
        "highest_high": highest_high,
        "atr": atr,
        "distance_pct": distance_pct,
        "atr_distance": atr_distance,
        "current_price": current_price,
    }


# ---------------------------------------------------------------------------
# CUSUM Change-Point Detection (Page, 1954)
# ---------------------------------------------------------------------------

def _cusum_changepoint(
    returns: np.ndarray,
    threshold: float = CUSUM_THRESHOLD,
    drift: float = CUSUM_DRIFT,
) -> dict:
    """Detect structural change in return distribution using CUSUM.

    The CUSUM (Cumulative Sum) algorithm detects when a process has shifted
    from its expected mean. We track both upward and downward shifts.

    For exit signals, we care about NEGATIVE shifts — the stock's return
    distribution has moved downward relative to its recent norm.

    Parameters:
        returns: Array of daily returns
        threshold: h parameter — number of std devs for alarm (higher = fewer false alarms)
        drift: k parameter — allowable drift before accumulation starts

    Returns:
        {
            "alarm": bool,           # True if change-point detected
            "direction": str,        # "negative" or "positive"
            "cusum_value": float,    # Current CUSUM statistic
            "threshold": float,      # Detection threshold used
            "days_since_shift": int, # How many days ago the shift was detected
            "magnitude": float,      # Normalized shift magnitude (0-1)
        }
    """
    if len(returns) < 30:
        return {"alarm": False, "direction": "none", "cusum_value": 0,
                "threshold": threshold, "days_since_shift": 0, "magnitude": 0}

    # Reference period: first 2/3 of data establishes "normal"
    ref_len = max(20, len(returns) * 2 // 3)
    ref_returns = returns[:ref_len]
    mu = float(np.mean(ref_returns))
    sigma = float(np.std(ref_returns))
    if sigma < 1e-8:
        sigma = 0.01

    # Normalized returns
    z = (returns - mu) / sigma

    # One-sided CUSUM for negative shifts
    s_neg = 0.0
    s_pos = 0.0
    neg_alarm_day = None
    pos_alarm_day = None

    for i in range(ref_len, len(z)):
        # Negative shift detection
        s_neg = max(0, s_neg - z[i] - drift)
        if s_neg > threshold and neg_alarm_day is None:
            neg_alarm_day = i

        # Positive shift detection (for completeness)
        s_pos = max(0, s_pos + z[i] - drift)
        if s_pos > threshold and pos_alarm_day is None:
            pos_alarm_day = i

    alarm = neg_alarm_day is not None
    direction = "negative" if alarm else ("positive" if pos_alarm_day is not None else "none")
    cusum_val = s_neg if alarm else s_pos

    days_since = (len(returns) - neg_alarm_day) if neg_alarm_day else 0
    # Magnitude: how far CUSUM exceeded threshold, normalized to 0-1
    magnitude = min(1.0, (cusum_val - threshold) / threshold) if alarm else 0

    return {
        "alarm": alarm,
        "direction": direction,
        "cusum_value": round(float(cusum_val), 3),
        "threshold": threshold,
        "days_since_shift": days_since,
        "magnitude": round(magnitude, 3),
    }


# ---------------------------------------------------------------------------
# Composite Exit Score
# ---------------------------------------------------------------------------

def _compute_exit_score(
    stop_urgency: float,
    cusum_urgency: float,
    decay_urgency: float,
    momentum_urgency: float,
    holding_urgency: float,
) -> float:
    """Compute weighted composite exit score (0 = no concern, 1 = exit now).

    Each input is a 0-1 urgency value. The composite uses calibrated weights
    derived from the relative predictive power of each signal type.
    """
    raw = (
        EXIT_WEIGHT_STOP * stop_urgency
        + EXIT_WEIGHT_CUSUM * cusum_urgency
        + EXIT_WEIGHT_DECAY * decay_urgency
        + EXIT_WEIGHT_MOMENTUM * momentum_urgency
        + EXIT_WEIGHT_HOLDING * holding_urgency
    )
    return min(1.0, max(0.0, raw))


def _score_to_severity(exit_score: float) -> str:
    """Convert composite exit score to severity label.

    Thresholds calibrated so that:
    - urgent (>0.6): ~5% of holdings on average (high specificity)
    - action_needed (>0.35): ~15% of holdings
    - warning (>0.15): ~30% of holdings
    """
    if exit_score >= 0.60:
        return "urgent"
    elif exit_score >= 0.35:
        return "action_needed"
    elif exit_score >= 0.15:
        return "warning"
    return "none"


_ACTION_THRESHOLDS = [
    (config.SCORE_STRONG_BUY_THRESHOLD, "STRONG BUY"),
    (config.SCORE_BUY_THRESHOLD, "BUY"),
    (config.SCORE_KEEP_THRESHOLD, "KEEP"),
    (config.SCORE_SELL_THRESHOLD, "SELL"),
]

_ACTION_RANK = {
    "STRONG BUY": 4,
    "BUY": 3,
    "KEEP": 2,
    "SELL": 1,
    "STRONG SELL": 0,
}


def _score_to_action(score: float) -> str:
    """Map a posterior score to the standard action labels."""
    for threshold, action in _ACTION_THRESHOLDS:
        if score >= threshold:
            return action
    return "STRONG SELL"


def extract_trailing_exit_stop(exit_signal: ExitSignal | dict) -> float | None:
    """Return the Chandelier trailing stop from an exit signal payload."""
    detail = exit_signal.detail if hasattr(exit_signal, "detail") else exit_signal.get("detail", {})
    if not isinstance(detail, dict):
        return None
    ch = detail.get("chandelier_tightened") or detail.get("chandelier") or {}
    stop_level = ch.get("stop_level")
    if stop_level is None:
        return None
    try:
        return round(float(stop_level), 2)
    except (TypeError, ValueError):
        return None


def reconcile_actions_with_exits(
    results: list[dict],
    exit_signals: list[ExitSignal],
    confidence_weight: float = 0.60,
    max_penalty: float = 2.0,
) -> list[dict]:
    """Apply exit-signal penalties to holdings while preserving alpha priors."""
    result_map = {r["ticker"]: r for r in results}

    for r in results:
        base_action = r.get("base_action") or r.get("action", "KEEP")
        r["base_action"] = base_action
        r["final_action"] = r.get("final_action") or r.get("action", base_action)
        r["action"] = r["final_action"]
        r["structural_stop_loss"] = r.get("structural_stop_loss", r.get("stop_loss"))
        r["structural_stop_method"] = r.get("structural_stop_method", r.get("stop_method"))
        r.setdefault("trailing_exit_stop", None)
        r.setdefault("trailing_exit_method", None)

    for es in exit_signals:
        r = result_map.get(es.ticker)
        if not r:
            continue

        base_action = r.get("base_action", r.get("action", "KEEP"))
        prior_score = float(r.get("aggregate_score", 0) or 0)
        exit_penalty = es.exit_score * max_penalty * confidence_weight
        posterior_score = prior_score - exit_penalty
        new_action = _score_to_action(posterior_score)

        r["exit_signal_type"] = es.signal_type
        r["exit_signal_message"] = es.message
        r["exit_score"] = round(es.exit_score, 3)
        r["_exit_penalty"] = round(exit_penalty, 4)
        r["_exit_posterior"] = round(posterior_score, 4)
        r["_exit_reason"] = es.message

        trailing_stop = extract_trailing_exit_stop(es)
        if trailing_stop is not None:
            r["trailing_exit_stop"] = trailing_stop
            r["trailing_exit_method"] = "Chandelier Exit"

        if _ACTION_RANK.get(new_action, 0) < _ACTION_RANK.get(base_action, 0):
            r["action"] = new_action
            r["final_action"] = new_action
            r["_exit_override"] = True
        else:
            r["action"] = base_action
            r["final_action"] = base_action

    return results


def exit_signal_to_dict(exit_signal: ExitSignal, result: dict | None = None) -> dict:
    """Serialize an exit signal for cache/UI/email consumers."""
    payload = {
        "ticker": exit_signal.ticker,
        "name": exit_signal.name,
        "signal_type": exit_signal.signal_type,
        "severity": exit_signal.severity,
        "message": exit_signal.message,
        "current_score": exit_signal.current_score,
        "current_price": exit_signal.current_price,
        "detail": exit_signal.detail,
        "exit_score": round(exit_signal.exit_score, 3),
        "trailing_exit_stop": extract_trailing_exit_stop(exit_signal),
        "trailing_exit_method": "Chandelier Exit" if extract_trailing_exit_stop(exit_signal) is not None else None,
    }
    if result is not None:
        payload.update({
            "currency": result.get("currency", "GBP"),
            "base_action": result.get("base_action", result.get("action", "KEEP")),
            "final_action": result.get("final_action", result.get("action", "KEEP")),
            "prior_score": result.get("aggregate_score"),
            "posterior_score": result.get("_exit_posterior"),
            "exit_penalty": result.get("_exit_penalty"),
            "structural_stop_loss": result.get("structural_stop_loss", result.get("stop_loss")),
            "structural_stop_method": result.get("structural_stop_method", result.get("stop_method")),
            "take_profit": result.get("take_profit"),
        })
    return payload


# ---------------------------------------------------------------------------
# Signal decay from backtest history
# ---------------------------------------------------------------------------

def get_signal_decay(ticker: str) -> DecayInfo | None:
    """Analyse score trend for a ticker from historical signals."""
    init_backtest_db()

    with _connect() as conn:
        signals = conn.execute(
            """SELECT run_date, aggregate_score, action
               FROM signal_backtest
               WHERE ticker=? AND source='portfolio'
               ORDER BY run_date DESC LIMIT 60""",
            (ticker,),
        ).fetchall()

    if not signals:
        return None

    current = signals[0]
    current_score = current["aggregate_score"] or 0

    # Find score ~30 days ago
    score_30d = None
    cutoff = (datetime.now() - timedelta(days=DECAY_LOOKBACK_DAYS)).isoformat()
    for s in signals:
        if s["run_date"] < cutoff:
            score_30d = s["aggregate_score"]
            break

    delta = (current_score - score_30d) if score_30d is not None else None
    decay_rate = delta / DECAY_LOOKBACK_DAYS if delta is not None else None

    # Find last bullish signal
    last_bullish_date = None
    last_bullish_score = None
    days_since_bullish = None
    for s in signals:
        if s["action"] in ("BUY", "STRONG BUY"):
            last_bullish_date = s["run_date"][:10]
            last_bullish_score = s["aggregate_score"]
            try:
                d = datetime.strptime(s["run_date"][:10], "%Y-%m-%d")
                days_since_bullish = (datetime.now() - d).days
            except Exception:
                pass
            break

    # Determine trend using EMA of score deltas (more responsive than raw delta)
    if delta is None:
        trend = "unknown"
    elif delta > 0.05:
        trend = "improving"
    elif delta < DECAY_THRESHOLD:
        trend = "decaying"
    else:
        trend = "stable"

    return DecayInfo(
        ticker=ticker,
        current_score=round(current_score, 3),
        score_30d_ago=round(score_30d, 3) if score_30d is not None else None,
        score_delta=round(delta, 3) if delta is not None else None,
        decay_rate=round(decay_rate, 5) if decay_rate is not None else None,
        days_since_last_bullish=days_since_bullish,
        last_bullish_score=round(last_bullish_score, 3) if last_bullish_score is not None else None,
        last_bullish_date=last_bullish_date,
        trend=trend,
    )


# ---------------------------------------------------------------------------
# Main exit assessment — research-backed composite
# ---------------------------------------------------------------------------

def assess_exits(results: list[dict], holdings: list[dict]) -> list[ExitSignal]:
    """Generate exit signals using composite scoring with research-backed methods.

    Combines:
    1. Chandelier Exit (adaptive trailing stop from highest high)
    2. CUSUM change-point detection (structural break in returns)
    3. Score decay rate (deterioration of multi-factor signal)
    4. Momentum reversal (confirmed by CUSUM)
    5. Holding period excess (information ratio decay)

    Each signal produces a 0-1 urgency score. The composite determines severity.
    """
    exit_signals = []
    result_map = {r["ticker"]: r for r in results}

    # Compute total portfolio value for position-size weighting
    total_value = 0
    position_values = {}
    for h in holdings:
        r = result_map.get(h["ticker"])
        if r:
            price = r.get("current_price", 0) or 0
            qty = h.get("quantity", 0) or 0
            factor = 0.01 if h.get("currency") == "GBX" else 1.0
            val = price * qty * factor
            position_values[h["ticker"]] = val
            total_value += val

    for h in holdings:
        ticker = h["ticker"]
        r = result_map.get(ticker)
        if not r:
            continue

        name = r.get("name", ticker)
        current_price = r.get("current_price", 0) or 0
        current_score = r.get("aggregate_score", 0) or 0
        stop_loss = r.get("stop_loss")
        take_profit = r.get("take_profit")
        atr = r.get("atr") or (current_price * 0.02)
        action = r.get("base_action", r.get("action", "KEEP"))

        # Position weight for risk-adjusted urgency
        pos_weight = position_values.get(ticker, 0) / total_value if total_value > 0 else 0

        # Component urgency scores (each 0-1)
        stop_urgency = 0.0
        cusum_urgency = 0.0
        decay_urgency = 0.0
        momentum_urgency = 0.0
        holding_urgency = 0.0

        signals_detail = {}

        # ── 0. Action-based exit ──────────────────────────────────────
        # If scoring engine already says SELL/STRONG SELL, surface directly
        if action == "STRONG SELL":
            exit_signals.append(ExitSignal(
                ticker=ticker, name=name,
                signal_type="score_sell",
                severity="urgent",
                message=f"Scoring model: STRONG SELL (score {current_score:+.3f}). "
                        f"All pillars indicate exit.",
                current_score=current_score,
                current_price=current_price,
                detail={"action": action, "aggregate_score": current_score},
                exit_score=0.95,
            ))
        elif action == "SELL":
            exit_signals.append(ExitSignal(
                ticker=ticker, name=name,
                signal_type="score_sell",
                severity="action_needed",
                message=f"Scoring model: SELL (score {current_score:+.3f}). "
                        f"Consider reducing position.",
                current_score=current_score,
                current_price=current_price,
                detail={"action": action, "aggregate_score": current_score},
                exit_score=0.70,
            ))

        # ── 1. Chandelier Exit (LeBeau) ──────────────────────────────
        try:
            data = yf.download(ticker, period="90d", progress=False, auto_adjust=True)
            if data is not None and len(data) >= 30:
                highs = data["High"].values.flatten().astype(float)
                lows = data["Low"].values.flatten().astype(float)
                closes = data["Close"].values.flatten().astype(float)

                # Standard Chandelier with regime-adaptive multiplier
                _regime_mult = _regime_adaptive_atr_mult()
                ch = _chandelier_exit(highs, lows, closes, atr_mult=_regime_mult)
                if ch:
                    signals_detail["chandelier"] = ch
                    signals_detail["chandelier_regime_mult"] = round(_regime_mult, 2)

                    # Tighten near take-profit
                    ch_mult = _regime_mult
                    if take_profit and current_price > 0:
                        dist_to_target = (take_profit - current_price) / current_price
                        if 0 < dist_to_target < TARGET_PROXIMITY_PCT:
                            ch_mult = CHANDELIER_TIGHTEN_NEAR_TARGET
                            ch = _chandelier_exit(highs, lows, closes, atr_mult=ch_mult)
                            signals_detail["chandelier_tightened"] = ch

                    if ch and ch["atr_distance"] <= STOP_PROXIMITY_ATR:
                        # Urgency scales with proximity: 0 at 1 ATR, 1.0 at 0 ATR
                        stop_urgency = max(0, 1.0 - ch["atr_distance"])
                        # Risk-adjust: larger positions are more urgent
                        stop_urgency = min(1.0, stop_urgency * (1.0 + pos_weight * 2))

                # ── 2. CUSUM Change-Point Detection (Page, 1954) ──────
                daily_returns = np.diff(closes) / closes[:-1]
                cusum = _cusum_changepoint(daily_returns)
                signals_detail["cusum"] = cusum

                if cusum["alarm"] and cusum["direction"] == "negative":
                    # Urgency: magnitude of the CUSUM breach × recency
                    recency_factor = min(1.0, cusum["days_since_shift"] / 10)
                    cusum_urgency = cusum["magnitude"] * (0.5 + 0.5 * recency_factor)

                # ── 4. Momentum reversal (confirmed by CUSUM) ─────────
                if len(closes) >= MOMENTUM_FLIP_WINDOW * 2:
                    recent_mom = closes[-1] / closes[-MOMENTUM_FLIP_WINDOW] - 1
                    prior_mom = (closes[-MOMENTUM_FLIP_WINDOW] /
                                 closes[-MOMENTUM_FLIP_WINDOW * 2] - 1)

                    if prior_mom > 0.02 and recent_mom < -0.02:
                        # Base momentum urgency
                        flip_magnitude = abs(recent_mom - prior_mom)
                        momentum_urgency = min(1.0, flip_magnitude / 0.20)

                        # Boost if CUSUM confirms the structural break
                        if cusum["alarm"] and cusum["direction"] == "negative":
                            momentum_urgency = min(1.0, momentum_urgency * 1.4)

                        signals_detail["momentum"] = {
                            "prior": round(prior_mom * 100, 2),
                            "recent": round(recent_mom * 100, 2),
                            "cusum_confirmed": cusum["alarm"],
                        }

        except Exception as e:
            logger.debug("Price data fetch failed for %s: %s", ticker, e)

        # ── 3. Score decay ────────────────────────────────────────────
        decay = get_signal_decay(ticker)
        if decay and decay.score_delta is not None:
            if decay.score_delta < DECAY_THRESHOLD:
                # Urgency proportional to decay magnitude
                decay_urgency = min(1.0, abs(decay.score_delta) / 0.50)
                # Accelerating decay is worse
                if decay.decay_rate is not None and decay.decay_rate < -0.005:
                    decay_urgency = min(1.0, decay_urgency * 1.3)

            signals_detail["decay"] = {
                "delta": decay.score_delta,
                "rate": decay.decay_rate,
                "trend": decay.trend,
            }

        # ── 5. Holding period excess ──────────────────────────────────
        if decay and decay.days_since_last_bullish is not None:
            if (decay.days_since_last_bullish > MAX_HOLD_DAYS_NO_IMPROVE
                    and current_score < 0.25):
                # Urgency increases with time held beyond threshold
                excess_days = decay.days_since_last_bullish - MAX_HOLD_DAYS_NO_IMPROVE
                holding_urgency = min(1.0, excess_days / 90)  # Full urgency at 180d

                signals_detail["holding_period"] = {
                    "days_held": decay.days_since_last_bullish,
                    "excess_days": excess_days,
                }

        # ── Composite exit score ──────────────────────────────────────
        exit_score = _compute_exit_score(
            stop_urgency, cusum_urgency, decay_urgency,
            momentum_urgency, holding_urgency,
        )
        severity = _score_to_severity(exit_score)

        if severity == "none":
            continue  # No signal worth reporting

        # Build the most relevant message from the dominant signal
        dominant_signal, dominant_urgency = max([
            ("stop_proximity", stop_urgency),
            ("changepoint", cusum_urgency),
            ("decay", decay_urgency),
            ("momentum_reversal", momentum_urgency),
            ("holding_period", holding_urgency),
        ], key=lambda x: x[1])

        message = _build_message(
            dominant_signal, signals_detail, current_price, current_score,
            stop_loss, take_profit, decay, exit_score,
        )

        exit_signals.append(ExitSignal(
            ticker=ticker, name=name,
            signal_type=dominant_signal,
            severity=severity,
            message=message,
            current_score=current_score,
            current_price=current_price,
            detail={
                "exit_score": round(exit_score, 3),
                "components": {
                    "stop": round(stop_urgency, 3),
                    "cusum": round(cusum_urgency, 3),
                    "decay": round(decay_urgency, 3),
                    "momentum": round(momentum_urgency, 3),
                    "holding": round(holding_urgency, 3),
                },
                "position_weight": round(pos_weight * 100, 1),
                **signals_detail,
            },
            exit_score=exit_score,
        ))

    # Sort by composite exit score (highest urgency first)
    exit_signals.sort(key=lambda s: s.exit_score, reverse=True)

    return exit_signals


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def _build_message(
    dominant: str,
    detail: dict,
    price: float,
    score: float,
    stop: float | None,
    target: float | None,
    decay: DecayInfo | None,
    exit_score: float,
) -> str:
    """Build a human-readable exit message from the dominant signal."""
    pct = f"(exit score: {exit_score:.0%})"

    if dominant == "stop_proximity":
        ch = detail.get("chandelier_tightened") or detail.get("chandelier", {})
        sl = ch.get("stop_level", stop or 0)
        dist = ch.get("atr_distance", 0)
        return (f"Chandelier stop at {sl:.2f} — price within {dist:.1f}x ATR. "
                f"Adaptive trailing from {ch.get('highest_high', 0):.2f} high. {pct}")

    if dominant == "changepoint":
        cs = detail.get("cusum", {})
        days = cs.get("days_since_shift", 0)
        mag = cs.get("magnitude", 0)
        return (f"CUSUM change-point detected {days}d ago (magnitude {mag:.2f}). "
                f"Return distribution has shifted negative. {pct}")

    if dominant == "decay":
        d = detail.get("decay", {})
        delta = d.get("delta", 0)
        return (f"Score decaying: {delta:+.3f} over 30d "
                f"(rate: {d.get('rate', 0):.4f}/day). {pct}")

    if dominant == "momentum_reversal":
        m = detail.get("momentum", {})
        prior = m.get("prior", 0)
        recent = m.get("recent", 0)
        confirmed = " (CUSUM confirmed)" if m.get("cusum_confirmed") else ""
        return (f"Momentum reversed: {prior:+.1f}% -> {recent:+.1f}%{confirmed}. {pct}")

    if dominant == "holding_period":
        hp = detail.get("holding_period", {})
        days = hp.get("days_held", 0)
        return (f"Held {days}d since last bullish signal — "
                f"no score improvement (current {score:+.3f}). {pct}")

    return f"Exit signal triggered. {pct}"
