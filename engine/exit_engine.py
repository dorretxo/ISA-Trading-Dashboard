"""Exit Intelligence Engine — signal decay detection and exit timing.

The entry side of the system is much richer than the exit side.
This module adds:

1. Signal Decay Tracking — monitors how a holding's score has changed
   since its last BUY/STRONG BUY signal. Detects deterioration patterns.

2. Exit Signals — generates explicit exit recommendations based on:
   - Score decay rate (score dropping X% over N days)
   - Stop-loss proximity (price approaching stop within ATR band)
   - Momentum reversal (positive → negative momentum flip)
   - Holding period excess (held too long without improvement)
   - Target approach (price near take-profit, consider locking gains)

3. Signal Freshness — tracks age of the last bullish signal for each
   holding. Older signals are less reliable.

Public API:
    assess_exits(results, holdings) -> list[ExitSignal]
    get_signal_decay(ticker) -> DecayInfo | None
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import yfinance as yf

import config
from engine.paper_trading import _connect
from engine.discovery_backtest import init_backtest_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DECAY_LOOKBACK_DAYS = 30       # Compare current score to score N days ago
DECAY_THRESHOLD = -0.15        # Score drop of 0.15+ triggers decay alert
MOMENTUM_FLIP_WINDOW = 10     # Days to detect momentum reversal
STOP_PROXIMITY_ATR = 1.0      # Alert when price within 1 ATR of stop
MAX_HOLD_DAYS_NO_IMPROVE = 90  # Flag if held 90+ days with no score improvement
TARGET_PROXIMITY_PCT = 0.03    # Alert when within 3% of take-profit


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DecayInfo:
    """Score decay analysis for a single holding."""
    ticker: str
    current_score: float
    score_30d_ago: float | None
    score_delta: float | None         # current - 30d ago
    decay_rate: float | None          # delta / days
    days_since_last_bullish: int | None
    last_bullish_score: float | None
    last_bullish_date: str | None
    trend: str                        # "improving" | "stable" | "decaying" | "unknown"


@dataclass
class ExitSignal:
    """An exit recommendation for a holding."""
    ticker: str
    name: str
    signal_type: str           # decay | stop_proximity | momentum_reversal |
                                # holding_period | target_approach
    severity: str              # "warning" | "action_needed" | "urgent"
    message: str
    current_score: float
    current_price: float
    detail: dict               # signal-specific data


# ---------------------------------------------------------------------------
# Signal decay from backtest history
# ---------------------------------------------------------------------------

def get_signal_decay(ticker: str) -> DecayInfo | None:
    """Analyse score trend for a ticker from historical signals."""
    init_backtest_db()

    with _connect() as conn:
        # Get recent portfolio signals for this ticker
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
    days_span = DECAY_LOOKBACK_DAYS
    decay_rate = delta / days_span if delta is not None else None

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

    # Determine trend
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
# Exit signal assessment
# ---------------------------------------------------------------------------

def assess_exits(results: list[dict], holdings: list[dict]) -> list[ExitSignal]:
    """Generate exit signals for all holdings.

    Checks multiple exit conditions and returns prioritised recommendations.
    """
    exit_signals = []
    result_map = {r["ticker"]: r for r in results}

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
        atr = r.get("atr") or (current_price * 0.02)  # fallback 2%

        # 1. Score decay
        decay = get_signal_decay(ticker)
        if decay and decay.trend == "decaying" and decay.score_delta is not None:
            severity = "urgent" if decay.score_delta < -0.25 else "action_needed"
            exit_signals.append(ExitSignal(
                ticker=ticker, name=name,
                signal_type="decay",
                severity=severity,
                message=f"Score decaying: {decay.score_delta:+.3f} over 30d "
                        f"({decay.current_score:.3f} ← {decay.score_30d_ago:.3f})",
                current_score=current_score,
                current_price=current_price,
                detail={
                    "score_delta": decay.score_delta,
                    "score_30d_ago": decay.score_30d_ago,
                    "decay_rate": decay.decay_rate,
                    "trend": decay.trend,
                },
            ))

        # 2. Stop-loss proximity
        if stop_loss and current_price > 0 and atr > 0:
            distance_to_stop = (current_price - stop_loss) / current_price
            atr_distance = (current_price - stop_loss) / atr
            if 0 < atr_distance <= STOP_PROXIMITY_ATR:
                exit_signals.append(ExitSignal(
                    ticker=ticker, name=name,
                    signal_type="stop_proximity",
                    severity="urgent",
                    message=f"Price within {atr_distance:.1f}x ATR of stop-loss "
                            f"({current_price:.2f} vs stop {stop_loss:.2f})",
                    current_score=current_score,
                    current_price=current_price,
                    detail={
                        "stop_loss": stop_loss,
                        "distance_pct": round(distance_to_stop * 100, 2),
                        "atr_distance": round(atr_distance, 2),
                    },
                ))

        # 3. Momentum reversal
        try:
            data = yf.download(ticker, period="30d", progress=False, auto_adjust=True)
            if data is not None and len(data) >= MOMENTUM_FLIP_WINDOW * 2:
                closes = data["Close"].values.flatten()
                recent_mom = (closes[-1] / closes[-MOMENTUM_FLIP_WINDOW] - 1)
                prior_mom = (closes[-MOMENTUM_FLIP_WINDOW] / closes[-MOMENTUM_FLIP_WINDOW * 2] - 1)

                if prior_mom > 0.02 and recent_mom < -0.02:
                    exit_signals.append(ExitSignal(
                        ticker=ticker, name=name,
                        signal_type="momentum_reversal",
                        severity="warning",
                        message=f"Momentum reversed: {prior_mom*100:+.1f}% → {recent_mom*100:+.1f}% "
                                f"(10-day windows)",
                        current_score=current_score,
                        current_price=current_price,
                        detail={
                            "prior_momentum": round(prior_mom * 100, 2),
                            "recent_momentum": round(recent_mom * 100, 2),
                        },
                    ))
        except Exception:
            pass

        # 4. Holding period without improvement
        if decay and decay.days_since_last_bullish is not None:
            if decay.days_since_last_bullish > MAX_HOLD_DAYS_NO_IMPROVE and current_score < 0.25:
                exit_signals.append(ExitSignal(
                    ticker=ticker, name=name,
                    signal_type="holding_period",
                    severity="warning",
                    message=f"Held {decay.days_since_last_bullish}d since last bullish signal "
                            f"(score was {decay.last_bullish_score:.3f}, now {current_score:.3f})",
                    current_score=current_score,
                    current_price=current_price,
                    detail={
                        "days_held": decay.days_since_last_bullish,
                        "last_bullish_date": decay.last_bullish_date,
                        "last_bullish_score": decay.last_bullish_score,
                    },
                ))

        # 5. Target approach — consider locking gains
        if take_profit and current_price > 0:
            distance_to_target = (take_profit - current_price) / current_price
            if 0 < distance_to_target <= TARGET_PROXIMITY_PCT:
                exit_signals.append(ExitSignal(
                    ticker=ticker, name=name,
                    signal_type="target_approach",
                    severity="action_needed",
                    message=f"Within {distance_to_target*100:.1f}% of take-profit target "
                            f"({current_price:.2f} → {take_profit:.2f}). Consider locking gains.",
                    current_score=current_score,
                    current_price=current_price,
                    detail={
                        "take_profit": take_profit,
                        "distance_pct": round(distance_to_target * 100, 2),
                    },
                ))

    # Sort: urgent first, then action_needed, then warning
    severity_order = {"urgent": 0, "action_needed": 1, "warning": 2}
    exit_signals.sort(key=lambda s: severity_order.get(s.severity, 3))

    return exit_signals
