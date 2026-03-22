"""Evaluation Harness — rigorous out-of-sample performance measurement.

Computes rolling and aggregate metrics from the signal_backtest table:

  - Sharpe / Sortino ratios
  - Maximum drawdown
  - Hit rate (% of signals with positive returns)
  - Benchmark-relative returns (vs SPY)
  - Turnover-adjusted returns
  - Per-regime performance breakdown
  - Per-horizon performance breakdown
  - Rolling IC (information coefficient) stability

Public API:
    compute_scorecard(source, min_signals) -> Scorecard
    compute_rolling_ic(pillar, window) -> list[dict]
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

import config
from engine.paper_trading import _connect
from engine.discovery_backtest import init_backtest_db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HorizonStats:
    """Performance at a single evaluation horizon."""
    horizon: str           # "30d" | "60d" | "90d"
    avg_return: float
    median_return: float
    std_return: float
    hit_rate: float        # % positive returns
    avg_benchmark: float   # average SPY return (90d only)
    alpha: float           # avg_return - avg_benchmark
    best: float
    worst: float
    sample_size: int


@dataclass
class RegimeBreakdown:
    """Performance within a market regime."""
    regime: str
    avg_return_90d: float
    hit_rate: float
    best_pillar: str | None
    sample_size: int


@dataclass
class Scorecard:
    """Complete evaluation scorecard."""
    source: str                              # "portfolio" | "discovery" | "all"
    as_of: str                               # ISO timestamp
    total_signals: int
    evaluated_signals: int
    pending_signals: int

    # Aggregate metrics (90d horizon)
    sharpe_ratio: float | None               # annualised
    sortino_ratio: float | None              # annualised (downside deviation)
    max_drawdown: float | None               # worst peak-to-trough in signal returns
    calmar_ratio: float | None               # return / max_drawdown

    # Hit rates
    overall_hit_rate: float | None           # % of 90d returns > 0
    action_accuracy: float | None            # % where action direction was correct
    beat_benchmark_rate: float | None        # % that beat SPY over 90d

    # Per-horizon
    horizons: list[HorizonStats] = field(default_factory=list)

    # Per-regime
    regimes: list[RegimeBreakdown] = field(default_factory=list)

    # Stop-loss / take-profit effectiveness
    stop_hit_rate: float | None = None
    target_hit_rate: float | None = None
    avg_stop_day: float | None = None
    avg_target_day: float | None = None

    # Forecast accuracy
    avg_forecast_error_5d: float | None = None
    avg_forecast_error_63d: float | None = None

    # Turnover
    avg_turnover_per_signal: float | None = None

    # Rolling stability
    ic_stability: float | None = None        # std of rolling IC


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_scorecard(source: str = "all", min_signals: int = 5) -> Scorecard | None:
    """Compute a comprehensive scorecard from evaluated backtest signals.

    Args:
        source: 'portfolio', 'discovery', or 'all'
        min_signals: minimum evaluated signals required

    Returns Scorecard or None if insufficient data.
    """
    init_backtest_db()

    with _connect() as conn:
        if source == "all":
            all_sigs = [dict(r) for r in conn.execute("SELECT * FROM signal_backtest").fetchall()]
            evaluated = [dict(r) for r in conn.execute(
                "SELECT * FROM signal_backtest WHERE evaluated_90d = 1"
            ).fetchall()]
        else:
            all_sigs = [dict(r) for r in conn.execute(
                "SELECT * FROM signal_backtest WHERE source=?", (source,)
            ).fetchall()]
            evaluated = [dict(r) for r in conn.execute(
                "SELECT * FROM signal_backtest WHERE source=? AND evaluated_90d = 1",
                (source,),
            ).fetchall()]

    total = len(all_sigs)
    n_eval = len(evaluated)
    pending = total - n_eval

    if n_eval < min_signals:
        return Scorecard(
            source=source,
            as_of=datetime.now().isoformat(timespec="seconds"),
            total_signals=total,
            evaluated_signals=n_eval,
            pending_signals=pending,
            sharpe_ratio=None, sortino_ratio=None,
            max_drawdown=None, calmar_ratio=None,
            overall_hit_rate=None, action_accuracy=None,
            beat_benchmark_rate=None,
        )

    returns_90d = np.array([s["return_90d"] or 0 for s in evaluated])

    # --- Sharpe (annualised from 90d returns) ---
    # Scale factor: 252/63 ≈ 4 periods per year
    periods_per_year = 252 / 63
    mean_ret = np.mean(returns_90d) / 100  # decimal
    std_ret = np.std(returns_90d, ddof=1) / 100 if len(returns_90d) > 1 else 0.01
    rf_per_period = 0.04 / periods_per_year  # ~4% annual risk-free
    sharpe = (mean_ret - rf_per_period) / std_ret * np.sqrt(periods_per_year) if std_ret > 0 else None

    # --- Sortino (downside deviation) ---
    downside = returns_90d[returns_90d < 0] / 100
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.01
    sortino = (mean_ret - rf_per_period) / downside_std * np.sqrt(periods_per_year) if downside_std > 0 else None

    # --- Max drawdown (sequential signal returns as a simulated equity curve) ---
    # Sort by run_date to simulate chronological entry
    sorted_eval = sorted(evaluated, key=lambda s: s["run_date"])
    equity = np.cumprod([1 + (s["return_90d"] or 0) / 100 for s in sorted_eval])
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else None

    # --- Calmar ---
    annual_ret = mean_ret * periods_per_year
    calmar = annual_ret / abs(max_dd) if max_dd and max_dd < 0 else None

    # --- Hit rates ---
    hit_rate = float(np.mean(returns_90d > 0))

    action_correct = [s["action_correct"] for s in evaluated if s["action_correct"] is not None]
    action_acc = sum(action_correct) / len(action_correct) if action_correct else None

    beat_market = [s["beat_market"] for s in evaluated if s["beat_market"] is not None]
    beat_rate = sum(beat_market) / len(beat_market) if beat_market else None

    # --- Per-horizon stats ---
    horizons = []
    for horizon, col_return in [("30d", "return_30d"), ("60d", "return_60d"), ("90d", "return_90d")]:
        col_flag = f"evaluated_{horizon}"
        h_sigs = [s for s in evaluated if s.get(col_flag) or s.get(col_return) is not None]
        h_rets = np.array([s[col_return] or 0 for s in h_sigs])

        if len(h_rets) < 3:
            continue

        spy_rets = [s.get("spy_return_90d", 0) or 0 for s in h_sigs] if horizon == "90d" else [0] * len(h_sigs)
        avg_spy = np.mean(spy_rets)

        horizons.append(HorizonStats(
            horizon=horizon,
            avg_return=round(float(np.mean(h_rets)), 2),
            median_return=round(float(np.median(h_rets)), 2),
            std_return=round(float(np.std(h_rets, ddof=1)), 2) if len(h_rets) > 1 else 0,
            hit_rate=round(float(np.mean(h_rets > 0)), 3),
            avg_benchmark=round(float(avg_spy), 2),
            alpha=round(float(np.mean(h_rets) - avg_spy), 2),
            best=round(float(np.max(h_rets)), 2),
            worst=round(float(np.min(h_rets)), 2),
            sample_size=len(h_rets),
        ))

    # --- Per-regime ---
    regime_breakdown = []
    regime_set = set(s["regime"] for s in evaluated if s["regime"])
    for regime in regime_set:
        r_sigs = [s for s in evaluated if s["regime"] == regime]
        r_rets = np.array([s["return_90d"] or 0 for s in r_sigs])
        if len(r_rets) < 3:
            continue

        # Best pillar by IC
        best_p, best_ic = None, -1
        for pillar_col in ["technical_score", "fundamental_score", "sentiment_score", "forecast_score"]:
            scores = np.array([s[pillar_col] or 0 for s in r_sigs])
            if np.std(scores) > 0:
                try:
                    from scipy.stats import spearmanr
                    ic, _ = spearmanr(scores, r_rets)
                    if not np.isnan(ic) and ic > best_ic:
                        best_ic = ic
                        best_p = pillar_col.replace("_score", "")
                except Exception:
                    pass

        regime_breakdown.append(RegimeBreakdown(
            regime=regime,
            avg_return_90d=round(float(np.mean(r_rets)), 2),
            hit_rate=round(float(np.mean(r_rets > 0)), 3),
            best_pillar=best_p,
            sample_size=len(r_rets),
        ))

    # --- Stop/target stats ---
    with_stops = [s for s in evaluated if s["stop_loss"]]
    stop_hit_rate = sum(1 for s in with_stops if s["stop_hit"]) / len(with_stops) if with_stops else None
    target_hit_rate = sum(1 for s in with_stops if s["target_hit"]) / len(with_stops) if with_stops else None
    stop_days = [s["stop_hit_day"] for s in with_stops if s["stop_hit_day"]]
    target_days = [s["target_hit_day"] for s in with_stops if s["target_hit_day"]]
    avg_stop_d = sum(stop_days) / len(stop_days) if stop_days else None
    avg_target_d = sum(target_days) / len(target_days) if target_days else None

    # --- Forecast accuracy ---
    err_5d = [s["forecast_error_5d"] for s in evaluated if s["forecast_error_5d"] is not None]
    err_63d = [s["forecast_error_63d"] for s in evaluated if s["forecast_error_63d"] is not None]

    # --- Rolling IC stability ---
    ic_values = _compute_rolling_ic_internal(evaluated, "technical_score", window=20)
    ic_std = float(np.std([x["ic"] for x in ic_values])) if len(ic_values) >= 3 else None

    return Scorecard(
        source=source,
        as_of=datetime.now().isoformat(timespec="seconds"),
        total_signals=total,
        evaluated_signals=n_eval,
        pending_signals=pending,
        sharpe_ratio=round(sharpe, 3) if sharpe is not None else None,
        sortino_ratio=round(sortino, 3) if sortino is not None else None,
        max_drawdown=round(max_dd * 100, 2) if max_dd is not None else None,
        calmar_ratio=round(calmar, 3) if calmar is not None else None,
        overall_hit_rate=round(hit_rate, 3),
        action_accuracy=round(action_acc, 3) if action_acc is not None else None,
        beat_benchmark_rate=round(beat_rate, 3) if beat_rate is not None else None,
        horizons=horizons,
        regimes=regime_breakdown,
        stop_hit_rate=round(stop_hit_rate, 3) if stop_hit_rate is not None else None,
        target_hit_rate=round(target_hit_rate, 3) if target_hit_rate is not None else None,
        avg_stop_day=round(avg_stop_d, 1) if avg_stop_d is not None else None,
        avg_target_day=round(avg_target_d, 1) if avg_target_d is not None else None,
        avg_forecast_error_5d=round(sum(err_5d) / len(err_5d), 2) if err_5d else None,
        avg_forecast_error_63d=round(sum(err_63d) / len(err_63d), 2) if err_63d else None,
        ic_stability=round(ic_std, 4) if ic_std is not None else None,
    )


# ---------------------------------------------------------------------------
# Rolling IC
# ---------------------------------------------------------------------------

def _compute_rolling_ic_internal(
    signals: list, pillar: str, window: int = 20,
) -> list[dict]:
    """Compute rolling Spearman IC for a pillar over a sliding window."""
    sorted_sigs = sorted(signals, key=lambda s: s["run_date"])
    results = []

    for i in range(window, len(sorted_sigs)):
        batch = sorted_sigs[i - window:i]
        scores = np.array([s[pillar] or 0 for s in batch])
        rets = np.array([s["return_90d"] or 0 for s in batch])

        if np.std(scores) == 0 or np.std(rets) == 0:
            continue

        try:
            from scipy.stats import spearmanr
            ic, _ = spearmanr(scores, rets)
            if np.isnan(ic):
                continue
        except Exception:
            rank_s = np.argsort(np.argsort(scores))
            rank_r = np.argsort(np.argsort(rets))
            ic = float(np.corrcoef(rank_s, rank_r)[0, 1])
            if np.isnan(ic):
                continue

        results.append({
            "date": batch[-1]["run_date"][:10],
            "ic": round(float(ic), 4),
            "sample": len(batch),
        })

    return results


def compute_rolling_ic(
    pillar: str = "technical_score",
    source: str = "all",
    window: int = 20,
) -> list[dict]:
    """Public API: compute rolling IC for a pillar.

    Args:
        pillar: column name (e.g. 'technical_score', 'forecast_score')
        source: 'portfolio', 'discovery', or 'all'
        window: rolling window size

    Returns list of {date, ic, sample} dicts.
    """
    init_backtest_db()
    with _connect() as conn:
        if source == "all":
            signals = [dict(r) for r in conn.execute(
                "SELECT * FROM signal_backtest WHERE evaluated_90d = 1 ORDER BY run_date"
            ).fetchall()]
        else:
            signals = [dict(r) for r in conn.execute(
                "SELECT * FROM signal_backtest WHERE source=? AND evaluated_90d = 1 ORDER BY run_date",
                (source,),
            ).fetchall()]

    return _compute_rolling_ic_internal(signals, pillar, window)
