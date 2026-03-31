"""Discovery Evaluation Harness — measures whether recommendations improve.

Tracks:
1. Forward returns of top-10 / top-30 picks vs SPY benchmark
2. Hit rate (% of picks with positive forward return)
3. Swap recommendation outcomes (did the swap actually help?)
4. Ranking stability (how much does the top-10 change day to day?)
5. Per-country / per-sector / per-tier hit rates

This is the measurement layer — without it, we're tuning blind.

Usage:
    from engine.discovery_eval import generate_discovery_report
    report = generate_discovery_report()
    print(report.summary)
"""

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from engine.discovery_backtest import init_backtest_db
from engine.paper_trading import _connect

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DiscoveryQualityReport:
    """Complete discovery quality assessment."""
    # Top-N forward returns
    top10_avg_return_30d: float = 0.0
    top10_avg_return_90d: float = 0.0
    top30_avg_return_30d: float = 0.0
    top30_avg_return_90d: float = 0.0

    # Hit rates
    top10_hit_rate_30d: float = 0.0   # % with positive 30d return
    top10_hit_rate_90d: float = 0.0
    top30_hit_rate_30d: float = 0.0
    top30_hit_rate_90d: float = 0.0

    # Benchmark comparison
    spy_avg_return_30d: float = 0.0
    spy_avg_return_90d: float = 0.0
    top10_excess_30d: float = 0.0   # top10 - SPY
    top10_excess_90d: float = 0.0

    # Swap outcomes
    swap_count: int = 0
    swap_success_rate: float = 0.0  # % where buy outperformed sell
    swap_avg_delta: float = 0.0     # avg (buy_return - sell_return)

    # Ranking stability
    top10_overlap_pct: float = 0.0  # avg % overlap between consecutive runs

    # Breakdowns
    by_sector: dict = field(default_factory=dict)
    by_country: dict = field(default_factory=dict)
    by_action: dict = field(default_factory=dict)

    # Meta
    total_signals: int = 0
    evaluated_signals: int = 0
    date_range: str = ""
    summary: str = ""


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def generate_discovery_report(lookback_days: int = 180) -> DiscoveryQualityReport:
    """Generate a comprehensive quality report for discovery recommendations.

    Args:
        lookback_days: How far back to look for evaluated signals

    Returns:
        DiscoveryQualityReport with all metrics computed
    """
    report = DiscoveryQualityReport()

    init_backtest_db()

    try:
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()

            # Get all discovery signals with evaluations
            rows = conn.execute("""
                SELECT * FROM signal_backtest
                WHERE source = 'discovery'
                AND run_date >= ?
                ORDER BY run_date DESC, COALESCE(final_rank, aggregate_score) DESC
            """, (cutoff,)).fetchall()

            if not rows:
                report.summary = f"No discovery signals found in last {lookback_days} days."
                return report

            report.total_signals = len(rows)

            # Group by run date
            runs = {}
            for r in rows:
                date = r["run_date"][:10]
                runs.setdefault(date, []).append(dict(r))

            # --- Forward returns by rank ---
            _compute_forward_returns(runs, report)

            # --- Benchmark comparison ---
            _compute_benchmark_comparison(runs, report, conn)

            # --- Swap outcomes ---
            _compute_swap_outcomes(report, conn, cutoff)

            # --- Ranking stability ---
            _compute_ranking_stability(runs, report)

            # --- Breakdowns ---
            _compute_breakdowns(rows, report)

            # --- Summary ---
            report.summary = _build_summary(report)

    except Exception as e:
        logger.error("Discovery evaluation failed: %s", e)
        report.summary = f"Evaluation error: {e}"

    return report


def _compute_forward_returns(runs: dict, report: DiscoveryQualityReport):
    """Compute top-10 and top-30 forward returns across all runs."""
    top10_returns_30d = []
    top10_returns_90d = []
    top30_returns_30d = []
    top30_returns_90d = []
    evaluated = 0

    for date, signals in runs.items():
        # Sort by live discovery ordering: final_rank first, aggregate as fallback.
        ranked = sorted(signals, key=_rank_score, reverse=True)

        for i, s in enumerate(ranked):
            ret_30d = s.get("return_30d")
            ret_90d = s.get("return_90d")

            if ret_30d is not None:
                evaluated += 1
                if i < 30:
                    top30_returns_30d.append(ret_30d)
                    if i < 10:
                        top10_returns_30d.append(ret_30d)

            if ret_90d is not None:
                if i < 30:
                    top30_returns_90d.append(ret_90d)
                    if i < 10:
                        top10_returns_90d.append(ret_90d)

    report.evaluated_signals = evaluated

    if top10_returns_30d:
        report.top10_avg_return_30d = float(np.mean(top10_returns_30d))
        report.top10_hit_rate_30d = float(np.mean([1 if r > 0 else 0 for r in top10_returns_30d]))
    if top10_returns_90d:
        report.top10_avg_return_90d = float(np.mean(top10_returns_90d))
        report.top10_hit_rate_90d = float(np.mean([1 if r > 0 else 0 for r in top10_returns_90d]))
    if top30_returns_30d:
        report.top30_avg_return_30d = float(np.mean(top30_returns_30d))
        report.top30_hit_rate_30d = float(np.mean([1 if r > 0 else 0 for r in top30_returns_30d]))
    if top30_returns_90d:
        report.top30_avg_return_90d = float(np.mean(top30_returns_90d))
        report.top30_hit_rate_90d = float(np.mean([1 if r > 0 else 0 for r in top30_returns_90d]))

    date_keys = sorted(runs.keys())
    if date_keys:
        report.date_range = f"{date_keys[0]} to {date_keys[-1]}"


def _compute_benchmark_comparison(runs: dict, report: DiscoveryQualityReport, conn):
    """Compare top-10 returns against SPY benchmark."""
    try:
        spy_rows = conn.execute("""
            SELECT return_30d, return_90d FROM signal_backtest
            WHERE ticker = 'SPY' OR ticker = '^GSPC'
            ORDER BY run_date DESC LIMIT 20
        """).fetchall()

        if spy_rows:
            spy_30d = [r["return_30d"] for r in spy_rows if r["return_30d"] is not None]
            spy_90d = [r["return_90d"] for r in spy_rows if r["return_90d"] is not None]
            if spy_30d:
                report.spy_avg_return_30d = float(np.mean(spy_30d))
            if spy_90d:
                report.spy_avg_return_90d = float(np.mean(spy_90d))
    except Exception:
        pass

    # Use SPY returns from yfinance as fallback
    if report.spy_avg_return_30d == 0 and report.spy_avg_return_90d == 0:
        try:
            import yfinance as yf
            spy = yf.download("SPY", period="120d", progress=False, auto_adjust=True)
            if spy is not None and len(spy) >= 90:
                close = spy["Close"]
                if hasattr(close, "iloc"):
                    if hasattr(close, "columns"):
                        close = close.iloc[:, 0]
                    report.spy_avg_return_30d = float(close.iloc[-1] / close.iloc[-22] - 1) * 100
                    report.spy_avg_return_90d = float(close.iloc[-1] / close.iloc[-64] - 1) * 100
        except Exception:
            pass

    report.top10_excess_30d = report.top10_avg_return_30d - report.spy_avg_return_30d
    report.top10_excess_90d = report.top10_avg_return_90d - report.spy_avg_return_90d


def _compute_swap_outcomes(report: DiscoveryQualityReport, conn, cutoff: str):
    """Evaluate swap recommendations: did the buy outperform the sell?"""
    try:
        # Look for swap pairs in the paper trading log or orchestrator signals
        swaps = conn.execute("""
            SELECT s1.ticker as sell_ticker, s1.return_90d as sell_return,
                   s2.ticker as buy_ticker, s2.return_90d as buy_return,
                   s1.run_date
            FROM signal_backtest s1
            JOIN signal_backtest s2 ON s1.run_date = s2.run_date
            WHERE s1.source = 'portfolio' AND s1.action IN ('SELL', 'STRONG SELL')
            AND s2.source = 'discovery' AND s2.action IN ('BUY', 'STRONG BUY')
            AND s1.return_90d IS NOT NULL AND s2.return_90d IS NOT NULL
            AND s1.run_date >= ?
            ORDER BY s1.run_date DESC
        """, (cutoff,)).fetchall()

        if swaps:
            report.swap_count = len(swaps)
            deltas = [s["buy_return"] - s["sell_return"] for s in swaps]
            report.swap_avg_delta = float(np.mean(deltas))
            report.swap_success_rate = float(np.mean([1 if d > 0 else 0 for d in deltas]))
    except Exception as e:
        logger.debug("Swap evaluation: %s", e)


def _compute_ranking_stability(runs: dict, report: DiscoveryQualityReport):
    """Measure how stable the top-10 is across consecutive runs."""
    dates = sorted(runs.keys())
    if len(dates) < 2:
        report.top10_overlap_pct = 1.0
        return

    overlaps = []
    for i in range(1, len(dates)):
        prev_top10 = set(
            s.get("ticker", "") for s in
            sorted(runs[dates[i - 1]], key=_rank_score, reverse=True)[:10]
        )
        curr_top10 = set(
            s.get("ticker", "") for s in
            sorted(runs[dates[i]], key=_rank_score, reverse=True)[:10]
        )
        if prev_top10 and curr_top10:
            overlap = len(prev_top10 & curr_top10) / 10.0
            overlaps.append(overlap)

    report.top10_overlap_pct = float(np.mean(overlaps)) if overlaps else 1.0


def _compute_breakdowns(rows: list, report: DiscoveryQualityReport):
    """Compute hit rates by sector, country, and action."""
    sector_data = {}
    country_data = {}
    action_data = {}

    for r in rows:
        ret = r.get("return_90d") if isinstance(r, dict) else r["return_90d"]
        if ret is None:
            continue

        hit = 1 if ret > 0 else 0

        # By sector
        sector = (r.get("sector") if isinstance(r, dict) else r["sector"]) or "Unknown"
        sector_data.setdefault(sector, []).append((ret, hit))

        # By country (extract from exchange suffix)
        ticker = r.get("ticker") if isinstance(r, dict) else r["ticker"]
        country = _ticker_to_country(ticker or "")
        country_data.setdefault(country, []).append((ret, hit))

        # By action
        action = (r.get("action") if isinstance(r, dict) else r["action"]) or "Unknown"
        action_data.setdefault(action, []).append((ret, hit))

    for label, data_dict, target in [
        ("sector", sector_data, report.by_sector),
        ("country", country_data, report.by_country),
        ("action", action_data, report.by_action),
    ]:
        for key, values in data_dict.items():
            returns = [v[0] for v in values]
            hits = [v[1] for v in values]
            target[key] = {
                "count": len(values),
                "avg_return": round(float(np.mean(returns)), 2),
                "hit_rate": round(float(np.mean(hits)), 3),
                "best": round(float(max(returns)), 2),
                "worst": round(float(min(returns)), 2),
            }


def _rank_score(signal: dict) -> float:
    """Discovery ordering metric: final_rank, falling back to aggregate_score."""
    rank = signal.get("final_rank")
    if rank is not None:
        try:
            return float(rank)
        except (TypeError, ValueError):
            pass
    agg = signal.get("aggregate_score", 0)
    try:
        return float(agg)
    except (TypeError, ValueError):
        return 0.0


def _ticker_to_country(ticker: str) -> str:
    """Infer country from ticker suffix."""
    if ".L" in ticker:
        return "UK"
    elif ".DE" in ticker:
        return "DE"
    elif ".PA" in ticker:
        return "FR"
    elif ".MC" in ticker:
        return "ES"
    elif ".MI" in ticker:
        return "IT"
    elif ".AS" in ticker:
        return "NL"
    elif ".SW" in ticker:
        return "CH"
    elif ".TO" in ticker:
        return "CA"
    elif ".AX" in ticker:
        return "AU"
    elif ".HK" in ticker:
        return "HK"
    elif "." not in ticker:
        return "US"
    return "Other"


def _build_summary(report: DiscoveryQualityReport) -> str:
    """Build human-readable summary."""
    lines = [
        f"Discovery Quality Report ({report.date_range})",
        f"  Signals: {report.total_signals} recorded, {report.evaluated_signals} evaluated",
        "",
        "  Top-10 Forward Returns:",
        f"    30d: {report.top10_avg_return_30d:+.1f}% (hit rate {report.top10_hit_rate_30d:.0%})",
        f"    90d: {report.top10_avg_return_90d:+.1f}% (hit rate {report.top10_hit_rate_90d:.0%})",
        f"    vs SPY 30d: {report.spy_avg_return_30d:+.1f}% → excess {report.top10_excess_30d:+.1f}%",
        f"    vs SPY 90d: {report.spy_avg_return_90d:+.1f}% → excess {report.top10_excess_90d:+.1f}%",
        "",
        "  Top-30 Forward Returns:",
        f"    30d: {report.top30_avg_return_30d:+.1f}% (hit rate {report.top30_hit_rate_30d:.0%})",
        f"    90d: {report.top30_avg_return_90d:+.1f}% (hit rate {report.top30_hit_rate_90d:.0%})",
        "",
        f"  Ranking Stability: {report.top10_overlap_pct:.0%} top-10 overlap between runs",
    ]

    if report.swap_count > 0:
        lines += [
            "",
            f"  Swap Outcomes ({report.swap_count} swaps):",
            f"    Success rate: {report.swap_success_rate:.0%}",
            f"    Avg delta (buy - sell): {report.swap_avg_delta:+.1f}%",
        ]

    # Best/worst sectors
    if report.by_sector:
        sorted_sectors = sorted(report.by_sector.items(),
                                key=lambda x: x[1]["avg_return"], reverse=True)
        if len(sorted_sectors) >= 2:
            best = sorted_sectors[0]
            worst = sorted_sectors[-1]
            lines += [
                "",
                f"  Best sector: {best[0]} ({best[1]['avg_return']:+.1f}%, n={best[1]['count']})",
                f"  Worst sector: {worst[0]} ({worst[1]['avg_return']:+.1f}%, n={worst[1]['count']})",
            ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick summary for email/dashboard
# ---------------------------------------------------------------------------

def get_discovery_scorecard() -> dict:
    """Return a compact scorecard dict for embedding in emails or dashboard.

    Returns dict with keys: top10_hit_rate, top10_avg_return, excess_vs_spy,
    swap_success_rate, ranking_stability, total_evaluated.
    """
    report = generate_discovery_report(lookback_days=90)
    return {
        "top10_hit_rate_90d": round(report.top10_hit_rate_90d, 3),
        "top10_avg_return_90d": round(report.top10_avg_return_90d, 2),
        "top30_hit_rate_90d": round(report.top30_hit_rate_90d, 3),
        "excess_vs_spy_90d": round(report.top10_excess_90d, 2),
        "swap_success_rate": round(report.swap_success_rate, 3),
        "ranking_stability": round(report.top10_overlap_pct, 3),
        "total_evaluated": report.evaluated_signals,
        "summary": report.summary,
    }
