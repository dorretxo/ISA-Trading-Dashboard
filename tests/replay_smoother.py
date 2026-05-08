"""Replay-based validation of the exit-action smoother.

Read-only walk over ``signal_backtest`` (default ``replay_pit_v1`` source —
65k rows, 1265 tickers with 30+ days history, full 30/60/90d return coverage).

For each (ticker, run_date, aggregate_score) row:

1. Synthesize the **raw** legacy action via the production threshold map.
2. Apply :func:`engine.action_smoother.smooth_exit_action` chronologically per
   ticker — using only past rows to compute σ_score and persistence (no
   look-ahead).
3. Compare RAW vs SMOOTHED on five metrics:
   - Action churn (flips per 30 days)
   - False-exit rate  ( SELL/STRONG SELL emitted AND return_30d > 0 )
   - Missed-exit rate ( KEEP held AND return_60d < -15% )
   - Smoother latency (days from first raw SELL to first smoothed SELL)
   - Action-as-cash Sharpe proxy on forward 30d returns

Run as a script:  python -m tests.replay_smoother  (or pytest-driven via
``test_replay_metrics`` once data is available).
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

import config
from engine.action_smoother import (
    DOWNSIDE_ACTIONS,
    smooth_exit_action,
)
from engine.score_history import ScoreHistoryRow

try:
    from engine.paper_trading import DB_PATH as _DEFAULT_DB_PATH
    DEFAULT_DB_PATH = Path(_DEFAULT_DB_PATH)
except Exception:
    DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / getattr(
        config, "PAPER_TRADING_DB", "paper_trading.db"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _legacy_action(score: float) -> str:
    """Mirror of engine.scoring.py:340-349 threshold map."""
    if score is None:
        return "KEEP"
    if score >= config.SCORE_STRONG_BUY_THRESHOLD:
        return "STRONG BUY"
    if score >= config.SCORE_BUY_THRESHOLD:
        return "BUY"
    if score >= config.SCORE_KEEP_THRESHOLD:
        return "KEEP"
    if score >= config.SCORE_SELL_THRESHOLD:
        return "SELL"
    return "STRONG SELL"


def _date_diff_days(d1: str, d2: str) -> int:
    """Approximate calendar-day diff between two ISO date prefixes."""
    from datetime import date

    try:
        a = date.fromisoformat(d1[:10])
        b = date.fromisoformat(d2[:10])
        return abs((a - b).days)
    except Exception:
        return 0


def _count_flips(actions: list[str]) -> int:
    return sum(1 for a, b in zip(actions, actions[1:]) if a != b)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def run_replay(
    source: str = "replay_pit_v1",
    limit_tickers: int | None = None,
    db_path: Path | None = None,
) -> dict:
    """Walk all rows for ``source``, comparing RAW vs SMOOTHED behaviour."""
    db = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not db.exists():
        raise FileNotFoundError(f"DB not found: {db}")

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """SELECT ticker, run_date, aggregate_score, return_30d, return_60d, return_90d
           FROM signal_backtest
           WHERE source = ? AND aggregate_score IS NOT NULL
           ORDER BY ticker, run_date""",
        (source,),
    ).fetchall()
    conn.close()

    by_ticker: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for r in rows:
        by_ticker[r["ticker"]].append(r)

    tickers = list(by_ticker.keys())
    if limit_tickers:
        tickers = tickers[:limit_tickers]

    raw_actions_total: list[str] = []
    smoothed_actions_total: list[str] = []
    raw_flips = 0
    smoothed_flips = 0
    raw_sells = 0
    smoothed_sells = 0
    raw_false_exits = 0
    smoothed_false_exits = 0
    smoothed_missed_exits = 0
    raw_missed_exits = 0
    raw_total_days = 0
    smoothed_total_days = 0
    latencies: list[int] = []
    n_tickers = 0

    raw_action_returns: list[float] = []      # forward-30d return when action != SELL/STRONG SELL
    smoothed_action_returns: list[float] = []
    raw_held_returns: list[float] = []        # legacy: held when KEEP/BUY
    smoothed_held_returns: list[float] = []

    for tk in tickers:
        ticker_rows = by_ticker[tk]
        if len(ticker_rows) < 5:
            continue
        n_tickers += 1

        history: list[ScoreHistoryRow] = []
        raw_actions: list[str] = []
        smoothed_actions: list[str] = []
        prev_smoothed_action: str | None = None
        first_raw_sell_date: str | None = None
        first_smoothed_sell_date: str | None = None

        for r in ticker_rows:
            score = float(r["aggregate_score"])
            run_date = r["run_date"]
            ret_30 = r["return_30d"]
            ret_60 = r["return_60d"]

            raw = _legacy_action(score)
            raw_actions.append(raw)
            raw_actions_total.append(raw)
            raw_total_days += 1
            if raw in DOWNSIDE_ACTIONS:
                raw_sells += 1
                if first_raw_sell_date is None:
                    first_raw_sell_date = run_date
                if ret_30 is not None and ret_30 > 0:
                    raw_false_exits += 1
            else:
                # Held — accumulate forward 30d returns for "Sharpe proxy"
                if ret_30 is not None:
                    raw_held_returns.append(float(ret_30))
                if raw == "KEEP" and ret_60 is not None and float(ret_60) < -15.0:
                    raw_missed_exits += 1

            # Compute σ over last 30 entries
            tail = history[-30:]
            scores = [h.aggregate_score for h in tail if h.aggregate_score is not None]
            score_vol = (
                float(statistics.pstdev(scores))
                if len(scores) >= 10
                else None
            )

            # N-of-M persistence — uses raw actions from history (the gate measures raw pressure)
            window = history[-config.EXIT_SMOOTHER_PERSISTENCE_N:]
            m_sell = sum(1 for h in window if h.action in DOWNSIDE_ACTIONS)
            m_strong = sum(1 for h in window if h.action == "STRONG SELL")
            since = next(
                (h.run_date[:10] for h in window if h.action in DOWNSIDE_ACTIONS),
                None,
            )

            sm = smooth_exit_action(
                raw_score=score,
                base_action=raw,
                prev_smoothed_action=prev_smoothed_action,
                score_vol=score_vol,
                vix_pct=50.0,                  # neutral — no VIX history per row
                persistence_m_sell=m_sell,
                persistence_m_strong=m_strong,
                persistence_n=config.EXIT_SMOOTHER_PERSISTENCE_N,
                persistence_since=since,
                cusum_urgent=False,
                cusum_score=0.0,
            )
            smoothed_actions.append(sm.action)
            smoothed_actions_total.append(sm.action)
            smoothed_total_days += 1
            if sm.action in DOWNSIDE_ACTIONS:
                smoothed_sells += 1
                if first_smoothed_sell_date is None:
                    first_smoothed_sell_date = run_date
                if ret_30 is not None and float(ret_30) > 0:
                    smoothed_false_exits += 1
            else:
                if ret_30 is not None:
                    smoothed_held_returns.append(float(ret_30))
                if sm.action == "KEEP" and ret_60 is not None and float(ret_60) < -15.0:
                    smoothed_missed_exits += 1

            history.append(
                ScoreHistoryRow(
                    run_date=run_date,
                    aggregate_score=score,
                    action=raw,                # gate sees raw pressure
                    smoothed_action=sm.action,
                )
            )
            prev_smoothed_action = sm.action

        raw_flips += _count_flips(raw_actions)
        smoothed_flips += _count_flips(smoothed_actions)

        if first_raw_sell_date and first_smoothed_sell_date:
            latencies.append(_date_diff_days(first_raw_sell_date, first_smoothed_sell_date))

    def _safe_pct(num: int, den: int) -> float:
        return (num / den * 100.0) if den else 0.0

    raw_held_sharpe = (
        statistics.mean(raw_held_returns) / statistics.pstdev(raw_held_returns)
        if len(raw_held_returns) >= 30 and statistics.pstdev(raw_held_returns) > 0
        else 0.0
    )
    smoothed_held_sharpe = (
        statistics.mean(smoothed_held_returns) / statistics.pstdev(smoothed_held_returns)
        if len(smoothed_held_returns) >= 30 and statistics.pstdev(smoothed_held_returns) > 0
        else 0.0
    )

    return {
        "tickers": n_tickers,
        "raw_total_days": raw_total_days,
        "raw_flips": raw_flips,
        "smoothed_flips": smoothed_flips,
        "raw_flip_rate_pct": _safe_pct(raw_flips, raw_total_days),
        "smoothed_flip_rate_pct": _safe_pct(smoothed_flips, smoothed_total_days),
        "churn_reduction_pct": _safe_pct(raw_flips - smoothed_flips, raw_flips) if raw_flips else 0.0,
        "raw_sell_emissions": raw_sells,
        "smoothed_sell_emissions": smoothed_sells,
        "raw_false_exits": raw_false_exits,
        "smoothed_false_exits": smoothed_false_exits,
        "raw_false_exit_rate_pct": _safe_pct(raw_false_exits, raw_sells),
        "smoothed_false_exit_rate_pct": _safe_pct(smoothed_false_exits, smoothed_sells),
        "false_exit_reduction_pct": _safe_pct(raw_false_exits - smoothed_false_exits, raw_false_exits)
        if raw_false_exits else 0.0,
        "raw_missed_exits": raw_missed_exits,
        "smoothed_missed_exits": smoothed_missed_exits,
        "missed_exit_increase_pct": _safe_pct(smoothed_missed_exits - raw_missed_exits, raw_missed_exits)
        if raw_missed_exits else 0.0,
        "median_latency_days": statistics.median(latencies) if latencies else 0,
        "n_latency_samples": len(latencies),
        "raw_held_30d_sharpe": raw_held_sharpe,
        "smoothed_held_30d_sharpe": smoothed_held_sharpe,
        "raw_held_n": len(raw_held_returns),
        "smoothed_held_n": len(smoothed_held_returns),
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_report(stats: dict) -> None:
    print()
    print("=" * 72)
    print(f"Exit-action smoother replay  ({stats['tickers']} tickers, "
          f"{stats['raw_total_days']} ticker-days)")
    print("=" * 72)

    def row(label, raw, smoothed, delta=None, suffix=""):
        delta_str = f"   delta: {delta}" if delta is not None else ""
        print(f"  {label:<32} raw={raw:>10}{suffix}   smoothed={smoothed:>10}{suffix}{delta_str}")

    row("Action flips",
        stats["raw_flips"], stats["smoothed_flips"],
        delta=f"{stats['churn_reduction_pct']:+.1f}% churn reduction")
    row("Flip rate (per ticker-day)",
        f"{stats['raw_flip_rate_pct']:.2f}", f"{stats['smoothed_flip_rate_pct']:.2f}",
        suffix="%")
    print()
    row("SELL/STRONG SELL emissions",
        stats["raw_sell_emissions"], stats["smoothed_sell_emissions"])
    row("False exits (sell + 30d > 0)",
        stats["raw_false_exits"], stats["smoothed_false_exits"],
        delta=f"{stats['false_exit_reduction_pct']:+.1f}% reduction")
    row("False-exit rate",
        f"{stats['raw_false_exit_rate_pct']:.1f}", f"{stats['smoothed_false_exit_rate_pct']:.1f}",
        suffix="%")
    print()
    row("Missed exits (held + 60d <-15%)",
        stats["raw_missed_exits"], stats["smoothed_missed_exits"],
        delta=f"{stats['missed_exit_increase_pct']:+.1f}% relative")
    row("Median latency (raw->smoothed)",
        "-", stats["median_latency_days"],
        delta=f"n={stats['n_latency_samples']} ticker pairs")
    print()
    row("Held forward-30d Sharpe proxy",
        f"{stats['raw_held_30d_sharpe']:.3f}", f"{stats['smoothed_held_30d_sharpe']:.3f}",
        delta=f"raw_n={stats['raw_held_n']}  smoothed_n={stats['smoothed_held_n']}")

    # Merge-gate verdict
    print()
    print("=" * 72)
    pass_churn = stats["churn_reduction_pct"] >= 40.0
    pass_false = stats["false_exit_reduction_pct"] >= 25.0 or stats["raw_false_exits"] == 0
    pass_missed = stats["missed_exit_increase_pct"] <= 5.0
    pass_latency = stats["median_latency_days"] <= 4
    verdict = all((pass_churn, pass_false, pass_missed, pass_latency))
    print(f"  MERGE GATES:  churn>=40%  {'OK' if pass_churn else 'FAIL'} "
          f"({stats['churn_reduction_pct']:+.1f}%)")
    print(f"                false-exits-25%  {'OK' if pass_false else 'FAIL'} "
          f"({stats['false_exit_reduction_pct']:+.1f}%)")
    print(f"                missed-exits<=+5%  {'OK' if pass_missed else 'FAIL'} "
          f"({stats['missed_exit_increase_pct']:+.1f}%)")
    print(f"                latency<=4d  {'OK' if pass_latency else 'FAIL'} "
          f"({stats['median_latency_days']}d)")
    print(f"  VERDICT: {'PASS — ship it' if verdict else 'FAIL — tune K_VOL / M_sell and re-run'}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="replay_pit_v1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--db-path", default=None,
                        help="Override DB path (default: engine.paper_trading.DB_PATH).")
    args = parser.parse_args()

    stats = run_replay(
        source=args.source,
        limit_tickers=args.limit,
        db_path=Path(args.db_path) if args.db_path else None,
    )
    print_report(stats)


if __name__ == "__main__":
    main()
