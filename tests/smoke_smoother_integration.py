"""End-to-end smoke for the exit-action smoother.

Bypasses the orchestrator entry point (which has a pre-existing unrelated
import bug) and directly exercises the wired path on a real ticker from the
production DB copy:

    score_history.get_score_series  ->  action_smoother.smooth_exit_action
    ->  result dict shape  ->  record_portfolio_signals (schema migration + write)

Run from the worktree with the staged DB copy.
"""

from __future__ import annotations

import sys
import sqlite3
from datetime import datetime
from pathlib import Path

import config
from engine import score_history, action_smoother
from engine.discovery_backtest import init_backtest_db, record_portfolio_signals
from engine.paper_trading import DB_PATH


def _print(label, val):
    print(f"  {label:<28} {val}")


def main() -> int:
    print(f"DB:  {DB_PATH}  (exists={DB_PATH.exists()})")
    if not DB_PATH.exists():
        print("ERROR: DB not found in worktree - copy production state first.")
        return 1

    # -- Step 1: schema migration runs on the real DB copy --------------
    print()
    print("--- Step 1: init_backtest_db migration ---")
    conn = sqlite3.connect(str(DB_PATH))
    before = {r[1] for r in conn.execute("PRAGMA table_info(signal_backtest)")}
    conn.close()
    smoother_cols = [
        "smoothed_action", "smoothing_reason", "smoothing_since_date",
        "score_vol_30d", "persistence_m", "persistence_n",
        "smoothing_band_low", "smoothing_band_high",
    ]
    missing_before = [c for c in smoother_cols if c not in before]
    _print("smoother cols missing pre-migrate:", missing_before)

    init_backtest_db()

    conn = sqlite3.connect(str(DB_PATH))
    after = {r[1] for r in conn.execute("PRAGMA table_info(signal_backtest)")}
    conn.close()
    missing_after = [c for c in smoother_cols if c not in after]
    _print("smoother cols missing post-migrate:", missing_after)
    assert not missing_after, f"migration failed for: {missing_after}"
    print("  OK Schema migration OK")

    # -- Step 2: pick a real ticker that has score history in the DB ----
    print()
    print("--- Step 2: per-ticker smoother walk ---")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    candidate_rows = conn.execute("""
        SELECT ticker, COUNT(*) AS n FROM signal_backtest
        WHERE source = 'portfolio'
        GROUP BY ticker
        ORDER BY n DESC
        LIMIT 5
    """).fetchall()
    if not candidate_rows:
        print("  WARN: no portfolio history to test against")
        return 0
    ticker = candidate_rows[0]["ticker"]
    _print("ticker:", f"{ticker} ({candidate_rows[0]['n']} rows of history)")

    history = score_history.get_score_series(ticker, n_days=60, source="portfolio")
    _print("history rows pulled:", len(history))
    if history:
        _print("oldest run_date:", history[0].run_date)
        _print("latest run_date:", history[-1].run_date)
        _print("latest aggregate_score:", history[-1].aggregate_score)
        _print("latest action:", history[-1].action)
        _print("latest smoothed_action:", history[-1].smoothed_action)

    # -- Step 3: σ_score and persistence reads -------------------------
    score_vol = score_history.compute_score_vol(history, lookback=30)
    m_sell, since_sell = score_history.count_recent_action(
        history, score_history.DOWN_ACTIONS, n=config.EXIT_SMOOTHER_PERSISTENCE_N,
    )
    m_strong, _ = score_history.count_recent_action(
        history, score_history.STRONG_DOWN_ACTIONS, n=config.EXIT_SMOOTHER_PERSISTENCE_N,
    )
    prev = score_history.get_prev_smoothed_action(history)
    _print("score_vol_30d:", f"{score_vol:.4f}" if score_vol is not None else "-")
    _print("persistence m_sell:", f"{m_sell} of {config.EXIT_SMOOTHER_PERSISTENCE_N}")
    _print("persistence m_strong:", m_strong)
    _print("prev_smoothed_action:", prev)

    # -- Step 4: drive the smoother across a small grid of hypothetical scores --
    print()
    print("--- Step 4: smoother decision grid for current ticker ---")
    grid = [-0.10, -0.20, -0.27, -0.35, -0.55, -0.70]
    for raw in grid:
        if raw >= config.SCORE_KEEP_THRESHOLD:
            base = "KEEP"
        elif raw >= config.SCORE_SELL_THRESHOLD:
            base = "SELL"
        else:
            base = "STRONG SELL"
        sm = action_smoother.smooth_exit_action(
            raw_score=raw,
            base_action=base,
            prev_smoothed_action=prev,
            score_vol=score_vol,
            vix_pct=50.0,
            persistence_m_sell=m_sell,
            persistence_m_strong=m_strong,
            persistence_n=config.EXIT_SMOOTHER_PERSISTENCE_N,
            persistence_since=since_sell,
            cusum_urgent=False,
            cusum_score=0.0,
        )
        band = (
            f"[{sm.band_low:+.3f}, {sm.band_high:+.3f}]"
            if sm.band_low is not None else "-"
        )
        marker = "->" if sm.action != base else " "
        print(f"    score {raw:+.2f}  raw={base:<11} {marker} smoothed={sm.action:<11}"
              f"  reason={sm.reason:<24}  band={band}")

    # -- Step 5: round-trip a synthetic result dict through record_portfolio_signals --
    print()
    print("--- Step 5: persist round-trip ---")
    synthetic_result = {
        "ticker": "ZZZSMOKE",   # unique sentinel - won't collide
        "name": "Smoke Test",
        "current_price": 100.0,
        "action": "KEEP",
        "base_action": "KEEP",
        "final_action": "KEEP",
        "aggregate_score": -0.27,
        "technical_score": -0.20,
        "fundamental_score": -0.30,
        "sentiment_score": -0.10,
        "forecast_score": -0.40,
        # The eight new smoother fields:
        "smoothed_action": "KEEP",
        "smoothing_reason": "band_held_keep",
        "smoothing_since_date": None,
        "score_vol_30d": 0.045,
        "persistence_m": 1,
        "persistence_n": 5,
        "smoothing_band_low": -0.32,
        "smoothing_band_high": -0.18,
    }
    # Pre-clean any prior smoke row from a partially-completed prior run.
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("DELETE FROM signal_backtest WHERE ticker='ZZZSMOKE'")
    conn.commit()
    conn.close()

    n = record_portfolio_signals(
        [synthetic_result],
        position_weights=None,
        regime={"regime_label": "NEUTRAL", "vix_level": 20.0},
    )
    _print("rows recorded:", n)

    # Verify the row landed with smoother fields
    conn = sqlite3.connect(str(DB_PATH))
    row = conn.execute("""
        SELECT smoothed_action, smoothing_reason, score_vol_30d, persistence_m,
               smoothing_band_low, smoothing_band_high
        FROM signal_backtest WHERE ticker='ZZZSMOKE' ORDER BY id DESC LIMIT 1
    """).fetchone()

    _print("smoothed_action:", row[0])
    _print("smoothing_reason:", row[1])
    _print("score_vol_30d:", row[2])
    _print("persistence_m:", row[3])
    _print("band_low / band_high:", f"{row[4]:+.3f} / {row[5]:+.3f}" if row[4] is not None else "-")
    conn.execute("DELETE FROM signal_backtest WHERE ticker='ZZZSMOKE'")
    conn.commit()
    conn.close()
    print("  OK Round-trip OK (test row cleaned up)")

    print()
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
