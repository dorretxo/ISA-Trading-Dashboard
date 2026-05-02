"""One-command warm start for the self-learning screener.

This orchestrates the safe sequence:
1. Evaluate already-recorded live signals.
2. Backfill PIT fundamentals from FMP.
3. Run PIT-safe monthly replay.
4. Build triple-barrier labels.
5. Try ML training/promotion.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import logging

import config
from engine.discovery_backtest import evaluate_matured_signals_batched, init_backtest_db
from engine.historical_replay import run_replay
from engine.labeling import update_triple_barrier_labels
from engine.ml_ranker import get_diagnostics, train_model
from utils.pit_backfill import backfill_tickers, signal_backtest_tickers
from utils.pit_store import all_tickers as pit_store_tickers

logger = logging.getLogger(__name__)


def warm_start(
    *,
    max_tickers: int | None = None,
    pit_limit: int | None = None,
    replay_start: str = "2022-01-31",
    replay_end: str | None = None,
    skip_pit: bool = False,
    resume_pit: bool = False,
    skip_replay: bool = False,
) -> dict:
    init_backtest_db()
    summary: dict = {}

    summary["evaluated_live_pairs"] = evaluate_matured_signals_batched(
        horizons=getattr(config, "SIGNAL_BACKTEST_BATCH_HORIZONS", [5, 10, 30, 60, 90])
    )

    tickers = signal_backtest_tickers(max_tickers=max_tickers)
    summary["tickers"] = len(tickers)

    if not skip_pit:
        pit_backfill_tickers = tickers
        if resume_pit:
            already = set(pit_store_tickers())
            pit_backfill_tickers = [t for t in tickers if t not in already]
            summary["pit_resume_skipped_existing"] = len(tickers) - len(pit_backfill_tickers)
        pit_results = backfill_tickers(
            pit_backfill_tickers,
            limit=pit_limit or getattr(config, "PIT_BACKFILL_DEFAULT_QUARTERS", 40),
        )
        summary["pit_snapshots"] = sum(pit_results.values())
        summary["pit_tickers_with_data"] = sum(1 for v in pit_results.values() if v > 0)
        summary["pit_tickers_attempted"] = len(pit_backfill_tickers)
    else:
        summary["pit_snapshots"] = "skipped"

    if not skip_replay:
        replay_stats = run_replay(
            start=replay_start,
            end=replay_end or datetime.now().date().isoformat(),
            tickers=tickers,
            max_tickers=max_tickers,
        )
        summary["replay"] = replay_stats.__dict__
    else:
        summary["replay"] = "skipped"

    summary["triple_barrier_labels"] = update_triple_barrier_labels()
    summary["ml_train_ready"] = train_model()
    summary["ml_diagnostics"] = get_diagnostics()
    return summary


def _main() -> None:
    parser = argparse.ArgumentParser(description="Warm-start screener learning safely")
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--pit-limit", type=int, default=getattr(config, "PIT_BACKFILL_DEFAULT_QUARTERS", 40))
    parser.add_argument("--replay-start", default="2022-01-31")
    parser.add_argument("--replay-end", default=datetime.now().date().isoformat())
    parser.add_argument("--skip-pit", action="store_true")
    parser.add_argument("--resume-pit", action="store_true", help="Skip tickers already present in pit_fundamentals.json")
    parser.add_argument("--skip-replay", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = warm_start(
        max_tickers=args.max_tickers,
        pit_limit=args.pit_limit,
        replay_start=args.replay_start,
        replay_end=args.replay_end,
        skip_pit=args.skip_pit,
        resume_pit=args.resume_pit,
        skip_replay=args.skip_replay,
    )
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    _main()
