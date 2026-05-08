"""Offline fresh replay refresh for replay/live parity diagnostics.

This command is intentionally separate from the live orchestrator. It rebuilds
PIT-safe replay rows for the latest discovery cohort, then rewrites the
replay/live parity report so factor drift is measured only on fresh comparable
rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from engine.historical_replay import run_replay
from utils.atomic_io import atomic_write_json
from utils.global_universe import is_excluded_ticker, resolve_yahoo_ticker
from utils.state_manager import load_state

logger = logging.getLogger(__name__)

_REFRESH_REPORT = ROOT / getattr(
    config,
    "HISTORICAL_REPLAY_FRESH_REPORT_PATH",
    "feature_cache/replay_parity_refresh.json",
)
_PARITY_REPORT = ROOT / getattr(
    config,
    "REPLAY_LIVE_PARITY_REPORT_PATH",
    "feature_cache/replay_live_parity_report.json",
)


def _coerce_date(value) -> date | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except Exception:
        return None


def _normalise_ticker(ticker: str) -> str:
    return resolve_yahoo_ticker(str(ticker or "").upper().strip())


def _normalise_candidates(candidates: list[dict], *, limit: int | None = None) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for candidate in candidates or []:
        ticker = _normalise_ticker(candidate.get("ticker") or candidate.get("symbol") or "")
        if not ticker or ticker in seen or is_excluded_ticker(ticker):
            continue
        item = dict(candidate)
        item["ticker"] = ticker
        out.append(item)
        seen.add(ticker)
        if limit and len(out) >= int(limit):
            break
    return out


def _latest_discovery_candidates_from_db(limit: int | None = None) -> tuple[list[dict], date | None]:
    try:
        from engine.discovery_backtest import init_backtest_db
        from engine.paper_trading import _connect

        init_backtest_db()
        with _connect() as conn:
            latest = conn.execute(
                "SELECT MAX(run_date) AS run_date FROM signal_backtest WHERE source='discovery'"
            ).fetchone()
            latest_day = _coerce_date(latest["run_date"] if latest else None)
            if latest_day is None:
                return [], None
            sql = """
                SELECT ticker, run_date, final_rank, aggregate_score, action
                FROM signal_backtest
                WHERE source='discovery' AND run_date LIKE ?
                ORDER BY final_rank DESC
            """
            params: list = [latest_day.isoformat() + "%"]
            if limit:
                sql += f" LIMIT {int(limit)}"
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows], latest_day
    except Exception as exc:
        logger.warning("Could not load latest discovery candidates from DB: %s", exc)
        return [], None


def latest_discovery_cohort(limit: int | None = None) -> tuple[list[dict], date]:
    """Return cached discovery candidates and the best replay as-of date."""
    state = load_state()
    state_candidates = _normalise_candidates(state.get("cached_discovery", []), limit=limit)
    state_day = _coerce_date(state.get("last_discovery_run"))
    if state_candidates and state_day is not None:
        return state_candidates, state_day

    db_candidates, db_day = _latest_discovery_candidates_from_db(limit=limit)
    db_candidates = _normalise_candidates(db_candidates, limit=limit)
    if db_candidates and db_day is not None:
        return db_candidates, db_day
    return [], date.today()


def replay_window(as_of: date, *, frequency: str, days: int) -> tuple[str, str]:
    freq = str(frequency or "latest").lower().strip()
    end = as_of
    if freq == "latest":
        start = end
    else:
        start = end - timedelta(days=max(1, int(days)) - 1)
    return start.isoformat(), end.isoformat()


def refresh_replay_parity(
    *,
    top_n: int | None = None,
    frequency: str = "latest",
    days: int | None = None,
    batch_size: int | None = None,
    include_forward_labels: bool = False,
    refresh_existing: bool = True,
) -> dict:
    """Run fresh offline replay for the latest discovery cohort and write parity."""
    if days is None:
        days = int(getattr(config, "HISTORICAL_REPLAY_FRESH_DEFAULT_DAYS", 7))
    candidates, as_of = latest_discovery_cohort(limit=top_n)
    tickers = [candidate["ticker"] for candidate in candidates]
    start, end = replay_window(as_of, frequency=frequency, days=days)
    stats = run_replay(
        start=start,
        end=end,
        tickers=tickers,
        batch_size=batch_size,
        frequency=frequency,
        include_forward_labels=include_forward_labels,
        refresh_existing=refresh_existing,
    )

    parity_payload: dict | None = None
    try:
        from daily_orchestrator import _write_replay_live_parity_report

        _write_replay_live_parity_report(candidates)
        if _PARITY_REPORT.exists():
            parity_payload = json.loads(_PARITY_REPORT.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Fresh replay completed but parity report write failed: %s", exc)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "offline_fresh_replay_parity",
        "as_of": as_of.isoformat(),
        "start": start,
        "end": end,
        "frequency": frequency,
        "include_forward_labels": include_forward_labels,
        "refresh_existing": refresh_existing,
        "requested_tickers": len(tickers),
        "tickers": tickers,
        "replay_stats": stats.__dict__,
        "parity_report_path": str(_PARITY_REPORT),
        "parity_summary": {
            key: parity_payload.get(key)
            for key in (
                "available",
                "sample",
                "available_pairs",
                "missing_replay",
                "stale_replay",
                "drifted_tickers",
                "drift_field_counts",
                "missing_field_counts",
                "live_missing_field_counts",
                "replay_missing_field_counts",
            )
        } if isinstance(parity_payload, dict) else None,
    }
    atomic_write_json(_REFRESH_REPORT, payload, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline fresh replay parity refresh.")
    parser.add_argument(
        "--top-n",
        type=int,
        default=int(getattr(config, "HISTORICAL_REPLAY_FRESH_DEFAULT_TOP_N", 250)),
        help="Number of latest cached discovery candidates to replay.",
    )
    parser.add_argument(
        "--frequency",
        choices=["latest", "daily", "weekly", "monthly"],
        default="latest",
        help="Replay cadence. latest writes one same-as-of row for parity.",
    )
    parser.add_argument("--days", type=int, default=int(getattr(config, "HISTORICAL_REPLAY_FRESH_DEFAULT_DAYS", 7)))
    parser.add_argument("--batch-size", type=int, default=getattr(config, "PRICE_CACHE_BATCH_SIZE", 75))
    parser.add_argument(
        "--with-forward-labels",
        action="store_true",
        help="Also compute forward labels where enough future prices exist. Off by default for fresh parity.",
    )
    parser.add_argument(
        "--no-refresh-existing",
        action="store_true",
        help="Skip rows that already exist for the same ticker/source/run_date.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    payload = refresh_replay_parity(
        top_n=args.top_n,
        frequency=args.frequency,
        days=args.days,
        batch_size=args.batch_size,
        include_forward_labels=args.with_forward_labels,
        refresh_existing=not args.no_refresh_existing,
    )
    print(json.dumps({
        "requested_tickers": payload["requested_tickers"],
        "replay_stats": payload["replay_stats"],
        "parity_summary": payload["parity_summary"],
        "report": str(_REFRESH_REPORT),
    }, indent=2))


if __name__ == "__main__":
    main()
