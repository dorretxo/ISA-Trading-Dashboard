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
from typing import Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from engine.historical_replay import reset_pit_cache, run_replay
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
_FUNDAMENTAL_QUEUE = ROOT / getattr(
    config,
    "HISTORICAL_REPLAY_FUNDAMENTAL_REFRESH_QUEUE_PATH",
    "feature_cache/replay_fundamental_refresh_queue.json",
)
_FUNDAMENTAL_REFRESH_FIELDS = tuple(getattr(
    config,
    "HISTORICAL_REPLAY_FUNDAMENTAL_REFRESH_FIELDS",
    (
        "quality_factor_score", "value_factor_score", "qmj_factor_score",
        "quality_score_fundamental", "gross_profitability", "fcf_to_assets",
        "fundamental_score", "pe_ratio", "revenue_growth",
        "f_score", "f_score_coverage", "gpa", "gpa_score",
    ),
))


def _coerce_date(value) -> date | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except Exception:
        return None


def _normalise_ticker(ticker: str) -> str:
    return resolve_yahoo_ticker(str(ticker or "").upper().strip())


def _finite(value) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if out == out else None
    except (TypeError, ValueError):
        return None


def _safe_priority(value) -> float:
    finite = _finite(value)
    return 0.0 if finite is None else float(finite)


def _is_fmp_statement_candidate(ticker: str) -> bool:
    symbol = str(ticker or "").upper().strip()
    return bool(symbol) and "." not in symbol


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


def _latest_replay_fundamental_missing(
    tickers: list[str],
    *,
    fields: tuple[str, ...] = _FUNDAMENTAL_REFRESH_FIELDS,
) -> dict[str, dict]:
    """Return replay-side missingness for PIT-backed fundamental fields."""
    out: dict[str, dict] = {}
    if not tickers:
        return out
    try:
        from engine.discovery_backtest import init_backtest_db
        from engine.paper_trading import _connect

        init_backtest_db()
        with _connect() as conn:
            valid = {str(row[1]) for row in conn.execute("PRAGMA table_info(signal_backtest)").fetchall()}
            compare_fields = [field for field in fields if field in valid]
            select_cols = ["ticker", "run_date"] + compare_fields
            for ticker in tickers:
                row = conn.execute(
                    f"""
                    SELECT {', '.join(select_cols)}
                    FROM signal_backtest
                    WHERE UPPER(ticker)=? AND source='replay_pit_v1'
                    ORDER BY run_date DESC, id DESC
                    LIMIT 1
                    """,
                    (ticker,),
                ).fetchone()
                if not row:
                    out[ticker] = {
                        "available": False,
                        "missing_fields": list(compare_fields) or ["replay_pit_v1_row"],
                        "replay_run_date": None,
                    }
                    continue
                row_d = dict(row)
                missing = [field for field in compare_fields if _finite(row_d.get(field)) is None]
                out[ticker] = {
                    "available": True,
                    "missing_fields": missing,
                    "replay_run_date": row_d.get("run_date"),
                }
    except Exception as exc:
        logger.warning("Could not inspect replay fundamental missingness: %s", exc)
    return out


def build_fundamental_refresh_queue(
    candidates: list[dict],
    *,
    missing_by_ticker: Mapping[str, Mapping] | None = None,
    max_items: int | None = None,
    generated_at: str | None = None,
) -> dict:
    """Build a PIT fundamental refresh queue from replay-side missingness."""
    normalised = _normalise_candidates(candidates, limit=None)
    tickers = [candidate["ticker"] for candidate in normalised]
    if missing_by_ticker is None:
        missing_by_ticker = _latest_replay_fundamental_missing(tickers)

    now = generated_at or datetime.now().isoformat(timespec="seconds")
    items: list[dict] = []
    seen: set[str] = set()
    for index, candidate in enumerate(normalised):
        ticker = candidate["ticker"]
        if ticker in seen:
            continue
        seen.add(ticker)
        record = dict(missing_by_ticker.get(ticker) or {})
        missing = [
            str(field)
            for field in (record.get("missing_fields") or [])
            if field in _FUNDAMENTAL_REFRESH_FIELDS or field == "replay_pit_v1_row"
        ]
        if not missing:
            continue
        action = str(candidate.get("action") or "").upper()
        ready_status = str(candidate.get("ready_contract_status") or "").upper()
        final_rank = _safe_priority(candidate.get("final_rank"))
        aggregate = _safe_priority(candidate.get("aggregate_score"))
        fmp_candidate = _is_fmp_statement_candidate(ticker)
        critical_hits = sum(1 for field in missing if field in {"quality_factor_score", "value_factor_score", "f_score", "gpa"})
        priority = len(missing) + 0.75 * critical_hits + final_rank + 0.25 * aggregate
        if action == "STRONG BUY":
            priority += 2.0
        elif action == "BUY":
            priority += 1.0
        if ready_status == "PASS":
            priority += 0.75
        if fmp_candidate:
            priority += 0.25
        priority += max(0.0, 1.0 - index / max(1, len(normalised)))
        items.append({
            "ticker": ticker,
            "name": candidate.get("name"),
            "priority": round(priority, 3),
            "missing_fields": missing,
            "last_action": candidate.get("action"),
            "last_final_rank": candidate.get("final_rank"),
            "ready_contract_status": candidate.get("ready_contract_status"),
            "strong_buy_eligible": candidate.get("strong_buy_eligible"),
            "replay_run_date": record.get("replay_run_date"),
            "replay_available": bool(record.get("available", False)),
            "fmp_statement_candidate": fmp_candidate,
            "source": "replay_live_parity",
        })

    if max_items is None:
        max_items = int(getattr(config, "DISCOVERY_FUNDAMENTAL_REFRESH_QUEUE_MAX", 200))
    items = sorted(items, key=lambda row: row.get("priority", 0), reverse=True)[:max(0, int(max_items))]
    return {
        "generated_at": now,
        "source": "replay_live_parity",
        "count": len(items),
        "fields": list(_FUNDAMENTAL_REFRESH_FIELDS),
        "items": items,
    }


def _refresh_fundamentals_for_parity(
    candidates: list[dict],
    *,
    queue_path: str | Path | None = None,
    max_tickers: int | None = None,
    limit: int | None = None,
    sleep_seconds: float = 0.0,
    allow_yfinance_fallback: bool = False,
) -> dict:
    queue_target = Path(queue_path or _FUNDAMENTAL_QUEUE)
    queue_payload = build_fundamental_refresh_queue(candidates)
    queue_target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(queue_target, queue_payload, indent=2)

    from utils.pit_backfill import refresh_queue_tickers

    refresh_payload = refresh_queue_tickers(
        queue_path=queue_target,
        max_tickers=max_tickers,
        limit=limit,
        sleep_seconds=sleep_seconds,
        yfinance_fallback=allow_yfinance_fallback,
        fmp_only=not allow_yfinance_fallback,
        write_results=True,
    )
    reset_pit_cache()
    return {
        "queue_path": str(queue_target),
        "queue_count": queue_payload.get("count", 0),
        "queue_fmp_candidates": sum(1 for row in queue_payload.get("items", []) if row.get("fmp_statement_candidate")),
        "refresh": refresh_payload,
    }


def refresh_replay_parity(
    *,
    top_n: int | None = None,
    frequency: str = "latest",
    days: int | None = None,
    batch_size: int | None = None,
    include_forward_labels: bool = False,
    refresh_existing: bool = True,
    refresh_fundamentals: bool = False,
    fundamental_max_tickers: int | None = None,
    fundamental_limit: int | None = None,
    fundamental_sleep_seconds: float = 0.0,
    allow_yfinance_fallback: bool = False,
    fundamental_queue_path: str | Path | None = None,
) -> dict:
    """Run fresh offline replay for the latest discovery cohort and write parity."""
    if days is None:
        days = int(getattr(config, "HISTORICAL_REPLAY_FRESH_DEFAULT_DAYS", 7))
    candidates, as_of = latest_discovery_cohort(limit=top_n)
    tickers = [candidate["ticker"] for candidate in candidates]
    start, end = replay_window(as_of, frequency=frequency, days=days)
    fundamental_refresh_payload = None
    if refresh_fundamentals:
        fundamental_refresh_payload = _refresh_fundamentals_for_parity(
            candidates,
            queue_path=fundamental_queue_path,
            max_tickers=fundamental_max_tickers,
            limit=fundamental_limit,
            sleep_seconds=fundamental_sleep_seconds,
            allow_yfinance_fallback=allow_yfinance_fallback,
        )
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
        "fundamental_refresh": fundamental_refresh_payload,
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
    parser.add_argument(
        "--refresh-fundamentals",
        action="store_true",
        help="FMP/PIT-backfill replay-missing fundamentals before rebuilding replay rows.",
    )
    parser.add_argument(
        "--fundamental-max-tickers",
        type=int,
        default=int(getattr(config, "HISTORICAL_REPLAY_FUNDAMENTAL_REFRESH_MAX_TICKERS", 40)),
        help="Maximum PIT fundamental queue tickers to refresh before replay.",
    )
    parser.add_argument(
        "--fundamental-limit",
        type=int,
        default=int(getattr(config, "PIT_BACKFILL_DEFAULT_QUARTERS", 40)),
        help="Quarterly statement snapshots per ticker for PIT fundamental refresh.",
    )
    parser.add_argument("--fundamental-sleep", type=float, default=0.0, help="Optional pause between FMP tickers.")
    parser.add_argument(
        "--fundamental-queue-path",
        default=None,
        help="Override the replay fundamental refresh queue path.",
    )
    parser.add_argument(
        "--allow-yfinance-fallback",
        action="store_true",
        help="Allow yfinance quarterly fallback for the pre-replay PIT refresh. Off by default.",
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
        refresh_fundamentals=args.refresh_fundamentals,
        fundamental_max_tickers=args.fundamental_max_tickers,
        fundamental_limit=args.fundamental_limit,
        fundamental_sleep_seconds=args.fundamental_sleep,
        allow_yfinance_fallback=args.allow_yfinance_fallback,
        fundamental_queue_path=args.fundamental_queue_path,
    )
    print(json.dumps({
        "requested_tickers": payload["requested_tickers"],
        "fundamental_refresh": payload["fundamental_refresh"],
        "replay_stats": payload["replay_stats"],
        "parity_summary": payload["parity_summary"],
        "report": str(_REFRESH_REPORT),
    }, indent=2))


if __name__ == "__main__":
    main()
