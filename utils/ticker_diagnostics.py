"""Ticker-level discovery diagnostics.

This module answers the practical "why was this not a buy?" question without
running the full screener.  It stitches together universe membership, feature
cache presence, latest discovery rows, gate blockers, TB outcomes, and the
replay-vs-live factor mismatch that can hide good mid-cap names.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import config
from engine.discovery_backtest import init_backtest_db
from engine.ml_ranker import FEATURE_COLS as _ML_FEATURE_COLS
from engine.paper_trading import _connect
from utils.atomic_io import atomic_write_json
from utils.global_universe import get_dynamic_entries, get_full_universe, get_global_universe

ROOT = Path(__file__).parent.parent
FEATURE_CACHE = ROOT / "feature_cache"

_FACTOR_COLUMNS = tuple(_ML_FEATURE_COLS)

_ROW_COLUMNS = (
    "id",
    "ticker",
    "run_date",
    "source",
    "action",
    "aggregate_score",
    "final_rank",
    "value_factor_score",
    "quality_factor_score",
    "qmj_factor_score",
    "momentum_factor_score",
    "volatility_factor_score",
    "pead_factor_score",
    "institutional_prior_score",
    "institutional_prior_percentile",
    "institutional_prior_confidence",
    "institutional_prior_components",
    "ready_contract_status",
    "ready_contract_reasons",
    "strong_buy_blockers",
    "strong_buy_eligible",
    "gate_v2_status",
    "gate_v2_reasons",
    "action_gate_ceiling",
    "action_gate_reasons",
    "threshold_profile",
    "meta_prob",
    "tb_label",
    "tb_return",
    "tb_days",
    "tb_hit",
)


def _norm_ticker(ticker: str) -> str:
    return str(ticker or "").upper().strip()


def _finite(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if not math.isfinite(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def _jsonish(value: Any) -> Any:
    if value is None or isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return value


def _table_columns(conn) -> set[str]:
    return {str(row[1]) for row in conn.execute("PRAGMA table_info(signal_backtest)").fetchall()}


def _row_to_dict(row, columns: list[str]) -> dict:
    out = dict(zip(columns, row))
    for key in ("institutional_prior_components", "ready_contract_reasons", "strong_buy_blockers", "gate_v2_reasons", "action_gate_reasons"):
        if key in out:
            out[key] = _jsonish(out.get(key))
    return out


def _latest_rows(ticker: str, limit: int = 12) -> list[dict]:
    init_backtest_db()
    with _connect() as conn:
        valid = _table_columns(conn)
        columns = [c for c in _ROW_COLUMNS if c in valid]
        rows = conn.execute(
            f"""
            SELECT {', '.join(columns)}
            FROM signal_backtest
            WHERE UPPER(ticker)=?
            ORDER BY run_date DESC, id DESC
            LIMIT ?
            """,
            (ticker, int(limit)),
        ).fetchall()
    return [_row_to_dict(tuple(row), columns) for row in rows]


def _source_summary(ticker: str) -> list[dict]:
    init_backtest_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                source,
                COUNT(*) AS rows,
                MIN(run_date) AS first_run,
                MAX(run_date) AS last_run,
                SUM(CASE WHEN action='STRONG BUY' THEN 1 ELSE 0 END) AS strong_buy_rows,
                SUM(CASE WHEN action='BUY' THEN 1 ELSE 0 END) AS buy_rows
            FROM signal_backtest
            WHERE UPPER(ticker)=?
            GROUP BY source
            ORDER BY rows DESC
            """,
            (ticker,),
        ).fetchall()
    return [
        {
            "source": row[0],
            "rows": row[1],
            "first_run": row[2],
            "last_run": row[3],
            "strong_buy_rows": row[4] or 0,
            "buy_rows": row[5] or 0,
        }
        for row in rows
    ]


def _tb_history(ticker: str) -> list[dict]:
    init_backtest_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT source, tb_label, COUNT(*) AS rows, AVG(tb_return) AS avg_tb_return
            FROM signal_backtest
            WHERE UPPER(ticker)=? AND tb_label IS NOT NULL
            GROUP BY source, tb_label
            ORDER BY source, tb_label
            """,
            (ticker,),
        ).fetchall()
    return [
        {
            "source": row[0],
            "tb_label": row[1],
            "rows": row[2],
            "avg_tb_return": row[3],
        }
        for row in rows
    ]


def _universe_presence(ticker: str) -> dict:
    static_entries = [
        entry._asdict()
        for entry in get_full_universe()
        if entry.ticker.upper() == ticker
    ]
    dynamic_entries = [
        entry
        for entry in get_dynamic_entries()
        if str(entry.get("ticker") or "").upper() == ticker
    ]
    static_global = {t.upper() for t in get_global_universe(include_dynamic=False)}
    full_global = {t.upper() for t in get_global_universe(include_dynamic=True)}
    return {
        "in_static_global_universe": ticker in static_global,
        "in_dynamic_supplement": bool(dynamic_entries),
        "in_effective_global_universe": ticker in full_global,
        "static_entries": static_entries,
        "dynamic_entries": dynamic_entries,
    }


def _feature_cache_presence(ticker: str, max_files: int = 14) -> dict:
    files = sorted(FEATURE_CACHE.glob("features_*.json"), reverse=True)[:max_files]
    history = []
    latest_present_row = None
    latest_file = None
    latest_count = None
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        features = payload.get("features") if isinstance(payload, dict) else None
        if not isinstance(features, dict):
            continue
        count = int(payload.get("ticker_count") or len(features))
        present = ticker in {str(k).upper() for k in features.keys()}
        if latest_file is None:
            latest_file = path.name
            latest_count = count
        row = None
        if present:
            for key, value in features.items():
                if str(key).upper() == ticker:
                    row = value if isinstance(value, dict) else None
                    break
            if latest_present_row is None:
                latest_present_row = row
        history.append({
            "file": path.name,
            "ticker_count": count,
            "present": present,
        })
    row_summary = {}
    if isinstance(latest_present_row, dict):
        for key in (
            "ticker",
            "last_price",
            "ret_30d",
            "ret_90d",
            "beta_90d",
            "avg_dollar_volume",
            "sector",
            "relative_strength",
            "above_sma50",
            "above_sma200",
        ):
            if key in latest_present_row:
                row_summary[key] = latest_present_row.get(key)
        row_summary["available_keys"] = sorted(latest_present_row.keys())
    return {
        "latest_file": latest_file,
        "latest_ticker_count": latest_count,
        "present_in_latest": bool(history and history[0]["present"]),
        "presence_history": history,
        "latest_present_row_summary": row_summary,
    }


def _cached_discovery_presence(ticker: str) -> dict:
    path = ROOT / getattr(config, "ORCHESTRATOR_STATE_FILE", "orchestrator_state.json")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"state_file": str(path), "available": False}
    candidates = state.get("cached_discovery") or []
    for idx, cand in enumerate(candidates, start=1):
        if str(cand.get("ticker") or "").upper() == ticker:
            return {
                "state_file": str(path),
                "available": True,
                "last_discovery_run": state.get("last_discovery_run"),
                "rank": idx,
                "candidate": {
                    key: _jsonish(cand.get(key))
                    for key in (
                        "ticker",
                        "action",
                        "aggregate_score",
                        "final_rank",
                        "institutional_prior_percentile",
                        "institutional_prior_confidence",
                        "institutional_prior_coverage",
                        "meta_label_proba",
                        "ready_contract_status",
                        "ready_contract_reasons",
                        "strong_buy_blockers",
                        "threshold_profile",
                    )
                    if key in cand
                },
            }
    return {
        "state_file": str(path),
        "available": True,
        "last_discovery_run": state.get("last_discovery_run"),
        "rank": None,
        "candidate": None,
        "cached_candidate_count": len(candidates),
    }


def _latest_by_source(rows: list[dict], source: str) -> dict | None:
    for row in rows:
        if str(row.get("source") or "") == source:
            return row
    return None


def _replay_live_parity(rows: list[dict], tolerance: float = 0.15) -> dict:
    live = _latest_by_source(rows, "discovery")
    replay = _latest_by_source(rows, "replay_pit_v1")
    comparisons = []
    if not live or not replay:
        return {
            "available": False,
            "reason": "missing latest discovery row" if not live else "missing latest replay row",
            "latest_discovery": live,
            "latest_replay": replay,
            "comparisons": comparisons,
        }
    max_date_gap_days = int(getattr(config, "REPLAY_LIVE_PARITY_MAX_DATE_GAP_DAYS", 7))
    date_gap_days = None
    try:
        live_date = datetime.fromisoformat(str(live.get("run_date"))[:10])
        replay_date = datetime.fromisoformat(str(replay.get("run_date"))[:10])
        date_gap_days = abs((live_date.date() - replay_date.date()).days)
    except Exception:
        date_gap_days = None
    if date_gap_days is not None and date_gap_days > max_date_gap_days:
        return {
            "available": False,
            "reason": f"stale replay row ({date_gap_days}d gap)",
            "latest_discovery_run": live.get("run_date"),
            "latest_replay_run": replay.get("run_date"),
            "date_gap_days": date_gap_days,
            "max_date_gap_days": max_date_gap_days,
            "comparisons": comparisons,
        }
    for column in _FACTOR_COLUMNS:
        live_val = _finite(live.get(column))
        replay_val = _finite(replay.get(column))
        if live_val is None or replay_val is None:
            comparisons.append({
                "field": column,
                "live": live_val,
                "replay": replay_val,
                "delta": None,
                "status": "missing",
            })
            continue
        delta = live_val - replay_val
        comparisons.append({
            "field": column,
            "live": live_val,
            "replay": replay_val,
            "delta": delta,
            "status": "drift" if abs(delta) > tolerance else "ok",
        })
    return {
        "available": True,
        "tolerance": tolerance,
        "latest_discovery_run": live.get("run_date"),
        "latest_replay_run": replay.get("run_date"),
        "date_gap_days": date_gap_days,
        "drift_fields": [row["field"] for row in comparisons if row["status"] == "drift"],
        "comparisons": comparisons,
    }


def build_ticker_diagnostic(ticker: str, *, row_limit: int = 12) -> dict:
    """Build a JSON-serializable why-not diagnostic for one ticker."""
    symbol = _norm_ticker(ticker)
    rows = _latest_rows(symbol, limit=row_limit)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ticker": symbol,
        "universe": _universe_presence(symbol),
        "feature_cache": _feature_cache_presence(symbol),
        "cached_discovery": _cached_discovery_presence(symbol),
        "source_summary": _source_summary(symbol),
        "latest_rows": rows,
        "tb_history": _tb_history(symbol),
        "replay_live_parity": _replay_live_parity(rows),
    }


def write_ticker_diagnostic(ticker: str, path: str | Path | None = None) -> Path:
    """Write a ticker diagnostic to feature_cache/ticker_diagnostics by default."""
    symbol = _norm_ticker(ticker)
    output = Path(path) if path else FEATURE_CACHE / "ticker_diagnostics" / f"{symbol}_why_not.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, build_ticker_diagnostic(symbol), indent=2)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a ticker why-not diagnostic.")
    parser.add_argument("ticker", help="Ticker to inspect, e.g. MLI")
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument("--write", action="store_true", help="Write to feature_cache/ticker_diagnostics")
    args = parser.parse_args()

    if args.write or args.output:
        path = write_ticker_diagnostic(args.ticker, args.output)
        print(path)
    else:
        print(json.dumps(build_ticker_diagnostic(args.ticker), indent=2, default=str))


if __name__ == "__main__":
    main()
