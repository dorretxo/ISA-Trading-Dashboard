"""Explain why strict replay/live parity is still blocked."""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from engine.coverage_evidence import (
    classify_evidence,
    is_fmp_statement_candidate,
    is_non_us,
    qmj_usable_components,
)
from engine.discovery_backtest import init_backtest_db
from engine.fscore_utils import f_score_min_coverage, normalize_f_score_coverage
from engine.ml_ranker import FEATURE_COLS
from engine.paper_trading import _connect
from utils.atomic_io import atomic_write_json
from utils.global_universe import resolve_yahoo_ticker
from utils.replay_live_parity import comparable_fields, compare_parity_rows, finite_float
from utils.replay_parity_refresh import latest_discovery_cohort


_DEFAULT_OUT = ROOT / "feature_cache" / "parity_blocker_breakdown.json"
_QUEUE_PATH = ROOT / getattr(
    config,
    "HISTORICAL_REPLAY_FUNDAMENTAL_REFRESH_QUEUE_PATH",
    "feature_cache/replay_fundamental_refresh_queue.json",
)
_LEDGER_PATH = ROOT / getattr(
    config,
    "HISTORICAL_REPLAY_FUNDAMENTAL_ATTEMPT_LEDGER_PATH",
    "feature_cache/replay_fundamental_attempt_ledger.json",
)
_QUALITY_PARITY_FIELDS = (
    "quality_factor_score",
    "qmj_factor_score",
    "quality_score_fundamental",
    "gross_profitability",
    "fcf_to_assets",
    "earnings_stability",
    "f_score",
    "f_score_coverage",
    "gpa",
    "gpa_score",
    "ev_ebit_score",
)
_COMPONENT_PARITY_FIELDS = (
    "gpa",
    "f_score_score",
    "fcf_yield",
    "ev_ebit_score",
)
_VALUE_FACTOR_DRILLDOWN_FIELDS = (
    "value_factor_score",
    "pe_ratio",
    "peg_ratio",
    "fcf_yield",
    "ev_ebit_score",
    "pb_score",
    "ps_score",
    "ev_ebit",
    "ebit_yield",
)
_F_SCORE_THRESHOLD_AUDIT_FILES = (
    "daily_orchestrator.py",
    "engine/discovery.py",
    "engine/distress.py",
    "engine/enterprise_factors.py",
    "engine/fundamental.py",
    "engine/institutional_prior.py",
    "engine/scoring.py",
    "engine/sleeves.py",
)


def _finite(value) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _is_fmp_statement_candidate(ticker: str) -> bool:
    return is_fmp_statement_candidate(ticker)


def _is_non_us(ticker: str) -> bool:
    return is_non_us(ticker)


def _parse_flags(value) -> dict:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    text = str(value).strip()
    if not text:
        return {}
    for loader in (json.loads, ast.literal_eval):
        try:
            loaded = loader(text)
        except Exception:
            continue
        if isinstance(loaded, Mapping):
            return dict(loaded)
    return {}


def _region_bucket(ticker: str, candidate: Mapping | None = None, row: Mapping | None = None) -> str:
    candidate = candidate or {}
    row = row or {}
    raw = (
        candidate.get("region")
        or candidate.get("country")
        or candidate.get("_region")
        or row.get("region")
        or row.get("country")
    )
    if raw:
        return str(raw)
    exchange = str(candidate.get("exchange") or row.get("exchange") or "").upper()
    suffix = str(ticker or "").upper().rsplit(".", 1)[-1] if "." in str(ticker or "") else ""
    suffix_map = {
        "L": "UK", "AX": "Australia", "TO": "Canada", "V": "Canada",
        "HK": "Hong Kong", "SW": "Switzerland", "PA": "France",
        "DE": "Germany", "F": "Germany", "MI": "Italy", "MC": "Spain",
        "AS": "Netherlands", "ST": "Nordics", "OL": "Nordics",
        "CO": "Nordics", "HE": "Nordics", "T": "Japan", "SI": "Singapore",
    }
    if suffix in suffix_map:
        return suffix_map[suffix]
    if exchange in {"NYSE", "NASDAQ", "AMEX", "OTC", "NYSEARCA"} or not suffix:
        return "US"
    return "Unknown"


def _source_bucket(ticker: str, candidate: Mapping | None = None, row: Mapping | None = None) -> str:
    candidate = candidate or {}
    row = row or {}
    source = (
        candidate.get("universe_source")
        or candidate.get("_universe_source")
        or candidate.get("candidate_source")
        or candidate.get("_source")
        or row.get("source")
    )
    if source and str(source).lower() not in {"discovery", "replay_pit_v1", "portfolio"}:
        return str(source)
    return "fmp_us" if _is_fmp_statement_candidate(ticker) else "global_non_us"


def _coverage_buckets(live: Mapping | None, replay: Mapping | None, *, config_module=None) -> list[str]:
    buckets: set[str] = set()
    min_cov = f_score_min_coverage(config_module)
    legacy_min = 0.45
    for row in (live, replay):
        if not row:
            continue
        f_score = _finite(row.get("f_score"))
        cov = normalize_f_score_coverage(row.get("f_score_coverage"))
        if cov is None:
            buckets.add("unknown_coverage")
            continue
        if cov == 0.0:
            if f_score is None:
                buckets.add("legacy_zero_coverage")
            else:
                buckets.add("computed_zero")
        elif cov < min_cov:
            buckets.add("low_coverage")
        if f_score is not None and legacy_min <= cov < min_cov:
            buckets.add("threshold_disagreement")
    return sorted(buckets)


def _qmj_lite_component_count(row: Mapping | None, *, config_module=None) -> int:
    return qmj_usable_components(row, config_module=config_module)


def _threshold_audit_sites() -> list[dict]:
    """Static guardrail for old F-score threshold idioms."""
    offenders: list[dict] = []
    patterns = ("0.45", "6.0 / 9.0", "6 / 9")
    for rel in _F_SCORE_THRESHOLD_AUDIT_FILES:
        path = ROOT / rel
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for lineno, line in enumerate(lines, start=1):
            lower = line.lower()
            if "f_score_min_coverage" in lower or "is_f_score_actionable" in lower:
                continue
            if not ("f_score" in lower or "f_cov" in lower or "fs_cov" in lower):
                continue
            if any(pattern in line for pattern in patterns):
                offenders.append({"file": rel, "line": lineno, "text": line.strip()})
    return offenders


def _load_json(path: str | Path) -> dict:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _queue_index(path: str | Path = _QUEUE_PATH) -> dict[str, dict]:
    payload = _load_json(path)
    rows = payload.get("items", [])
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("ticker") or "").upper(): dict(row)
        for row in rows
        if isinstance(row, Mapping) and row.get("ticker")
    }


def _ledger_index(path: str | Path = _LEDGER_PATH) -> dict[str, dict]:
    payload = _load_json(path)
    tickers = payload.get("tickers", {})
    return tickers if isinstance(tickers, dict) else {}


def _latest_pair(conn, ticker: str, select_cols: list[str]) -> tuple[dict | None, dict | None]:
    cols = ", ".join(select_cols)
    live = conn.execute(
        f"""
        SELECT {cols}
        FROM signal_backtest
        WHERE UPPER(ticker)=? AND source='discovery'
        ORDER BY run_date DESC, id DESC
        LIMIT 1
        """,
        (ticker,),
    ).fetchone()
    replay = conn.execute(
        f"""
        SELECT {cols}
        FROM signal_backtest
        WHERE UPPER(ticker)=? AND source='replay_pit_v1'
        ORDER BY run_date DESC, id DESC
        LIMIT 1
        """,
        (ticker,),
    ).fetchone()
    return (dict(live) if live else None, dict(replay) if replay else None)


def _latest_source_row(conn, ticker: str, source: str, select_cols: list[str]) -> dict | None:
    cols = ", ".join(select_cols)
    row = conn.execute(
        f"""
        SELECT {cols}
        FROM signal_backtest
        WHERE UPPER(ticker)=? AND source=?
        ORDER BY run_date DESC, id DESC
        LIMIT 1
        """,
        (ticker, source),
    ).fetchone()
    return dict(row) if row else None


def _field_breakdown(comparisons: list[dict]) -> tuple[list[str], list[str], list[str], list[str]]:
    live_missing: list[str] = []
    replay_missing: list[str] = []
    both_missing: list[str] = []
    drifted: list[str] = []
    for comp in comparisons:
        field = str(comp.get("field") or "")
        if comp.get("status") == "drift":
            drifted.append(field)
        elif comp.get("status") == "missing":
            live_is_missing = comp.get("live") is None
            replay_is_missing = comp.get("replay") is None
            if live_is_missing and replay_is_missing:
                both_missing.append(field)
            elif live_is_missing:
                live_missing.append(field)
            elif replay_is_missing:
                replay_missing.append(field)
    return live_missing, replay_missing, both_missing, drifted


def _comparison_map(comparisons: list[dict]) -> dict[str, dict]:
    return {str(comp.get("field")): dict(comp) for comp in comparisons}


def _value_factor_drilldown_row(
    *,
    ticker: str,
    live: Mapping,
    replay: Mapping,
    fields: list[str],
    evidence_class: str,
    source_bucket: str,
    region: str,
    component_drifted: list[str],
    live_missing: list[str],
    replay_missing: list[str],
    both_missing: list[str],
    default_tolerance: float,
    config_module=None,
) -> dict:
    cfg = config_module or config
    compared = compare_parity_rows(
        live,
        replay,
        fields,
        default_tolerance=default_tolerance,
    )
    return {
        "ticker": ticker,
        "evidence_class": evidence_class,
        "source_bucket": source_bucket,
        "region": region,
        "actionable": evidence_class not in _non_actionable_evidence_classes(cfg),
        "live_pit_source": live.get("pit_source") or live.get("_pit_source"),
        "replay_pit_source": replay.get("pit_source") or replay.get("_pit_source"),
        "component_drift_fields": component_drifted,
        "live_missing_fields": live_missing,
        "replay_missing_fields": replay_missing,
        "both_missing_fields": both_missing,
        "comparisons": _comparison_map(compared["comparisons"]),
    }


def _quality_score_source(row: Mapping | None) -> str:
    if not row:
        return "missing_row"
    quality = finite_float(row.get("quality_score_fundamental"))
    if quality is None:
        return "none"

    component_keys = (
        "gpa",
        "gpa_score",
        "gross_profitability",
        "fcf_to_assets",
        "earnings_stability",
        "roe",
    )
    component_count = sum(1 for key in component_keys if finite_float(row.get(key)) is not None)
    if component_count:
        pit_source = str(row.get("pit_source") or row.get("_pit_source") or "unknown")
        return f"{pit_source}_component_backed_{component_count}"

    if quality == 0.0:
        return "legacy_zero_without_components"
    if abs(quality - 0.85) <= 1e-9:
        return "live_info_fallback_without_components"
    return "opaque_without_components"


def _quality_source_family(source: str) -> str:
    if "_component_backed_" not in source:
        return source
    provider = source.split("_component_backed_", 1)[0]
    if provider.startswith("yfinance"):
        return "yfinance_component_backed"
    if provider in {"fmp", "unknown"}:
        return "pit_component_backed"
    return f"{provider}_component_backed"


def _non_actionable_evidence_classes(config_module=None) -> set[str]:
    cfg = config_module or config
    raw = getattr(
        cfg,
        "REPLAY_LIVE_PARITY_NON_ACTIONABLE_EVIDENCE_CLASSES",
        ["yfinance_balance_only", "no_data"],
    )
    return {str(item) for item in (raw or [])}


def _structural_drift_fields(config_module=None) -> set[str]:
    """Fields whose drift is explained by known live-only structural inputs.

    See `REPLAY_LIVE_PARITY_STRUCTURAL_DRIFT_FIELDS` in config.py.  These are
    stored aggregates the canonicalizer does not recompute, so the per-input
    null-list trick cannot reach them.  Drift on these fields is reported in
    `structural_drift_summary` rather than `actionable_drift_summary`.
    """
    cfg = config_module or config
    raw = getattr(cfg, "REPLAY_LIVE_PARITY_STRUCTURAL_DRIFT_FIELDS", [])
    return {str(item) for item in (raw or [])}


def _actionable_drift(rows: list[dict], *, config_module=None) -> dict[str, int]:
    non_actionable = _non_actionable_evidence_classes(config_module)
    structural = _structural_drift_fields(config_module)
    out: dict[str, int] = {}
    for row in rows:
        field = str(row.get("field"))
        if field in structural:
            # Drift on this field is known-structural; routed to
            # structural_drift_summary, never to actionable.
            continue
        by_evidence = row.get("drifted_by_evidence", {}) or {}
        actionable = sum(
            int(count)
            for evidence_class, count in by_evidence.items()
            if str(evidence_class) not in non_actionable
        )
        if actionable:
            out[field] = actionable
    return dict(sorted(out.items(), key=lambda item: item[1], reverse=True))


def _structural_drift(rows: list[dict], *, config_module=None) -> dict[str, int]:
    """Drift counts on `REPLAY_LIVE_PARITY_STRUCTURAL_DRIFT_FIELDS` only.

    Returns the same evidence-class-filtered count shape as `_actionable_drift`
    but restricted to fields tagged as structural.  Lets the operator see
    *how much* known-structural drift exists without it polluting the
    actionable gate signal.
    """
    non_actionable = _non_actionable_evidence_classes(config_module)
    structural = _structural_drift_fields(config_module)
    out: dict[str, int] = {}
    for row in rows:
        field = str(row.get("field"))
        if field not in structural:
            continue
        by_evidence = row.get("drifted_by_evidence", {}) or {}
        count = sum(
            int(c)
            for evidence_class, c in by_evidence.items()
            if str(evidence_class) not in non_actionable
        )
        if count:
            out[field] = count
    return dict(sorted(out.items(), key=lambda item: item[1], reverse=True))


def _priority(row: dict, *, critical_fields: set[str]) -> float:
    replay_missing = set(row.get("replay_missing_fields") or [])
    drifted = set(row.get("drift_fields") or [])
    score = 0.0
    score += 3.0 * len(replay_missing & critical_fields)
    score += 1.5 * len(drifted & critical_fields)
    score += len(replay_missing - critical_fields)
    score += 0.5 * len(drifted - critical_fields)
    if row.get("fmp_statement_candidate"):
        score += 1.0
    if row.get("recent_unresolved_attempt"):
        score -= 2.0
    if row.get("non_us"):
        score -= 0.5
    return round(score, 3)


def _increment_segment(summary: dict[str, dict], key: str, *, blocked: bool, buckets: list[str]) -> None:
    item = summary.setdefault(key or "Unknown", {"sample": 0, "blocked": 0, "buckets": Counter()})
    item["sample"] += 1
    if blocked:
        item["blocked"] += 1
    item["buckets"].update(buckets)


def _serialise_segment_summary(summary: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key, item in sorted(summary.items()):
        out[key] = {
            "sample": item.get("sample", 0),
            "blocked": item.get("blocked", 0),
            "buckets": dict(item.get("buckets", {})),
        }
    return out


def _build_holdings_discovery_parity(conn, *, tolerance: float, valid: set[str]) -> dict:
    fields = comparable_fields(_QUALITY_PARITY_FIELDS, valid)
    if not fields:
        return {"available": False, "reason": "no comparable quality fields"}
    select_cols = ["ticker", "run_date", "source"] + fields
    portfolio_tickers = [
        str(row[0] or "").upper()
        for row in conn.execute(
            "SELECT DISTINCT UPPER(ticker) FROM signal_backtest WHERE source='portfolio'"
        ).fetchall()
        if row[0]
    ]
    rows: list[dict] = []
    drift_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    for ticker in sorted(set(portfolio_tickers)):
        portfolio = _latest_source_row(conn, ticker, "portfolio", select_cols)
        discovery = _latest_source_row(conn, ticker, "discovery", select_cols)
        if not portfolio or not discovery:
            rows.append({
                "ticker": ticker,
                "available": False,
                "reason": "missing portfolio row" if not portfolio else "missing discovery row",
            })
            continue
        compared = compare_parity_rows(
            portfolio,
            discovery,
            fields,
            default_tolerance=tolerance,
        )
        for field in compared["drift_fields"]:
            drift_counts[field] += 1
        for field in set(compared["live_missing_fields"]) | set(compared["replay_missing_fields"]):
            missing_counts[field] += 1
        rows.append({
            "ticker": ticker,
            "available": True,
            "latest_portfolio_run": portfolio.get("run_date"),
            "latest_discovery_run": discovery.get("run_date"),
            "drift_fields": compared["drift_fields"],
            "portfolio_missing_fields": compared["live_missing_fields"],
            "discovery_missing_fields": compared["replay_missing_fields"],
            "comparisons": compared["comparisons"],
        })
    available_rows = [row for row in rows if row.get("available")]
    drifted = [row for row in available_rows if row.get("drift_fields")]
    return {
        "available": True,
        "comparison_scope": "portfolio_vs_discovery_quality_fields",
        "fields": fields,
        "sample": len(portfolio_tickers),
        "available_pairs": len(available_rows),
        "missing_pairs": len(rows) - len(available_rows),
        "drifted_tickers": len(drifted),
        "drift_field_counts": dict(drift_counts),
        "missing_field_counts": dict(missing_counts),
        "top_drifted": sorted(drifted, key=lambda row: len(row.get("drift_fields") or []), reverse=True)[:40],
    }


def build_blocker_breakdown(*, top_n: int | None = None) -> dict:
    candidates, as_of = latest_discovery_cohort(limit=top_n)
    candidate_by_ticker: dict[str, dict] = {}
    tickers: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        ticker = resolve_yahoo_ticker(str(candidate.get("ticker") or "").upper().strip())
        if ticker and ticker not in seen:
            tickers.append(ticker)
            seen.add(ticker)
            candidate_by_ticker[ticker] = dict(candidate)

    init_backtest_db()
    queue = _queue_index()
    ledger = _ledger_index()
    critical_fields = set(getattr(config, "ML_RANKER_PARITY_CRITICAL_FIELDS", []) or [])

    ticker_rows: list[dict] = []
    value_factor_drift_rows: list[dict] = []
    field_summary: dict[str, Counter] = defaultdict(Counter)
    field_drift_by_evidence: dict[str, Counter] = defaultdict(Counter)
    component_summary: dict[str, Counter] = defaultdict(Counter)
    component_drift_by_evidence: dict[str, Counter] = defaultdict(Counter)
    category_counts: Counter = Counter()
    bucket_counts: Counter = Counter()
    evidence_class_counts: Counter = Counter()
    quality_source_counts: Counter = Counter()
    region_summary: dict[str, dict] = {}
    source_summary: dict[str, dict] = {}
    threshold_audit = _threshold_audit_sites()
    with _connect() as conn:
        valid = {str(row[1]) for row in conn.execute("PRAGMA table_info(signal_backtest)").fetchall()}
        fields = comparable_fields(FEATURE_COLS, valid)
        extra_fields = {
            "id", "ticker", "run_date", "source", "action", "exchange", "sector",
            "action_gate_ceiling", "action_gate_flags_json",
            "f_score", "f_score_coverage", "gpa", "gpa_score", "gross_profitability",
            "earnings_stability", "qmj_factor_score", "qmj_component_count",
            "quality_score_fundamental", "fcf_yield", "ev_ebit", "ev_ebit_score",
            "ebit_yield", "fcf_to_assets", "peg_ratio", "pb_score", "ps_score",
            "roe", "pit_source", "_pit_source",
        }
        select_fields = sorted((set(fields) | extra_fields) & valid)
        # Drilldown is explanatory, so include value inputs even when they are
        # excluded from headline parity counts (for example peg_ratio).
        value_drilldown_fields = [field for field in _VALUE_FACTOR_DRILLDOWN_FIELDS if field in valid]
        holdings_discovery_parity = _build_holdings_discovery_parity(
            conn,
            tolerance=float(getattr(config, "REPLAY_LIVE_PARITY_TOLERANCE", 0.15)),
            valid=valid,
        )
        for ticker in tickers:
            live, replay = _latest_pair(conn, ticker, select_fields)
            candidate = candidate_by_ticker.get(ticker, {})
            fmp_candidate = _is_fmp_statement_candidate(ticker)
            non_us = _is_non_us(ticker)
            region = _region_bucket(ticker, candidate, live)
            source_bucket = _source_bucket(ticker, candidate, live)
            queue_item = queue.get(ticker, {})
            ledger_item = ledger.get(ticker, {})
            recent_unresolved = bool(ledger_item) and not bool(ledger_item.get("last_resolved"))
            candidate_qmj_components = candidate.get("qmj_component_count")
            if candidate_qmj_components is None:
                candidate_qmj_components = qmj_usable_components(candidate, config_module=config)
            candidate_pit_source = (
                candidate.get("pit_source")
                or candidate.get("_pit_source")
                or (live or {}).get("pit_source")
                or (live or {}).get("_pit_source")
            )
            evidence_class = classify_evidence(
                ticker=ticker,
                pit_source=candidate_pit_source,
                qmj_components=int(candidate_qmj_components or 0),
            )

            if not live or not replay:
                buckets = [evidence_class]
                evidence_class_counts[evidence_class] += 1
                bucket_counts.update(buckets)
                for bucket in buckets:
                    category_counts[bucket] += 1
                row = {
                    "ticker": ticker,
                    "paired": False,
                    "reason": "missing live discovery row" if not live else "missing replay_pit_v1 row",
                    "region": region,
                    "source_bucket": source_bucket,
                    "evidence_class": evidence_class,
                    "blocker_buckets": buckets,
                    "fmp_statement_candidate": fmp_candidate,
                    "non_us": non_us,
                    "queued": bool(queue_item),
                    "queue_priority": queue_item.get("priority"),
                    "recent_unresolved_attempt": recent_unresolved,
                    "last_attempted_at": ledger_item.get("last_attempted_at"),
                    "last_snapshots_written": ledger_item.get("last_snapshots_written"),
                }
                category_counts[row["reason"]] += 1
                _increment_segment(region_summary, region, blocked=True, buckets=buckets)
                _increment_segment(source_summary, source_bucket, blocked=True, buckets=buckets)
                ticker_rows.append(row)
                continue

            compared = compare_parity_rows(
                live,
                replay,
                fields,
                default_tolerance=float(getattr(config, "REPLAY_LIVE_PARITY_TOLERANCE", 0.15)),
            )
            live_missing, replay_missing, both_missing, drifted = _field_breakdown(compared["comparisons"])
            component_compared = compare_parity_rows(
                live,
                replay,
                _COMPONENT_PARITY_FIELDS,
                default_tolerance=float(getattr(config, "REPLAY_LIVE_PARITY_TOLERANCE", 0.15)),
            )
            (
                component_live_missing,
                component_replay_missing,
                component_both_missing,
                component_drifted,
            ) = _field_breakdown(component_compared["comparisons"])
            live_qmj_components = _qmj_lite_component_count(live, config_module=config)
            replay_qmj_components = _qmj_lite_component_count(replay, config_module=config)
            row_pit_source = candidate_pit_source or live.get("pit_source") or live.get("_pit_source")
            evidence_class = classify_evidence(
                ticker=ticker,
                pit_source=row_pit_source,
                qmj_components=live_qmj_components,
            )
            value_drilldown = {}
            if "value_factor_score" in drifted and value_drilldown_fields:
                drilldown_row = _value_factor_drilldown_row(
                    ticker=ticker,
                    live=live,
                    replay=replay,
                    fields=value_drilldown_fields,
                    evidence_class=evidence_class,
                    source_bucket=source_bucket,
                    region=region,
                    component_drifted=component_drifted,
                    live_missing=live_missing,
                    replay_missing=replay_missing,
                    both_missing=both_missing,
                    default_tolerance=float(getattr(config, "REPLAY_LIVE_PARITY_TOLERANCE", 0.15)),
                )
                value_drilldown = drilldown_row["comparisons"]
                value_factor_drift_rows.append(drilldown_row)
            for field in live_missing:
                field_summary[field]["live_missing"] += 1
            for field in replay_missing:
                field_summary[field]["replay_missing"] += 1
            for field in both_missing:
                field_summary[field]["both_missing"] += 1
            for field in drifted:
                field_summary[field]["drifted"] += 1
                field_drift_by_evidence[field][evidence_class] += 1
            for field in component_live_missing:
                component_summary[field]["live_missing"] += 1
            for field in component_replay_missing:
                component_summary[field]["replay_missing"] += 1
            for field in component_both_missing:
                component_summary[field]["both_missing"] += 1
            for field in component_drifted:
                component_summary[field]["drifted"] += 1
                component_drift_by_evidence[field][evidence_class] += 1

            blockers = set(live_missing) | set(replay_missing) | set(both_missing) | set(drifted)
            if blockers:
                category_counts["blocked"] += 1
            else:
                category_counts["clean"] += 1
            if replay_missing:
                category_counts["replay_missing"] += 1
            if live_missing:
                category_counts["live_missing"] += 1
            if both_missing:
                category_counts["both_missing"] += 1
            if drifted:
                category_counts["drifted"] += 1
            if blockers and fmp_candidate:
                category_counts["blocked_fmp_candidate"] += 1
            if blockers and non_us:
                category_counts["blocked_non_us"] += 1
            if blockers and recent_unresolved:
                category_counts["blocked_recent_unresolved"] += 1

            buckets = _coverage_buckets(live, replay, config_module=config)
            flags = _parse_flags((candidate or {}).get("action_gate_flags") or live.get("action_gate_flags_json"))
            if flags.get("fundamental_coverage") == "fail":
                buckets.append("fundamental_coverage_fail")
            if live_qmj_components < int(getattr(config, "QMJ_LITE_MIN_COMPONENTS", 2)):
                buckets.append("qmj_component_count_lt_2")
            if replay_qmj_components < int(getattr(config, "QMJ_LITE_MIN_COMPONENTS", 2)):
                buckets.append("replay_qmj_component_count_lt_2")
            live_quality_source = _quality_score_source(live)
            replay_quality_source = _quality_score_source(replay)
            source_pair = f"{live_quality_source} -> {replay_quality_source}"
            quality_source_counts[source_pair] += 1
            if _quality_source_family(live_quality_source) != _quality_source_family(replay_quality_source):
                buckets.append("quality_source_drift")
            evidence_class_counts[evidence_class] += 1
            buckets.append(evidence_class)
            buckets = sorted(set(buckets))
            bucket_counts.update(buckets)
            for bucket in buckets:
                category_counts[bucket] += 1
            _increment_segment(region_summary, region, blocked=bool(blockers), buckets=buckets)
            _increment_segment(source_summary, source_bucket, blocked=bool(blockers), buckets=buckets)

            row = {
                "ticker": ticker,
                "paired": True,
                "latest_discovery_run": live.get("run_date"),
                "latest_replay_run": replay.get("run_date"),
                "region": region,
                "source_bucket": source_bucket,
                "evidence_class": evidence_class,
                "blocker_buckets": buckets,
                "live_qmj_component_count": live_qmj_components,
                "replay_qmj_component_count": replay_qmj_components,
                "component_live_missing_fields": component_live_missing,
                "component_replay_missing_fields": component_replay_missing,
                "component_both_missing_fields": component_both_missing,
                "component_drift_fields": component_drifted,
                "value_factor_drilldown": value_drilldown,
                "quality_score_fundamental_source": {
                    "live": live_quality_source,
                    "replay": replay_quality_source,
                },
                "fmp_statement_candidate": fmp_candidate,
                "non_us": non_us,
                "queued": bool(queue_item),
                "queue_priority": queue_item.get("priority"),
                "queue_missing_fields": queue_item.get("missing_fields", []),
                "recent_unresolved_attempt": recent_unresolved,
                "last_attempted_at": ledger_item.get("last_attempted_at"),
                "last_snapshots_written": ledger_item.get("last_snapshots_written"),
                "last_missing_fields": ledger_item.get("last_missing_fields", []),
                "live_missing_fields": live_missing,
                "replay_missing_fields": replay_missing,
                "both_missing_fields": both_missing,
                "drift_fields": drifted,
                "critical_blockers": sorted(blockers & critical_fields),
            }
            row["priority"] = _priority(row, critical_fields=critical_fields)
            ticker_rows.append(row)

    field_rows = []
    for field, counts in field_summary.items():
        total = sum(counts.values())
        field_rows.append({
            "field": field,
            "total": total,
            "live_missing": counts.get("live_missing", 0),
            "replay_missing": counts.get("replay_missing", 0),
            "both_missing": counts.get("both_missing", 0),
            "drifted": counts.get("drifted", 0),
            "drifted_by_evidence": dict(field_drift_by_evidence.get(field, {})),
            "critical": field in critical_fields,
        })
    field_rows.sort(key=lambda row: (row["critical"], row["total"]), reverse=True)

    component_rows = []
    for field, counts in component_summary.items():
        total = sum(counts.values())
        component_rows.append({
            "field": field,
            "total": total,
            "live_missing": counts.get("live_missing", 0),
            "replay_missing": counts.get("replay_missing", 0),
            "both_missing": counts.get("both_missing", 0),
            "drifted": counts.get("drifted", 0),
            "drifted_by_evidence": dict(component_drift_by_evidence.get(field, {})),
        })
    component_rows.sort(key=lambda row: row["total"], reverse=True)

    non_actionable = _non_actionable_evidence_classes(config)
    structural_fields = _structural_drift_fields(config)
    actionable_drift_summary = {
        "field": _actionable_drift(field_rows, config_module=config),
        "component": _actionable_drift(component_rows, config_module=config),
        "non_actionable_classes": sorted(non_actionable),
        "structural_fields_excluded": sorted(structural_fields),
    }
    structural_drift_summary = {
        "field": _structural_drift(field_rows, config_module=config),
        "component": _structural_drift(component_rows, config_module=config),
        "structural_fields": sorted(structural_fields),
    }

    fmp_targets = [
        row for row in ticker_rows
        if row.get("paired")
        and row.get("fmp_statement_candidate")
        and not row.get("recent_unresolved_attempt")
        and (row.get("replay_missing_fields") or row.get("critical_blockers"))
    ]
    fmp_targets.sort(key=lambda row: row.get("priority", 0), reverse=True)
    non_us_blocked = [
        row for row in ticker_rows
        if row.get("paired") and row.get("non_us") and (
            row.get("replay_missing_fields") or row.get("drift_fields")
        )
    ]
    non_us_blocked.sort(key=lambda row: len(row.get("replay_missing_fields") or []) + len(row.get("drift_fields") or []), reverse=True)
    value_factor_drift_rows.sort(
        key=lambda row: (
            not bool(row.get("actionable")),
            str(row.get("evidence_class") or ""),
            str(row.get("ticker") or ""),
        )
    )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of": as_of.isoformat(),
        "sample": len(tickers),
        "categories": dict(category_counts),
        "blocker_buckets": dict(bucket_counts),
        "evidence_class_summary": dict(evidence_class_counts),
        "region_summary": _serialise_segment_summary(region_summary),
        "source_summary": _serialise_segment_summary(source_summary),
        "threshold_audit": {
            "remaining_old_threshold_sites": len(threshold_audit),
            "sites": threshold_audit,
        },
        "holdings_discovery_parity": holdings_discovery_parity if "holdings_discovery_parity" in locals() else {},
        "critical_fields": sorted(critical_fields),
        "field_summary": field_rows,
        "component_summary": component_rows,
        "actionable_drift_summary": actionable_drift_summary,
        "structural_drift_summary": structural_drift_summary,
        "value_factor_drift_drilldown": value_factor_drift_rows[:80],
        "quality_score_fundamental_source_summary": dict(quality_source_counts),
        "fmp_targets": fmp_targets[:80],
        "non_us_blocked": non_us_blocked[:80],
        "ticker_rows": sorted(ticker_rows, key=lambda row: row.get("priority", 0), reverse=True),
    }


def write_blocker_breakdown(*, top_n: int | None = None, path: str | Path | None = None) -> dict:
    payload = build_blocker_breakdown(top_n=top_n)
    out = Path(path or _DEFAULT_OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, payload, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a replay/live parity blocker breakdown report.")
    parser.add_argument("--top-n", type=int, default=int(getattr(config, "HISTORICAL_REPLAY_FRESH_DEFAULT_TOP_N", 250)))
    parser.add_argument("--path", default=str(_DEFAULT_OUT))
    args = parser.parse_args()
    payload = write_blocker_breakdown(top_n=args.top_n, path=args.path)
    print(json.dumps({
        "path": args.path,
        "sample": payload.get("sample"),
        "categories": payload.get("categories"),
        "evidence_class_summary": payload.get("evidence_class_summary"),
        "top_fields": payload.get("field_summary", [])[:10],
        "component_summary": payload.get("component_summary", [])[:10],
        "actionable_drift_summary": payload.get("actionable_drift_summary"),
        "structural_drift_summary": payload.get("structural_drift_summary"),
        "value_factor_drift_drilldown": payload.get("value_factor_drift_drilldown", [])[:10],
        "quality_score_fundamental_source_summary": payload.get("quality_score_fundamental_source_summary"),
        "top_fmp_targets": [
            {
                "ticker": row.get("ticker"),
                "priority": row.get("priority"),
                "replay_missing_fields": row.get("replay_missing_fields"),
                "critical_blockers": row.get("critical_blockers"),
            }
            for row in payload.get("fmp_targets", [])[:10]
        ],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
