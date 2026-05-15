#!/usr/bin/env python3
"""ISA Portfolio Autonomous Orchestrator (v4.0).

Headless script designed to run via Windows Task Scheduler or cron.
Evaluates the portfolio, optionally runs the Global Discovery Engine,
and emails an actionable brief ONLY when strict quantitative hurdles are met.

Usage:
    python daily_orchestrator.py                     # Full run (portfolio + weekly discovery)
    python daily_orchestrator.py --dry-run            # Same but no email sent
    python daily_orchestrator.py --portfolio-only     # Skip discovery entirely
    python daily_orchestrator.py --force-discovery    # Force discovery even if not scheduled
    python daily_orchestrator.py --dry-run --portfolio-only  # Quick test
"""

import argparse
import json
import logging
import os
import socket
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from engine.fscore_utils import is_f_score_actionable, normalize_f_score_coverage
from utils.data_fetch import load_portfolio
from utils.state_manager import (
    get_cached_discovery,
    is_on_cooldown,
    load_state,
    prune_expired_cooldowns,
    save_state,
    set_cooldown,
    should_run_discovery,
)
from utils.email_sender import build_alert_email, send_email
from utils.atomic_io import atomic_write_json
from utils.discovery_digest import candidate_entry_ready
from engine.paper_trading import init_db as _init_paper_db, log_signal as _log_paper_signal, resolve_pending_signals
from engine.discovery_backtest import (
    record_discovery_picks,
    record_portfolio_signals,
    record_borderline_signals,
    evaluate_matured_signals,
)
from engine.exit_engine import reconcile_actions_with_exits, exit_signal_to_dict

# ---------------------------------------------------------------------------
# Logging setup — console + file
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"


def _setup_logging(dry_run: bool = False) -> None:
    """Configure root logger to output to console and a rotating log."""
    level = logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=[
        logging.StreamHandler(sys.stdout),
    ])
    # Suppress noisy third-party loggers
    for name in ("urllib3", "yfinance", "filelock"):
        logging.getLogger(name).setLevel(logging.WARNING)


logger = logging.getLogger("orchestrator")
_STRONG_BUY_BLOCKER_REPORT = ROOT / "feature_cache" / "strong_buy_blockers.json"
_FUNDAMENTAL_REFRESH_QUEUE = ROOT / "feature_cache" / "fundamental_refresh_queue.json"
_THRESHOLD_LEARNER_UPDATE_REPORT = ROOT / getattr(
    config,
    "THRESHOLD_LEARNER_UPDATE_REPORT_PATH",
    "feature_cache/threshold_learner_update_report.json",
)
_READY_CONTRACT_CALIBRATION_REPORT = ROOT / getattr(
    config,
    "READY_STRONG_BUY_CALIBRATION_REPORT_PATH",
    "feature_cache/ready_contract_calibration.json",
)
_FINAL_STRONG_BUY_VETO_REPORT = ROOT / getattr(
    config,
    "READY_STRONG_BUY_VETO_REPORT_PATH",
    "feature_cache/final_strong_buy_veto_report.json",
)
_LIVE_RUN_SANITY_REPORT = ROOT / getattr(
    config,
    "DISCOVERY_LIVE_SANITY_REPORT_PATH",
    "feature_cache/live_run_sanity.json",
)
_REPLAY_LIVE_PARITY_REPORT = ROOT / getattr(
    config,
    "REPLAY_LIVE_PARITY_REPORT_PATH",
    "feature_cache/replay_live_parity_report.json",
)


def _paper_trading_enabled_for_run(dry_run: bool) -> bool:
    """Return True when this run may write to the paper-trading ledger."""
    if not getattr(config, "PAPER_TRADING_ENABLED", False):
        return False
    return (not dry_run) or bool(getattr(config, "PAPER_TRADING_LOG_DRY_RUN", False))


def _listish(value) -> list:
    """Return a compact list for JSON/list/string fields."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [value] if value else []
    return [value]


def _finite_float(value):
    try:
        out = float(value)
        return out if out == out and abs(out) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[idx]


def _write_strong_buy_blocker_report(candidates: list[dict]) -> None:
    """Persist the latest STRONG BUY rejection anatomy for audit/debugging."""
    meta_values = [
        v for v in (_finite_float(c.get("meta_label_proba")) for c in candidates)
        if v is not None
    ]
    try:
        from engine.auto_tune import get_meta_strong_buy_min_prob
        threshold = float(get_meta_strong_buy_min_prob(candidates))
    except Exception:
        threshold = float(getattr(config, "META_LABEL_STRONG_BUY_MIN_PROB", 0.60))
    blocker_counts: Counter[str] = Counter()
    gate_reason_counts: Counter[str] = Counter()
    action_gate_reason_counts: Counter[str] = Counter()
    action_gate_flag_fail_counts: Counter[str] = Counter()
    ready_status_counts: Counter[str] = Counter()
    gate_status_counts: Counter[str] = Counter()
    action_gate_ceiling_counts: Counter[str] = Counter()
    threshold_profile_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()

    top_blocked: list[dict] = []
    for c in candidates:
        action_counts[str(c.get("action") or "UNKNOWN")] += 1
        ready_status_counts[str(c.get("ready_contract_status") or "UNKNOWN")] += 1
        gate_status_counts[str(c.get("gate_v2_status") or "UNKNOWN")] += 1
        action_gate_ceiling_counts[str(c.get("action_gate_ceiling") or "UNKNOWN")] += 1
        threshold_profile_counts[str(c.get("threshold_profile") or "UNKNOWN")] += 1
        blockers = _listish(c.get("strong_buy_blockers")) or _listish(c.get("ready_contract_reasons"))
        gate_reasons = _listish(c.get("gate_v2_reasons"))
        action_gate_reasons = _listish(c.get("action_gate_reasons"))
        action_gate_flags = c.get("action_gate_flags") or {}
        for reason in blockers:
            blocker_counts[str(reason)] += 1
        for reason in gate_reasons:
            gate_reason_counts[str(reason)] += 1
        for reason in action_gate_reasons:
            action_gate_reason_counts[str(reason)] += 1
        if isinstance(action_gate_flags, dict):
            for key, value in action_gate_flags.items():
                if str(value).lower() == "fail":
                    action_gate_flag_fail_counts[str(key)] += 1
        if blockers and len(top_blocked) < 25:
            top_blocked.append({
                "ticker": c.get("ticker"),
                "action": c.get("action"),
                "final_rank": c.get("final_rank"),
                "aggregate_score": c.get("aggregate_score"),
                "meta_label_proba": c.get("meta_label_proba"),
                "strong_buy_eligible": c.get("strong_buy_eligible"),
                "ready_contract_status": c.get("ready_contract_status"),
                "gate_v2_status": c.get("gate_v2_status"),
                "action_gate_ceiling": c.get("action_gate_ceiling"),
                "threshold_profile": c.get("threshold_profile"),
                "blockers": [str(x) for x in blockers[:8]],
                "gate_v2_reasons": [str(x) for x in gate_reasons[:8]],
                "action_gate_reasons": [str(x) for x in action_gate_reasons[:8]],
            })

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_candidates": len(candidates),
        "strong_buy_eligible_count": sum(1 for c in candidates if bool(c.get("strong_buy_eligible"))),
        "action_counts": dict(action_counts),
        "ready_contract_status_counts": dict(ready_status_counts),
        "gate_v2_status_counts": dict(gate_status_counts),
        "action_gate_ceiling_counts": dict(action_gate_ceiling_counts),
        "threshold_profile_counts": dict(threshold_profile_counts),
        "meta_label_proba": {
            "count": len(meta_values),
            "threshold": threshold,
            "static_threshold": float(getattr(config, "META_LABEL_STRONG_BUY_MIN_PROB", 0.60)),
            "min": min(meta_values) if meta_values else None,
            "p50": _percentile(meta_values, 0.50),
            "p90": _percentile(meta_values, 0.90),
            "p95": _percentile(meta_values, 0.95),
            "max": max(meta_values) if meta_values else None,
            "count_at_or_above_threshold": sum(1 for v in meta_values if v >= threshold),
        },
        "top_blockers": blocker_counts.most_common(40),
        "top_gate_v2_reasons": gate_reason_counts.most_common(40),
        "top_action_gate_reasons": action_gate_reason_counts.most_common(40),
        "action_gate_fail_counts": dict(action_gate_flag_fail_counts),
        "top_blocked_candidates": top_blocked,
    }
    atomic_write_json(_STRONG_BUY_BLOCKER_REPORT, report, indent=2)


def _write_ready_contract_calibration_report(candidates: list[dict]) -> None:
    """Persist a score-based view of candidates closest to the ready contract."""
    min_near_score = float(getattr(config, "READY_STRONG_BUY_NEAR_READY_MIN_SCORE", 0.70))
    min_rr = float(getattr(config, "READY_STRONG_BUY_MIN_RR", 1.50))
    min_conf = float(getattr(config, "READY_STRONG_BUY_MIN_CONFIDENCE", 0.70))
    ready_rows: list[dict] = []
    blocker_counts: Counter[str] = Counter()
    soft_pass_counts: Counter[str] = Counter()

    for candidate in candidates:
        core_blockers = _listish(candidate.get("ready_contract_core_reasons"))
        blockers = core_blockers or _listish(candidate.get("ready_contract_reasons"))
        final_blockers = _listish(candidate.get("strong_buy_blockers")) or _listish(candidate.get("ready_contract_reasons"))
        soft_passes = _listish(candidate.get("ready_contract_soft_passes"))
        for blocker in blockers:
            blocker_counts[str(blocker)] += 1
        for soft_pass in soft_passes:
            soft_pass_counts[str(soft_pass)] += 1

        score = _finite_float(candidate.get("ready_contract_score"))
        if score is None:
            rr = _finite_float(candidate.get("r_r_ratio")) or 0.0
            confidence = _finite_float(candidate.get("effective_data_confidence")) or 0.0
            prior_pct = _finite_float(candidate.get("institutional_prior_percentile")) or 0.0
            score = min(1.0, 0.35 * min(1.0, rr / max(min_rr, 1e-9)) + 0.30 * min(1.0, confidence / max(min_conf, 1e-9)) + 0.35 * prior_pct)

        if score < min_near_score and candidate.get("ready_contract_status") != "PASS":
            continue

        ready_rows.append({
            "ticker": candidate.get("ticker"),
            "action": candidate.get("action"),
            "final_rank": candidate.get("final_rank"),
            "aggregate_score": candidate.get("aggregate_score"),
            "ready_contract_score": round(float(score), 4),
            "ready_contract_core_status": candidate.get("ready_contract_core_status"),
            "ready_contract_core_reasons": [str(x) for x in core_blockers[:8]],
            "ready_contract_status": candidate.get("ready_contract_status"),
            "entry_stance": candidate.get("entry_stance"),
            "r_r_ratio": candidate.get("r_r_ratio"),
            "effective_data_confidence": candidate.get("effective_data_confidence"),
            "meta_label_proba": candidate.get("meta_label_proba"),
            "institutional_prior_percentile": candidate.get("institutional_prior_percentile"),
            "institutional_prior_confidence": candidate.get("institutional_prior_confidence"),
            "institutional_prior_coverage": candidate.get("institutional_prior_coverage"),
            "ready_contract_reasons": [str(x) for x in final_blockers[:8]],
            "ready_contract_soft_passes": [str(x) for x in soft_passes[:6]],
            "action_gate_ceiling": candidate.get("action_gate_ceiling"),
            "action_gate_reasons": [str(x) for x in _listish(candidate.get("action_gate_reasons"))[:6]],
        })

    ready_rows.sort(
        key=lambda row: (
            row.get("ready_contract_status") == "PASS",
            float(row.get("ready_contract_score") or 0.0),
            float(row.get("final_rank") or 0.0),
        ),
        reverse=True,
    )
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_candidates": len(candidates),
        "thresholds": {
            "near_ready_min_score": min_near_score,
            "min_rr": min_rr,
            "min_confidence": min_conf,
            "prior_soft_percentile": float(getattr(config, "READY_STRONG_BUY_PRIOR_SOFT_PERCENTILE", 0.80)),
            "prior_soft_confidence": float(getattr(config, "READY_STRONG_BUY_PRIOR_SOFT_CONFIDENCE", 0.55)),
            "prior_soft_coverage": float(getattr(config, "READY_STRONG_BUY_PRIOR_SOFT_COVERAGE", 0.40)),
        },
        "ready_contract_core_status_counts": dict(Counter(str(c.get("ready_contract_core_status") or "UNKNOWN") for c in candidates)),
        "ready_contract_status_counts": dict(Counter(str(c.get("ready_contract_status") or "UNKNOWN") for c in candidates)),
        "top_blockers": blocker_counts.most_common(40),
        "soft_pass_counts": soft_pass_counts.most_common(20),
        "near_ready_candidates": ready_rows[:40],
        "historical_matured_calibration": _historical_ready_contract_calibration(),
    }
    atomic_write_json(_READY_CONTRACT_CALIBRATION_REPORT, payload, indent=2)


def _write_final_strong_buy_veto_report(candidates: list[dict]) -> None:
    """Explain which final layer vetoed candidates that passed core readiness."""
    try:
        from engine.auto_tune import get_core_ready_meta_strong_buy_min_prob
        core_meta_threshold = float(get_core_ready_meta_strong_buy_min_prob(candidates))
    except Exception:
        core_meta_threshold = float(getattr(config, "META_LABEL_STRONG_BUY_MIN_PROB", 0.60))

    rows: list[dict] = []
    layer_counts: Counter[str] = Counter()
    for candidate in candidates:
        if str(candidate.get("ready_contract_core_status") or "").upper() != "PASS":
            continue
        layers: list[str] = []
        meta = _finite_float(candidate.get("meta_label_proba"))
        if meta is not None and meta < core_meta_threshold:
            layers.append("meta_gate")
        ceiling = str(candidate.get("action_gate_ceiling") or "STRONG BUY")
        if ceiling != "STRONG BUY":
            layers.append("action_gate")
        if candidate.get("action") != "STRONG BUY" and not layers:
            layers.append("percentile_or_score_floor")
        for layer in layers:
            layer_counts[layer] += 1
        rows.append({
            "ticker": candidate.get("ticker"),
            "action": candidate.get("action"),
            "final_rank": candidate.get("final_rank"),
            "aggregate_score": candidate.get("aggregate_score"),
            "veto_layers": layers,
            "core_meta_threshold": core_meta_threshold,
            "meta_label_proba": meta,
            "action_gate_ceiling": ceiling,
            "action_gate_reasons": [str(x) for x in _listish(candidate.get("action_gate_reasons"))[:8]],
            "ready_contract_score": candidate.get("ready_contract_score"),
            "ready_contract_soft_passes": [str(x) for x in _listish(candidate.get("ready_contract_soft_passes"))[:6]],
        })

    rows.sort(key=lambda row: (float(row.get("final_rank") or 0.0), float(row.get("ready_contract_score") or 0.0)), reverse=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_candidates": len(candidates),
        "core_ready_candidates": len(rows),
        "core_meta_threshold": core_meta_threshold,
        "veto_layer_counts": dict(layer_counts),
        "candidates": rows,
    }
    atomic_write_json(_FINAL_STRONG_BUY_VETO_REPORT, payload, indent=2)


def _strong_buy_rows(candidates: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for candidate in candidates or []:
        if str(candidate.get("action") or "").upper() != "STRONG BUY":
            continue
        ticker = str(candidate.get("ticker") or "").upper()
        if not ticker:
            continue
        rows.append({
            "ticker": ticker,
            "name": candidate.get("name"),
            "final_rank": candidate.get("final_rank"),
            "aggregate_score": candidate.get("aggregate_score"),
            "ready_contract_status": candidate.get("ready_contract_status"),
            "ready_contract_score": candidate.get("ready_contract_score"),
            "entry_stance": candidate.get("entry_stance"),
            "meta_label_proba": candidate.get("meta_label_proba"),
            "action_gate_ceiling": candidate.get("action_gate_ceiling"),
            "strong_buy_eligible": candidate.get("strong_buy_eligible"),
        })
    return rows


def _read_json_dict(path: Path) -> dict:
    try:
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}
    return {}


def _bucket_label(value, edges: list[tuple[float, str]], *, missing: str = "missing") -> str:
    val = _finite_float(value)
    if val is None:
        return missing
    for edge, label in edges:
        if val < edge:
            return label
    return edges[-1][1] if edges else str(val)


def _summarise_tb_rows(rows: list[dict], key: str) -> list[dict]:
    buckets: dict[str, dict] = {}
    for row in rows:
        label = str(row.get(key) or "UNKNOWN")
        bucket = buckets.setdefault(
            label,
            {"bucket": label, "n": 0, "wins": 0, "losses": 0, "flats": 0, "tb_returns": []},
        )
        bucket["n"] += 1
        try:
            tb_label = int(row.get("tb_label"))
        except (TypeError, ValueError):
            tb_label = 0
        if tb_label > 0:
            bucket["wins"] += 1
        elif tb_label < 0:
            bucket["losses"] += 1
        else:
            bucket["flats"] += 1
        tb_ret = _finite_float(row.get("tb_return"))
        if tb_ret is not None:
            bucket["tb_returns"].append(tb_ret)

    out: list[dict] = []
    for bucket in buckets.values():
        decisive = bucket["wins"] + bucket["losses"]
        returns = bucket.pop("tb_returns")
        bucket["hit_rate"] = round(bucket["wins"] / decisive, 4) if decisive else None
        bucket["avg_tb_return"] = round(sum(returns) / len(returns), 4) if returns else None
        out.append(bucket)
    out.sort(key=lambda item: (item["n"], str(item["bucket"])), reverse=True)
    return out


def _historical_ready_contract_calibration() -> dict:
    """Summarise mature live-discovery outcomes by ready-contract features."""
    try:
        from engine.discovery_backtest import init_backtest_db
        from engine.paper_trading import _connect

        init_backtest_db()
        with _connect() as conn:
            valid = {str(row[1]) for row in conn.execute("PRAGMA table_info(signal_backtest)").fetchall()}
            wanted = [
                "ticker", "run_date", "action", "ready_contract_status",
                "strong_buy_eligible", "action_gate_ceiling", "r_r_ratio",
                "planned_position_weight", "institutional_prior_percentile",
                "institutional_prior_confidence", "meta_prob", "tb_label", "tb_return",
            ]
            columns = [c for c in wanted if c in valid]
            if "tb_label" not in columns:
                return {"available": False, "reason": "tb_label column missing"}
            rows = [
                dict(zip(columns, tuple(row)))
                for row in conn.execute(
                    f"""
                    SELECT {', '.join(columns)}
                    FROM signal_backtest
                    WHERE source='discovery'
                      AND tb_label IS NOT NULL
                    ORDER BY run_date DESC
                    """
                ).fetchall()
            ]
    except Exception as exc:
        return {"available": False, "reason": str(exc)}

    for row in rows:
        row["prior_bucket"] = _bucket_label(
            row.get("institutional_prior_percentile"),
            [(0.50, "<50%"), (0.70, "50-70%"), (0.85, "70-85%"), (0.95, "85-95%"), (9.0, ">=95%")],
        )
        row["meta_bucket"] = _bucket_label(
            row.get("meta_prob"),
            [(0.45, "<45%"), (0.50, "45-50%"), (0.55, "50-55%"), (0.65, "55-65%"), (9.0, ">=65%")],
        )
        row["rr_bucket"] = _bucket_label(
            row.get("r_r_ratio"),
            [(1.0, "<1.0x"), (1.5, "1.0-1.5x"), (2.0, "1.5-2.0x"), (9.0, ">=2.0x")],
        )
        row["position_bucket"] = _bucket_label(
            row.get("planned_position_weight"),
            [(0.005, "<0.5%"), (0.02, "0.5-2%"), (0.05, "2-5%"), (9.0, ">=5%")],
        )
        row["eligible_bucket"] = "eligible" if int(row.get("strong_buy_eligible") or 0) else "not_eligible"

    by_ready = _summarise_tb_rows(rows, "ready_contract_status")
    by_prior = _summarise_tb_rows(rows, "prior_bucket")
    recommendations: list[str] = []
    ready_pass = next((b for b in by_ready if str(b.get("bucket")).upper() == "PASS"), None)
    ready_fail = next((b for b in by_ready if str(b.get("bucket")).upper() == "FAIL"), None)
    if ready_pass and ready_pass.get("n", 0) >= 30 and (ready_pass.get("hit_rate") or 0.0) < 0.45:
        recommendations.append("Ready-contract PASS has weak mature hit rate; tighten or shadow-calibrate the weakest passing feature.")
    if ready_fail and ready_fail.get("n", 0) >= 100 and (ready_fail.get("hit_rate") or 0.0) > 0.55:
        recommendations.append("Ready-contract FAIL bucket is producing wins; inspect dominant blockers for excessive conservatism.")
    high_prior = [b for b in by_prior if str(b.get("bucket")) in {"85-95%", ">=95%"}]
    if high_prior and sum(b.get("n", 0) for b in high_prior) >= 30:
        wins = sum(b.get("wins", 0) for b in high_prior)
        losses = sum(b.get("losses", 0) for b in high_prior)
        if wins + losses and wins / (wins + losses) < 0.45:
            recommendations.append("High institutional-prior names are not validating; reduce prior weight until parity/replay fixes land.")

    overall = _summarise_tb_rows([{**row, "all": "all"} for row in rows], "all")
    return {
        "available": True,
        "sample": len(rows),
        "uses": "source='discovery' rows with mature 30d triple-barrier labels",
        "overall": overall[0] if overall else None,
        "by_ready_contract_status": by_ready,
        "by_action": _summarise_tb_rows(rows, "action"),
        "by_action_gate_ceiling": _summarise_tb_rows(rows, "action_gate_ceiling"),
        "by_prior_percentile": by_prior,
        "by_meta_probability": _summarise_tb_rows(rows, "meta_bucket"),
        "by_reward_risk": _summarise_tb_rows(rows, "rr_bucket"),
        "by_position_weight": _summarise_tb_rows(rows, "position_bucket"),
        "by_strong_buy_eligibility": _summarise_tb_rows(rows, "eligible_bucket"),
        "recommendations": recommendations,
    }


def _write_replay_live_parity_report(candidates: list[dict]) -> None:
    """Compare latest live discovery ML features against PIT replay features."""
    try:
        from engine.ml_ranker import FEATURE_COLS as _ML_FEATURE_COLS

        fields = list(_ML_FEATURE_COLS)
        comparison_scope = "ml_ranker_feature_cols"
    except Exception:
        fields = [
            "value_factor_score",
            "quality_factor_score",
            "qmj_factor_score",
            "momentum_factor_score",
            "volatility_factor_score",
            "pead_factor_score",
            "institutional_prior_score",
            "institutional_prior_percentile",
        ]
        comparison_scope = "core_factor_cols"
    tickers = []
    seen = set()
    for candidate in candidates or []:
        ticker = str(candidate.get("ticker") or "").upper().strip()
        if ticker and ticker not in seen:
            tickers.append(ticker)
            seen.add(ticker)
    if not tickers:
        atomic_write_json(
            _REPLAY_LIVE_PARITY_REPORT,
            {"generated_at": datetime.now().isoformat(timespec="seconds"), "sample": 0, "rows": []},
            indent=2,
        )
        return

    tolerance = float(getattr(config, "REPLAY_LIVE_PARITY_TOLERANCE", 0.15))
    max_date_gap_days = int(getattr(config, "REPLAY_LIVE_PARITY_MAX_DATE_GAP_DAYS", 7))
    rows_out: list[dict] = []
    drift_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    live_missing_counts: Counter[str] = Counter()
    replay_missing_counts: Counter[str] = Counter()
    missing_replay = 0
    stale_replay = 0
    try:
        from engine.discovery_backtest import init_backtest_db
        from engine.paper_trading import _connect

        init_backtest_db()
        from utils.replay_live_parity import adaptive_weight_fields, comparable_fields, compare_parity_rows
        from engine.coverage_evidence import classify_evidence, qmj_usable_components

        with _connect() as conn:
            valid = {str(row[1]) for row in conn.execute("PRAGMA table_info(signal_backtest)").fetchall()}
            compare_fields = comparable_fields(fields, valid)
            adaptive_fields = adaptive_weight_fields(valid)
            readiness_fields = comparable_fields(
                getattr(config, "REPLAY_LIVE_PARITY_READINESS_FIELDS", ()),
                valid,
            )
            select_fields = sorted(set(compare_fields) | set(adaptive_fields) | set(readiness_fields))
            select_cols = ["ticker", "run_date", "source"] + select_fields
            for ticker in tickers:
                live = conn.execute(
                    f"""
                    SELECT {', '.join(select_cols)}
                    FROM signal_backtest
                    WHERE UPPER(ticker)=? AND source='discovery'
                    ORDER BY run_date DESC, id DESC
                    LIMIT 1
                    """,
                    (ticker,),
                ).fetchone()
                replay = conn.execute(
                    f"""
                    SELECT {', '.join(select_cols)}
                    FROM signal_backtest
                    WHERE UPPER(ticker)=? AND source='replay_pit_v1'
                    ORDER BY run_date DESC, id DESC
                    LIMIT 1
                    """,
                    (ticker,),
                ).fetchone()
                if not live or not replay:
                    if not replay:
                        missing_replay += 1
                    rows_out.append({
                        "ticker": ticker,
                        "available": False,
                        "reason": "missing live discovery row" if not live else "missing replay_pit_v1 row",
                    })
                    continue
                live_d = dict(live)
                replay_d = dict(replay)
                date_gap_days = None
                try:
                    live_date = datetime.fromisoformat(str(live_d.get("run_date"))[:10])
                    replay_date = datetime.fromisoformat(str(replay_d.get("run_date"))[:10])
                    date_gap_days = abs((live_date.date() - replay_date.date()).days)
                except Exception:
                    date_gap_days = None
                if date_gap_days is not None and date_gap_days > max_date_gap_days:
                    stale_replay += 1
                    rows_out.append({
                        "ticker": ticker,
                        "available": False,
                        "reason": f"stale replay row ({date_gap_days}d gap)",
                        "latest_discovery_run": live_d.get("run_date"),
                        "latest_replay_run": replay_d.get("run_date"),
                        "date_gap_days": date_gap_days,
                    })
                    continue
                main_compare = compare_parity_rows(
                    live_d,
                    replay_d,
                    compare_fields,
                    default_tolerance=tolerance,
                )
                comparisons = main_compare["comparisons"]
                live_missing_fields = main_compare["live_missing_fields"]
                replay_missing_fields = main_compare["replay_missing_fields"]
                drift_fields = main_compare["drift_fields"]
                for field in drift_fields:
                    drift_counts[field] += 1
                for field in set(live_missing_fields) | set(replay_missing_fields):
                    missing_counts[field] += 1
                for field in live_missing_fields:
                    live_missing_counts[field] += 1
                for field in replay_missing_fields:
                    replay_missing_counts[field] += 1

                adaptive_compare = compare_parity_rows(
                    live_d,
                    replay_d,
                    adaptive_fields,
                    default_tolerance=tolerance,
                )
                readiness_compare = compare_parity_rows(
                    live_d,
                    replay_d,
                    readiness_fields,
                    default_tolerance=tolerance,
                )
                # Classify evidence so the ML gate can consume actionable
                # (non-noise) drift counts in preference to raw drifted_tickers.
                # Same classifier as parity_blocker_breakdown.py for consistency.
                evidence_class = classify_evidence(
                    ticker=ticker,
                    pit_source=live_d.get("pit_source") or live_d.get("_pit_source"),
                    qmj_components=qmj_usable_components(live_d, config_module=config),
                )
                rows_out.append({
                    "ticker": ticker,
                    "available": True,
                    "evidence_class": evidence_class,
                    "latest_discovery_run": live_d.get("run_date"),
                    "latest_replay_run": replay_d.get("run_date"),
                    "date_gap_days": date_gap_days,
                    "drift_fields": drift_fields,
                    "adaptive_weight_drift_fields": adaptive_compare["drift_fields"],
                    "readiness_drift_fields": readiness_compare["drift_fields"],
                    "live_missing_fields": live_missing_fields,
                    "replay_missing_fields": replay_missing_fields,
                    "comparisons": comparisons,
                    "adaptive_weight_comparisons": adaptive_compare["comparisons"],
                    "readiness_comparisons": readiness_compare["comparisons"],
                })
    except Exception as exc:
        atomic_write_json(
            _REPLAY_LIVE_PARITY_REPORT,
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "available": False,
                "reason": str(exc),
            },
            indent=2,
        )
        return

    available_rows = [row for row in rows_out if row.get("available")]
    drifted = [row for row in available_rows if row.get("drift_fields")]
    adaptive_drift_counts: Counter[str] = Counter()
    adaptive_missing_counts: Counter[str] = Counter()
    adaptive_live_missing_counts: Counter[str] = Counter()
    adaptive_replay_missing_counts: Counter[str] = Counter()
    readiness_drift_counts: Counter[str] = Counter()
    readiness_missing_counts: Counter[str] = Counter()
    readiness_live_missing_counts: Counter[str] = Counter()
    readiness_replay_missing_counts: Counter[str] = Counter()
    for row in available_rows:
        for field in row.get("adaptive_weight_drift_fields") or []:
            adaptive_drift_counts[field] += 1
        comparisons = row.get("adaptive_weight_comparisons") or []
        for comp in comparisons:
            if comp.get("status") != "missing":
                continue
            field = comp.get("field")
            adaptive_missing_counts[field] += 1
            if comp.get("live") is None:
                adaptive_live_missing_counts[field] += 1
            if comp.get("replay") is None:
                adaptive_replay_missing_counts[field] += 1
        for field in row.get("readiness_drift_fields") or []:
            readiness_drift_counts[field] += 1
        readiness_comparisons = row.get("readiness_comparisons") or []
        for comp in readiness_comparisons:
            if comp.get("status") != "missing":
                continue
            field = comp.get("field")
            readiness_missing_counts[field] += 1
            if comp.get("live") is None:
                readiness_live_missing_counts[field] += 1
            if comp.get("replay") is None:
                readiness_replay_missing_counts[field] += 1
    adaptive_drifted = [row for row in available_rows if row.get("adaptive_weight_drift_fields")]
    readiness_drifted = [row for row in available_rows if row.get("readiness_drift_fields")]
    adaptive_weight_parity = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "available": True,
        "comparison_scope": "adaptive_weight_fields",
        "fields": adaptive_fields if "adaptive_fields" in locals() else [],
        "tolerance": tolerance,
        "sample": len(tickers),
        "available_pairs": len(available_rows),
        "missing_replay": missing_replay,
        "stale_replay": stale_replay,
        "max_date_gap_days": max_date_gap_days,
        "drifted_tickers": len(adaptive_drifted),
        "drift_field_counts": dict(adaptive_drift_counts),
        "missing_field_counts": dict(adaptive_missing_counts),
        "live_missing_field_counts": dict(adaptive_live_missing_counts),
        "replay_missing_field_counts": dict(adaptive_replay_missing_counts),
    }
    readiness_parity = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "available": True,
        "comparison_scope": "readiness_execution_fields",
        "fields": readiness_fields if "readiness_fields" in locals() else [],
        "tolerance": tolerance,
        "sample": len(tickers),
        "available_pairs": len(available_rows),
        "missing_replay": missing_replay,
        "stale_replay": stale_replay,
        "max_date_gap_days": max_date_gap_days,
        "drifted_tickers": len(readiness_drifted),
        "drift_field_counts": dict(readiness_drift_counts),
        "missing_field_counts": dict(readiness_missing_counts),
        "live_missing_field_counts": dict(readiness_live_missing_counts),
        "replay_missing_field_counts": dict(readiness_replay_missing_counts),
    }
    # Actionable counts exclude rows whose evidence class is known-noisy
    # (yfinance_balance_only, no_data) AND ignore drift on fields that are
    # structurally explained by live-only inputs replay cannot supply.  See
    # config.REPLAY_LIVE_PARITY_NON_ACTIONABLE_EVIDENCE_CLASSES and
    # REPLAY_LIVE_PARITY_STRUCTURAL_DRIFT_FIELDS.  The ML ranker gate
    # consumes these instead of raw drifted_tickers when
    # ML_RANKER_PARITY_USE_ACTIONABLE_DRIFT is True, so the gate fires on
    # genuine divergence rather than data noise.
    non_actionable_classes = set(getattr(
        config, "REPLAY_LIVE_PARITY_NON_ACTIONABLE_EVIDENCE_CLASSES",
        ["yfinance_balance_only", "no_data"],
    ) or [])
    structural_drift_fields = set(getattr(
        config, "REPLAY_LIVE_PARITY_STRUCTURAL_DRIFT_FIELDS", []) or [])
    actionable_available_pairs = 0
    actionable_drifted_tickers = 0
    actionable_replay_missing_counts: Counter[str] = Counter()
    actionable_evidence_class_counts: Counter[str] = Counter()
    for row in available_rows:
        cls = row.get("evidence_class") or "no_data"
        actionable_evidence_class_counts[cls] += 1
        if cls in non_actionable_classes:
            continue
        actionable_available_pairs += 1
        if set(row.get("drift_fields") or []) - structural_drift_fields:
            actionable_drifted_tickers += 1
        # Pure replay-missing: live has a value but replay does not.  Avoids
        # over-counting both-missing rows (where both live and replay are
        # None — a coverage gap, not a replay-side bug).  compare_parity_rows
        # populates both live_missing_fields AND replay_missing_fields when
        # both sides are None, so reading row.get("replay_missing_fields")
        # directly inflates the actionable critical-missingness rate.
        for comp in row.get("comparisons") or []:
            if comp.get("status") != "missing":
                continue
            if comp.get("live") is None or comp.get("replay") is not None:
                continue
            f = comp.get("field")
            if not f or f in structural_drift_fields:
                continue
            actionable_replay_missing_counts[f] += 1

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "available": True,
        "comparison_scope": comparison_scope,
        "fields": compare_fields if "compare_fields" in locals() else fields,
        "tolerance": tolerance,
        "sample": len(tickers),
        "available_pairs": len(available_rows),
        "missing_replay": missing_replay,
        "stale_replay": stale_replay,
        "max_date_gap_days": max_date_gap_days,
        "drifted_tickers": len(drifted),
        "drift_field_counts": dict(drift_counts),
        "missing_field_counts": dict(missing_counts),
        "live_missing_field_counts": dict(live_missing_counts),
        "replay_missing_field_counts": dict(replay_missing_counts),
        "actionable_available_pairs": actionable_available_pairs,
        "actionable_drifted_tickers": actionable_drifted_tickers,
        "actionable_replay_missing_field_counts": dict(actionable_replay_missing_counts),
        "actionable_evidence_class_counts": dict(actionable_evidence_class_counts),
        "actionable_non_actionable_classes": sorted(non_actionable_classes),
        "actionable_structural_fields_excluded": sorted(structural_drift_fields),
        "adaptive_weight_parity": adaptive_weight_parity,
        "readiness_parity": readiness_parity,
        "top_drifted": sorted(
            drifted,
            key=lambda row: len(row.get("drift_fields") or []),
            reverse=True,
        )[:40],
        "top_replay_missing": sorted(
            [row for row in available_rows if row.get("replay_missing_fields")],
            key=lambda row: len(row.get("replay_missing_fields") or []),
            reverse=True,
        )[:80],
    }
    atomic_write_json(_REPLAY_LIVE_PARITY_REPORT, payload, indent=2)


def _recorded_discovery_rows_for_day(day_prefix: str) -> list[dict]:
    try:
        from engine.discovery_backtest import init_backtest_db
        from engine.paper_trading import _connect

        init_backtest_db()
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT ticker, action, run_date, final_rank, aggregate_score,
                       meta_prob, ready_contract_status, strong_buy_eligible
                FROM signal_backtest
                WHERE source='discovery' AND run_date LIKE ?
                ORDER BY final_rank DESC
                """,
                (day_prefix + "%",),
            ).fetchall()
        return [dict(row) for row in rows]
    except Exception as exc:
        logger.warning("Failed to query discovery rows for live sanity report: %s", exc)
        return []


def _write_live_run_sanity_report(
    candidates: list[dict],
    *,
    dry_run: bool,
    discovery_ran: bool,
    email_sent: bool,
    email_subject: str | None,
    email_html: str | None,
    n_recorded: int | None,
) -> dict:
    """Persist the live handoff check: cache -> email -> signal_backtest."""
    day_prefix = datetime.now().strftime("%Y-%m-%d")
    cached_strong = _strong_buy_rows(candidates)
    cached_tickers = [row["ticker"] for row in cached_strong]
    cached_all_tickers = {
        str(candidate.get("ticker") or "").upper()
        for candidate in candidates or []
        if candidate.get("ticker")
    }
    recorded_rows = _recorded_discovery_rows_for_day(day_prefix)
    recorded_current_rows = [
        row for row in recorded_rows
        if not cached_all_tickers or str(row.get("ticker") or "").upper() in cached_all_tickers
    ]
    recorded_strong = [
        row for row in recorded_current_rows
        if str(row.get("action") or "").upper() == "STRONG BUY"
    ]
    recorded_tickers = [str(row.get("ticker") or "").upper() for row in recorded_strong]
    extra_recorded_strong_tickers = sorted({
        str(row.get("ticker") or "").upper()
        for row in recorded_rows
        if str(row.get("action") or "").upper() == "STRONG BUY"
        and str(row.get("ticker") or "").upper() not in cached_all_tickers
    })

    subject = email_subject or ""
    body = email_html or ""
    subject_tickers = [ticker for ticker in cached_tickers if ticker in subject]
    body_tickers = [ticker for ticker in cached_tickers if ticker in body]
    visible_tickers = sorted(set(subject_tickers) | set(body_tickers))

    veto_report = _read_json_dict(_FINAL_STRONG_BUY_VETO_REPORT)
    veto_candidates = veto_report.get("candidates") or []
    if not isinstance(veto_candidates, list):
        veto_candidates = []
    veto_strong_count = sum(
        1 for row in veto_candidates
        if isinstance(row, dict) and str(row.get("action") or "").upper() == "STRONG BUY"
    )

    warnings: list[str] = []
    missing_subject = sorted(set(cached_tickers) - set(subject_tickers))
    missing_body = sorted(set(cached_tickers) - set(body_tickers))
    missing_db = sorted(set(cached_tickers) - set(recorded_tickers))
    email_was_built = bool(subject or body)
    if cached_tickers and email_was_built and missing_subject:
        warnings.append("Cached STRONG BUY tickers missing from email subject: " + ", ".join(missing_subject))
    if cached_tickers and email_was_built and missing_body:
        warnings.append("Cached STRONG BUY tickers missing from email body: " + ", ".join(missing_body))
    if discovery_ran and missing_db:
        warnings.append("Cached STRONG BUY tickers missing from signal_backtest today: " + ", ".join(missing_db))
    if discovery_ran and len(cached_tickers) != len(recorded_tickers):
        warnings.append(
            f"Cached STRONG BUY count {len(cached_tickers)} != signal_backtest count {len(recorded_tickers)}"
        )
    if discovery_ran and candidates and int(n_recorded or 0) <= 0:
        warnings.append("Discovery ran but record_discovery_picks upserted zero rows")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "date": day_prefix,
        "dry_run": bool(dry_run),
        "discovery_ran": bool(discovery_ran),
        "email_sent": bool(email_sent),
        "record_discovery_picks_count": int(n_recorded or 0),
        "cached": {
            "total_candidates": len(candidates or []),
            "action_counts": dict(Counter(str(c.get("action") or "UNKNOWN") for c in candidates or [])),
            "entry_ready_count": sum(1 for c in candidates or [] if bool(c.get("entry_ready"))),
            "strong_buy_count": len(cached_tickers),
            "strong_buy_tickers": cached_tickers,
            "strong_buy_candidates": cached_strong,
        },
        "email": {
            "subject": subject,
            "strong_buy_count_visible": len(visible_tickers),
            "strong_buy_tickers_visible": visible_tickers,
            "strong_buy_tickers_in_subject": subject_tickers,
            "strong_buy_tickers_in_body": body_tickers,
        },
        "signal_backtest": {
            "recorded_today_count": len(recorded_rows),
            "matched_current_candidates_count": len(recorded_current_rows),
            "strong_buy_count": len(recorded_tickers),
            "strong_buy_tickers": recorded_tickers,
            "strong_buy_rows": recorded_strong,
            "extra_strong_buy_tickers_today": extra_recorded_strong_tickers,
        },
        "final_veto_report": {
            "core_ready_candidates": veto_report.get("core_ready_candidates"),
            "core_meta_threshold": veto_report.get("core_meta_threshold"),
            "veto_layer_counts": veto_report.get("veto_layer_counts") or {},
            "strong_buy_count": veto_strong_count,
        },
        "warnings": warnings,
        "ok": not warnings,
    }
    atomic_write_json(_LIVE_RUN_SANITY_REPORT, payload, indent=2)
    return payload


def _candidate_missing_fundamental_fields(candidate: dict) -> list[str]:
    missing: list[str] = []
    f_score = _finite_float(candidate.get("f_score"))
    f_cov = normalize_f_score_coverage(candidate.get("f_score_coverage"))
    if not is_f_score_actionable(f_score, f_cov, config_module=config):
        missing.append("f_score")
    if _finite_float(candidate.get("gpa")) is None:
        missing.append("gpa")

    ev_ebit = _finite_float(candidate.get("ev_ebit"))
    fcf_to_assets = _finite_float(candidate.get("fcf_to_assets"))
    revenue_growth = _finite_float(candidate.get("revenue_growth"))
    pe = _finite_float(candidate.get("pe_ratio"))
    value_ok = (
        (ev_ebit is not None and 0 < ev_ebit <= 25)
        or (fcf_to_assets is not None and fcf_to_assets >= 0.03)
        or (revenue_growth is not None and revenue_growth >= 0.15)
        or (pe is not None and 0 < pe <= 30)
    )
    if not value_ok:
        missing.append("value_support")

    gate_reasons = [str(x) for x in _listish(candidate.get("gate_v2_reasons"))]
    if any("F-score missing" in reason for reason in gate_reasons) and "f_score" not in missing:
        missing.append("f_score")
    if any("GPA missing" in reason for reason in gate_reasons) and "gpa" not in missing:
        missing.append("gpa")
    if any("No clear value" in reason for reason in gate_reasons) and "value_support" not in missing:
        missing.append("value_support")
    return missing


def _write_fundamental_refresh_queue(candidates: list[dict]) -> None:
    """Track recurring finalists whose missing data blocks gate decisions."""
    now = datetime.now().isoformat(timespec="seconds")
    existing: dict[str, dict] = {}
    if _FUNDAMENTAL_REFRESH_QUEUE.exists():
        try:
            raw = json.loads(_FUNDAMENTAL_REFRESH_QUEUE.read_text(encoding="utf-8"))
            rows = raw.get("items", raw if isinstance(raw, list) else [])
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict) and row.get("ticker"):
                        existing[str(row["ticker"]).upper()] = row
        except Exception:
            existing = {}

    min_priority = float(getattr(config, "DISCOVERY_FUNDAMENTAL_REFRESH_QUEUE_MIN_PRIORITY", 1.0))
    for candidate in candidates:
        ticker = str(candidate.get("ticker") or "").upper()
        if not ticker:
            continue
        missing = _candidate_missing_fundamental_fields(candidate)
        if not missing:
            continue
        final_rank = _finite_float(candidate.get("final_rank")) or 0.0
        ready_lane = _finite_float(candidate.get("ready_lane_score")) or 0.0
        priority = len(missing) + final_rank + ready_lane
        if str(candidate.get("action") or "") in ("BUY", "STRONG BUY"):
            priority += 0.5
        if priority < min_priority:
            continue
        previous = existing.get(ticker, {})
        seen_count = int(previous.get("seen_count", 0) or 0) + 1
        existing[ticker] = {
            "ticker": ticker,
            "name": candidate.get("name"),
            "seen_count": seen_count,
            "first_seen": previous.get("first_seen") or now,
            "last_seen": now,
            "priority": round(priority + 0.25 * max(0, seen_count - 1), 3),
            "missing_fields": missing,
            "last_action": candidate.get("action"),
            "last_final_rank": final_rank,
            "ready_lane_score": ready_lane,
            "gate_v2_reasons": _listish(candidate.get("gate_v2_reasons"))[:6],
            "strong_buy_blockers": _listish(candidate.get("strong_buy_blockers"))[:6],
        }

    max_items = int(getattr(config, "DISCOVERY_FUNDAMENTAL_REFRESH_QUEUE_MAX", 200))
    items = sorted(existing.values(), key=lambda row: row.get("priority", 0), reverse=True)[:max_items]
    payload = {
        "generated_at": now,
        "count": len(items),
        "items": items,
    }
    atomic_write_json(_FUNDAMENTAL_REFRESH_QUEUE, payload, indent=2)


def _update_threshold_learner_from_matured_labels() -> int:
    """Feed live, profile-tagged triple-barrier outcomes into Tier-5 learner."""
    if not getattr(config, "THRESHOLD_LEARNER_ENABLED", True):
        return 0
    try:
        from engine.discovery_backtest import init_backtest_db
        from engine.paper_trading import _connect
        from engine.threshold_learner import (
            OutcomeBatch,
            PROFILES,
            load_state as _load_tl_state,
            save_state as _save_tl_state,
            update_posteriors,
        )

        init_backtest_db()
        infer_missing_profile = bool(getattr(config, "THRESHOLD_LEARNER_INFER_MISSING_PROFILE", True))
        fallback_profile = str(getattr(config, "THRESHOLD_LEARNER_MISSING_PROFILE_FALLBACK", "moderate")).strip()
        if fallback_profile not in PROFILES:
            fallback_profile = "moderate"
        with _connect() as conn:
            if infer_missing_profile:
                sql = """
                    SELECT
                        id,
                        CASE
                            WHEN threshold_profile IS NULL OR TRIM(threshold_profile) = ''
                            THEN ?
                            ELSE TRIM(threshold_profile)
                        END AS effective_profile,
                        tb_label,
                        CASE
                            WHEN threshold_profile IS NULL OR TRIM(threshold_profile) = ''
                            THEN 1
                            ELSE 0
                        END AS profile_inferred
                    FROM signal_backtest
                    WHERE source='discovery'
                      AND tb_label IS NOT NULL
                      AND COALESCE(tb_label_processed_for_threshold, 0) = 0
                """
                params = (fallback_profile,)
            else:
                sql = """
                    SELECT id, TRIM(threshold_profile) AS effective_profile, tb_label, 0 AS profile_inferred
                    FROM signal_backtest
                    WHERE source='discovery'
                      AND threshold_profile IS NOT NULL
                      AND TRIM(threshold_profile) <> ''
                      AND tb_label IS NOT NULL
                      AND COALESCE(tb_label_processed_for_threshold, 0) = 0
                """
                params = ()
            rows = conn.execute(
                sql,
                params,
            ).fetchall()
            if not rows:
                logger.info("Threshold learner: no unprocessed profile-tagged TB outcomes.")
                return 0

            by_profile: dict[str, dict[str, int]] = {}
            processed_ids: list[int] = []
            inferred_missing = 0
            ignored_profiles: Counter[str] = Counter()
            for row_id, profile, label, profile_inferred in rows:
                processed_ids.append(int(row_id))
                profile_name = str(profile or "").strip()
                if profile_name not in PROFILES:
                    ignored_profiles[profile_name or "UNKNOWN"] += 1
                    continue
                if int(profile_inferred or 0):
                    inferred_missing += 1
                bucket = by_profile.setdefault(profile_name, {"wins": 0, "losses": 0})
                try:
                    label_int = int(label)
                except (TypeError, ValueError):
                    continue
                if label_int > 0:
                    bucket["wins"] += 1
                elif label_int < 0:
                    bucket["losses"] += 1

            batches = [
                OutcomeBatch(profile=profile, successes=counts["wins"], failures=counts["losses"])
                for profile, counts in sorted(by_profile.items())
                if counts["wins"] or counts["losses"]
            ]
            if batches:
                tl_state = _load_tl_state(config)
                update_posteriors(tl_state, batches)
                _save_tl_state(tl_state, config)
                logger.info(
                    "Threshold learner updated from %d profile-tagged TB outcomes: %s",
                    sum(b.successes + b.failures for b in batches),
                    {b.profile: {"wins": b.successes, "losses": b.failures} for b in batches},
                )
            else:
                logger.info("Threshold learner saw %d flat/invalid TB outcomes; marking processed.", len(rows))

            conn.executemany(
                "UPDATE signal_backtest SET tb_label_processed_for_threshold=1 WHERE id=?",
                [(row_id,) for row_id in processed_ids],
            )
            atomic_write_json(
                _THRESHOLD_LEARNER_UPDATE_REPORT,
                {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "selected_rows": len(rows),
                    "processed_rows": len(processed_ids),
                    "inferred_missing_profile": inferred_missing,
                    "fallback_profile": fallback_profile if infer_missing_profile else None,
                    "ignored_profiles": dict(ignored_profiles),
                    "batches": {
                        b.profile: {"wins": b.successes, "losses": b.failures}
                        for b in batches
                    },
                },
                indent=2,
            )
            return len(processed_ids)
    except Exception as exc:
        logger.warning("Threshold learner outcome update failed: %s", exc)
        return 0


# ---------------------------------------------------------------------------
# Single-instance lock
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorRunLock:
    """Exclusive lock file for a single orchestrator process."""
    path: Path
    token: str
    metadata: dict
    reclaimed: dict | None = None
    released: bool = False

    def release(self) -> None:
        """Release the lock if we still own it."""
        if self.released:
            return
        try:
            current = _read_lock_metadata(self.path)
            if current and current.get("token") not in (None, self.token):
                logger.warning(
                    "Lock ownership changed while running; leaving %s in place.",
                    self.path,
                )
                return
            self.path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Failed to release orchestrator lock %s: %s", self.path, e)
        finally:
            self.released = True


class OrchestratorAlreadyRunning(RuntimeError):
    """Raised when another orchestrator invocation is already active."""

    def __init__(self, metadata: dict | None = None):
        super().__init__("another orchestrator run is already active")
        self.metadata = metadata or {}


def _read_lock_metadata(path: Path) -> dict | None:
    """Best-effort JSON loader for the lock file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _lock_age_seconds(path: Path, metadata: dict | None) -> float:
    """Estimate how old the lock is, preferring its started_at timestamp."""
    if metadata:
        started_at = metadata.get("started_at")
        if started_at:
            try:
                started = datetime.fromisoformat(str(started_at))
                return max((datetime.now() - started).total_seconds(), 0.0)
            except ValueError:
                pass
    try:
        return max(time.time() - path.stat().st_mtime, 0.0)
    except OSError:
        return 0.0


def _lock_pid_is_alive(metadata: dict | None) -> bool:
    """Best-effort local PID liveness check for reclaiming interrupted locks."""
    if not metadata:
        return True
    if metadata.get("hostname") not in (None, socket.gethostname()):
        return True
    try:
        pid = int(metadata.get("pid"))
    except (TypeError, ValueError):
        return True
    if pid <= 0 or pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            import subprocess

            proc = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return str(pid) in (proc.stdout or "")
        except Exception:
            return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _is_lock_stale(path: Path, metadata: dict | None, stale_seconds: int) -> bool:
    """Return True when the existing lock is old enough to reclaim safely."""
    if not _lock_pid_is_alive(metadata):
        return True
    return _lock_age_seconds(path, metadata) >= max(int(stale_seconds), 1)


def _acquire_orchestrator_lock(
    *,
    dry_run: bool,
    force_discovery: bool,
    portfolio_only: bool,
    path: Path | None = None,
    stale_seconds: int | None = None,
) -> OrchestratorRunLock:
    """Acquire the single-instance orchestrator lock or raise if busy."""
    lock_path = path or (ROOT / getattr(config, "ORCHESTRATOR_LOCK_FILE", "feature_cache/orchestrator.lock"))
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    if stale_seconds is None:
        max_runtime = int(getattr(config, "ORCHESTRATOR_MAX_RUNTIME", 3600))
        stale_seconds = int(
            getattr(config, "ORCHESTRATOR_LOCK_STALE_SECONDS", max_runtime + 1800)
        )

    token = f"{os.getpid()}-{time.time_ns()}"
    metadata = {
        "pid": os.getpid(),
        "token": token,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "cwd": str(ROOT),
        "argv": list(sys.argv),
        "dry_run": bool(dry_run),
        "force_discovery": bool(force_discovery),
        "portfolio_only": bool(portfolio_only),
    }
    reclaimed: dict | None = None

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(metadata, f, default=str)
                f.flush()
                os.fsync(f.fileno())
            return OrchestratorRunLock(
                path=lock_path,
                token=token,
                metadata=metadata,
                reclaimed=reclaimed,
            )
        except FileExistsError:
            existing = _read_lock_metadata(lock_path)
            if not _is_lock_stale(lock_path, existing, stale_seconds):
                raise OrchestratorAlreadyRunning(existing)

            logger.warning(
                "Stale orchestrator lock detected at %s (age %.0fs >= %ss); reclaiming.",
                lock_path,
                _lock_age_seconds(lock_path, existing),
                stale_seconds,
            )
            try:
                lock_path.unlink()
                reclaimed = existing
            except FileNotFoundError:
                continue
            except OSError as e:
                raise OrchestratorAlreadyRunning({
                    "path": str(lock_path),
                    "error": f"stale_lock_reclaim_failed: {e}",
                }) from e


# ---------------------------------------------------------------------------
# Decision log — append-only JSONL file
# ---------------------------------------------------------------------------

def _log_decision(event: str, details: dict | None = None) -> None:
    """Append a structured decision record to the JSONL log file."""
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        "details": details or {},
    }
    log_path = ROOT / config.ORCHESTRATOR_LOG_FILE
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError as e:
        logger.warning("Failed to write decision log: %s", e)


# ---------------------------------------------------------------------------
# Timeout wrapper (thread-based, Windows-compatible)
# ---------------------------------------------------------------------------

def _run_with_timeout(func, args=(), timeout_seconds: int = 300):
    """Run func(*args) with a timeout.  Raises TimeoutError on expiry.

    Uses threading.Thread.join(timeout) because Windows lacks signal.SIGALRM.
    Logs a heartbeat every 60s while waiting so operators can see the process
    is alive even when the inner function is silent.
    """
    result_box = [None]
    error_box = [None]
    _start = time.time()

    def _target():
        try:
            result_box[0] = func(*args)
        except Exception as e:
            error_box[0] = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()

    # Poll with heartbeat instead of blocking join — gives visibility
    _HEARTBEAT_INTERVAL = 60  # seconds
    remaining = timeout_seconds
    while remaining > 0 and thread.is_alive():
        wait = min(_HEARTBEAT_INTERVAL, remaining)
        thread.join(timeout=wait)
        remaining -= wait
        if thread.is_alive() and remaining > 0:
            elapsed = time.time() - _start
            logger.info("[heartbeat] %s running for %.0fs (timeout in %.0fs)",
                        func.__name__, elapsed, remaining)

    if thread.is_alive():
        elapsed = time.time() - _start
        logger.error("[timeout] %s exceeded %ds timeout (ran %.0fs). "
                     "Thread is orphaned — it will finish naturally but results are discarded.",
                     func.__name__, timeout_seconds, elapsed)
        _log_decision("timeout", {
            "function": func.__name__,
            "timeout_s": timeout_seconds,
            "elapsed_s": round(elapsed, 1),
        })
        raise TimeoutError(f"{func.__name__} exceeded {timeout_seconds}s timeout")
    if error_box[0]:
        raise error_box[0]
    return result_box[0]


# ---------------------------------------------------------------------------
# Portfolio analysis step
# ---------------------------------------------------------------------------

def _run_portfolio_analysis(holdings: list[dict]) -> tuple:
    """Run analyse_portfolio and return (results, risk_data, position_weights)."""
    from engine.scoring import analyse_portfolio
    return analyse_portfolio(holdings)


def _get_regime() -> dict:
    """Get VIX regime, returning neutral defaults on failure."""
    try:
        from engine.regime import get_vix_regime
        return get_vix_regime()
    except Exception as e:
        logger.warning("VIX regime detection failed: %s", e)
        return {"vix_level": 0.0, "vix_percentile": 50.0, "regime_label": "NEUTRAL"}


# ---------------------------------------------------------------------------
# Discovery step
# ---------------------------------------------------------------------------

def _run_discovery_pipeline(holdings: list[dict], risk_data: dict):
    """Run the global discovery engine. Returns DiscoveryResult."""
    from engine.discovery import run_discovery
    return run_discovery(holdings, risk_data)


def _reconstitute_dynamic_universe() -> dict | None:
    """Refresh the dynamic universe supplement before discovery.

    Pulls fresh ETF-holdings candidates through the validation gates so that
    new entrants are admitted only if they pass liquidity/mcap floors. Stale
    tickers are pruned after N consecutive misses. Failures are non-fatal —
    discovery will fall back to the existing supplement and direct ETF inject.
    """
    try:
        from utils.global_universe import (
            decompose_etf_holdings,
            reconstitute_universe,
        )
    except Exception as e:
        logger.debug("Universe reconstitution modules unavailable: %s", e)
        return None

    extras: list[str] = []
    if getattr(config, "DISCOVERY_USE_ETF_DECOMPOSITION", True):
        try:
            extras = decompose_etf_holdings() or []
        except Exception as e:
            logger.warning("ETF decomposition failed pre-reconstitution: %s", e)
            extras = []

    # Benchmark-driven entrants (S&P 500, S&P MidCap 400, FTSE 100/250,
    # CAC, DAX, IBEX, ...). Each constituent still has to clear the dynamic
    # universe liquidity/mcap floors before it enters live discovery.
    if getattr(config, "DISCOVERY_USE_BENCHMARK_REBUILD", True):
        try:
            from utils.benchmark_constituents import get_benchmark_tickers
            ttl_days = int(getattr(config, "DISCOVERY_BENCHMARK_TTL_DAYS", 30))
            bench = get_benchmark_tickers(ttl_days=ttl_days) or []
            before = len(extras)
            seen = set(extras)
            for t in bench:
                if t not in seen:
                    extras.append(t)
                    seen.add(t)
            logger.info(
                "Benchmark constituents: %d unique tickers, %d net-new",
                len(bench),
                len(extras) - before,
            )
        except Exception as e:
            logger.warning("Benchmark constituents fetch failed: %s", e)

    try:
        summary = reconstitute_universe(extra_tickers=extras)
        logger.info(
            "Dynamic universe reconstituted: %d validated, %d pruned "
            "(extras fed in: %d)",
            summary.get("validated", 0),
            summary.get("pruned", 0),
            len(extras),
        )
        return summary
    except Exception as e:
        logger.warning("reconstitute_universe() failed: %s", e)
        return None


def _dynamic_universe_cache_fresh() -> bool:
    """Return True when dynamic universe validation was refreshed recently."""
    try:
        ttl_hours = float(getattr(config, "DISCOVERY_RECONSTITUTE_TTL_HOURS", 24))
        if ttl_hours <= 0:
            return False
        path = ROOT / "feature_cache" / "universe_dynamic.json"
        if not path.exists():
            return False
        import json
        from datetime import datetime

        payload = json.loads(path.read_text(encoding="utf-8"))
        refreshed_at = payload.get("refreshed_at")
        if not refreshed_at:
            return False
        refreshed = datetime.fromisoformat(refreshed_at)
        if getattr(config, "DISCOVERY_USE_BENCHMARK_REBUILD", True):
            try:
                from utils.benchmark_constituents import get_benchmark_cache_metadata
                bench_meta = get_benchmark_cache_metadata()
                missing = bench_meta.get("missing") or []
                required = {
                    str(key).upper()
                    for key in getattr(config, "DISCOVERY_REQUIRED_BENCHMARKS", ("SP500", "SPMIDCAP400"))
                }
                missing_required = [key for key in missing if str(key).upper() in required]
                if missing_required:
                    logger.info(
                        "Dynamic universe reconstitution required: benchmark cache missing %s",
                        ", ".join(map(str, missing_required[:8])),
                    )
                    return False
                bench_refreshed_at = bench_meta.get("latest_refreshed_at")
                if bench_refreshed_at:
                    bench_refreshed = datetime.fromisoformat(bench_refreshed_at)
                    if bench_refreshed > refreshed:
                        logger.info(
                            "Dynamic universe reconstitution required: benchmark cache newer than dynamic universe"
                        )
                        return False
            except Exception as exc:
                logger.debug("Benchmark freshness check failed: %s", exc)
        age_hours = (datetime.now() - refreshed).total_seconds() / 3600.0
        if age_hours < ttl_hours:
            logger.info(
                "Dynamic universe reconstitution skipped: cache fresh %.1fh < %.1fh",
                age_hours,
                ttl_hours,
            )
            return True
    except Exception as exc:
        logger.debug("Dynamic universe freshness check failed: %s", exc)
    return False


def save_discovery_results(disc_result, state: dict) -> int:
    """Persist discovery results to state and record picks for backtest.

    Extracted so both the background orchestrator and the UI button can call it.
    Returns the number of backtest picks recorded.
    """
    n_candidates = len(disc_result.candidates)

    state["cached_discovery"] = [
        {
            "ticker": c.ticker,
            "name": c.name,
            "exchange": c.exchange,
            "country": c.country,
            "sector": c.sector,
            "industry": c.industry,
            "market_cap": c.market_cap,
            "currency": c.currency,
            "aggregate_score": c.aggregate_score,
            "technical_score": c.technical_score,
            "fundamental_score": c.fundamental_score,
            "sentiment_score": c.sentiment_score,
            "forecast_score": c.forecast_score,
            "action": c.action,
            "why": c.why,
            "fx_penalty_applied": c.fx_penalty_applied,
            "fx_penalty_pct": c.fx_penalty_pct,
            "max_correlation": c.max_correlation,
            "correlated_with": c.correlated_with,
            "sector_weight_if_added": c.sector_weight_if_added,
            "portfolio_fit_score": c.portfolio_fit_score,
            "momentum_score": c.momentum_score,
            "return_90d": c.return_90d,
            "return_30d": c.return_30d,
            "volume_ratio": c.volume_ratio,
            "expected_return_90d": c.expected_return_90d,
            "analyst_target": getattr(c, "analyst_target", None),
            "analyst_upside": getattr(c, "analyst_upside", None),
            "num_analysts": getattr(c, "num_analysts", None),
            "insider_buys": getattr(c, "insider_buys", 0),
            "insider_sells": getattr(c, "insider_sells", 0),
            "insider_net": getattr(c, "insider_net", ""),
            "pe_ratio": getattr(c, "pe_ratio", None),
            "peg_ratio": getattr(c, "peg_ratio", None),
            "revenue_growth": getattr(c, "revenue_growth", None),
            "beta_90d": getattr(c, "beta_90d", None),
            "debt_to_equity": getattr(c, "debt_to_equity", None),
            "entry_stance": getattr(c, "entry_stance", "Ready"),
            "ticker_identity_warning": getattr(c, "ticker_identity_warning", None),
            "parabolic_penalty": c.parabolic_penalty,
            "is_parabolic": c.is_parabolic,
            "earnings_near": c.earnings_near,
            "earnings_imminent": c.earnings_imminent,
            "earnings_days": c.earnings_days,
            "cap_tier": c.cap_tier,
            "confidence_discount": c.confidence_discount,
            "effective_data_confidence": getattr(c, "effective_data_confidence", None),
            "max_weight_scale": c.max_weight_scale,
            "post_earnings_recent": c.post_earnings_recent,
            "post_earnings_days": c.post_earnings_days,
            "earnings_miss": c.earnings_miss,
            "earnings_miss_pct": c.earnings_miss_pct,
            "near_52w_high": c.near_52w_high,
            "pct_from_52w_high": c.pct_from_52w_high,
            "entry_lens": getattr(c, "entry_lens", "momentum"),
            "entry_price": getattr(c, "entry_price", None),
            "entry_method": getattr(c, "entry_method", ""),
            "entry_zone_low": getattr(c, "entry_zone_low", None),
            "entry_zone_high": getattr(c, "entry_zone_high", None),
            "fill_probability": getattr(c, "fill_probability", None),
            "stop_loss": getattr(c, "stop_loss", None),
            "stop_method": getattr(c, "stop_method", ""),
            "stop_distance_pct": getattr(c, "stop_distance_pct", None),
            "take_profit": getattr(c, "take_profit", None),
            "target_method": getattr(c, "target_method", ""),
            "position_size_shares": getattr(c, "position_size_shares", 0),
            "position_weight": getattr(c, "position_weight", 0),
            "risk_amount": getattr(c, "risk_amount", 0),
            "r_r_ratio": getattr(c, "r_r_ratio", None),
            "sizing_method": getattr(c, "sizing_method", ""),
            "kelly_cap_fraction": getattr(c, "kelly_cap_fraction", None),
            "support_levels": getattr(c, "support_levels", {}),
            "regime_info": getattr(c, "regime_info", {}),
            "regime": getattr(c, "regime", None),
            # Dividend safety
            "dividend_yield": getattr(c, "dividend_yield", None),
            "payout_ratio": getattr(c, "payout_ratio", None),
            "ex_dividend_date": getattr(c, "ex_dividend_date", None),
            "ex_dividend_days": getattr(c, "ex_dividend_days", None),
            "five_year_avg_yield": getattr(c, "five_year_avg_yield", None),
            # Balance sheet strength
            "balance_sheet_grade": getattr(c, "balance_sheet_grade", None),
            "net_debt_ebitda": getattr(c, "net_debt_ebitda", None),
            "current_ratio": getattr(c, "current_ratio", None),
            "cash_to_debt": getattr(c, "cash_to_debt", None),
            # Governance red flag
            "governance_flag": getattr(c, "governance_flag", False),
            "governance_reasons": getattr(c, "governance_reasons", []),
            # Asymmetric / binary outcome flag
            "asymmetric_risk_flag": getattr(c, "asymmetric_risk_flag", False),
            "asymmetric_risk_reason": getattr(c, "asymmetric_risk_reason", None),
            # Enterprise factor bundle (roadmap items #1, #2)
            "enterprise_value": getattr(c, "enterprise_value", None),
            "ev_ebit": getattr(c, "ev_ebit", None),
            "ev_ebitda": getattr(c, "ev_ebitda", None),
            "ebit_yield": getattr(c, "ebit_yield", None),
            "ev_ebit_score": getattr(c, "ev_ebit_score", None),
            "fcf_to_assets": getattr(c, "fcf_to_assets", None),
            "gpa": getattr(c, "gpa", None),
            "gpa_score": getattr(c, "gpa_score", None),
            "f_score": getattr(c, "f_score", None),
            "f_score_gate": getattr(c, "f_score_gate", False),
            "f_score_score": getattr(c, "f_score_score", None),
            "f_score_coverage": getattr(c, "f_score_coverage", None),
            "qmj_component_count": getattr(c, "qmj_component_count", None),
            "pit_source": getattr(c, "pit_source", None),
            # Day-1 quality / self-learning diagnostics
            "institutional_prior_score": getattr(c, "institutional_prior_score", None),
            "institutional_prior_percentile": getattr(c, "institutional_prior_percentile", None),
            "institutional_prior_confidence": getattr(c, "institutional_prior_confidence", None),
            "institutional_prior_coverage": getattr(c, "institutional_prior_coverage", None),
            "institutional_prior_rank": getattr(c, "institutional_prior_rank", None),
            "strong_buy_eligible": getattr(c, "strong_buy_eligible", None),
            "ready_lane_score": getattr(c, "ready_lane_score", None),
            "ready_lane_missing_fields": getattr(c, "ready_lane_missing_fields", None),
            "ready_contract_core_status": getattr(c, "ready_contract_core_status", None),
            "ready_contract_core_reasons": getattr(c, "ready_contract_core_reasons", None),
            "ready_contract_status": getattr(c, "ready_contract_status", None),
            "ready_contract_reasons": getattr(c, "ready_contract_reasons", None),
            "ready_contract_score": getattr(c, "ready_contract_score", None),
            "ready_contract_soft_passes": getattr(c, "ready_contract_soft_passes", None),
            "strong_buy_blockers": getattr(c, "strong_buy_blockers", None),
            "gate_v2_status": getattr(c, "gate_v2_status", None),
            "gate_v2_reasons": getattr(c, "gate_v2_reasons", None),
            "action_gate_ceiling": getattr(c, "action_gate_ceiling", None),
            "action_gate_reasons": getattr(c, "action_gate_reasons", None),
            "action_gate_flags": getattr(c, "action_gate_flags", None),
            "threshold_profile": getattr(c, "threshold_profile", None),
            "meta_label_proba": getattr(c, "meta_success_prob", None),
            "ml_alpha_raw": getattr(c, "ml_alpha_raw", None),
            "ml_shadow_score": getattr(c, "ml_shadow_score", None),
            "sleeve_composite": getattr(c, "sleeve_composite", None),
            "sleeve_momentum": getattr(c, "sleeve_momentum", None),
            "sleeve_quality": getattr(c, "sleeve_quality", None),
            "sleeve_value": getattr(c, "sleeve_value", None),
            "sleeve_low_risk": getattr(c, "sleeve_low_risk", None),
            "sleeve_pead": getattr(c, "sleeve_pead", None),
            "sleeve_ready": getattr(c, "sleeve_ready", None),
            # Split rank components (roadmap item #5)
            "selection_rank": getattr(c, "selection_rank", None),
            "timing_rank": getattr(c, "timing_rank", None),
            "portfolio_fit_rank": getattr(c, "portfolio_fit_rank", None),
            # Cold-start scorecard fields (Phase 2 Steps 4 + 5).
            "sb_score": getattr(c, "sb_score", None),
            "scorecard_override": getattr(c, "scorecard_override", False),
            "scorecard_override_blockers": getattr(c, "scorecard_override_blockers", None),
            "conformal_p": getattr(c, "conformal_p", None),
            "final_rank": c.final_rank,
        }
        for c in disc_result.candidates
    ]
    for payload in state["cached_discovery"]:
        try:
            payload["entry_ready"] = bool(candidate_entry_ready(payload))
        except Exception:
            payload["entry_ready"] = False
    state["cached_discovery_meta"] = {
        "screened_count": disc_result.screened_count,
        "after_momentum_screen": disc_result.after_momentum_screen,
        "after_quick_filter": disc_result.after_quick_filter,
        "after_corr_filter": disc_result.after_corr_filter,
        "after_quick_rank": disc_result.after_quick_rank,
        "fully_scored": disc_result.fully_scored,
        "run_time_seconds": disc_result.run_time_seconds,
        "fx_penalties_applied": disc_result.fx_penalties_applied,
        "stage_timings": getattr(disc_result, "stage_timings", {}),
    }
    state["last_discovery_run"] = datetime.now().isoformat()

    save_state(state)
    logger.info("Discovery results saved to state (%d candidates).", n_candidates)
    try:
        _write_strong_buy_blocker_report(state["cached_discovery"])
    except Exception as e:
        logger.warning("Failed to write STRONG BUY blocker report: %s", e)
    try:
        _write_ready_contract_calibration_report(state["cached_discovery"])
    except Exception as e:
        logger.warning("Failed to write ready contract calibration report: %s", e)
    try:
        _write_final_strong_buy_veto_report(state["cached_discovery"])
    except Exception as e:
        logger.warning("Failed to write final STRONG BUY veto report: %s", e)
    # Auto-chain: refresh replay rows on today's discovery cohort BEFORE writing
    # the parity report.  Without this, today's live rows (computed at the
    # orchestrator run time, often before US market close when the price cache
    # is one day stale) get compared against replay rows from a previous day
    # that ran with a fresher cache.  The timing gap produces phantom drift on
    # technical_score, momentum, and return fields — fields that are noisier
    # the further apart the live and replay price-cache endpoints are.
    # Observed 2026-05-14: pre-refresh parity showed 43.1% actionable drift;
    # after running this refresh manually, drift dropped to 20.6% on the same
    # cohort.  Auto-chaining ensures the orchestrator's daily parity report is
    # always cohort-aligned without manual intervention.
    if bool(getattr(config, "ORCHESTRATOR_AUTO_REPLAY_REFRESH", True)):
        try:
            from utils.replay_parity_refresh import refresh_replay_parity
            refresh_replay_parity(refresh_existing=True)
            logger.info("Auto-chained replay refresh complete.")
        except Exception as e:
            logger.warning("Auto-chained replay refresh failed (non-fatal): %s", e)
    try:
        _write_replay_live_parity_report(state["cached_discovery"])
    except Exception as e:
        logger.warning("Failed to write replay/live parity report: %s", e)
    try:
        _write_fundamental_refresh_queue(state["cached_discovery"])
    except Exception as e:
        logger.warning("Failed to write fundamental refresh queue: %s", e)

    # Record picks for backtest tracking
    n_recorded = 0
    try:
        n_recorded = record_discovery_picks(disc_result.candidates)
        logger.info("Recorded %d discovery picks for backtest.", n_recorded)
    except Exception as e:
        logger.warning("Failed to record discovery picks: %s", e)

    # Active labelling: record borderline candidates (just-below STRONG BUY) so
    # their realised 90d returns become high-information-gain training samples
    # for boundary refinement.  Settles 2010 / Cohn-Atlas-Ladner 1994.
    try:
        n_borderline = record_borderline_signals(disc_result.candidates)
        if n_borderline > 0:
            logger.info("Active labelling: recorded %d borderline candidates.", n_borderline)
    except Exception as e:
        logger.warning("Active labelling failed (non-fatal): %s", e)

    return n_recorded


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------

def _evaluate_swaps(
    results: list[dict],
    cached_candidates: list[dict],
    state: dict,
) -> list[dict]:
    """Evaluate discovery candidates against the weakest holdings (multi-swap).

    Returns list of swap recommendations that pass ALL hurdles:
    1. candidate.aggregate_score - target.aggregate_score >= HURDLE_RATE
    2. candidate.portfolio_fit_score >= PORTFOLIO_FIT_MIN
    3. candidate is not on cooldown

    Multi-swap: evaluates up to MAX_SWAPS_PER_RUN weakest holdings as swap targets.
    Each candidate can only be recommended once, and each holding can only be
    swapped out once per run.
    """
    if not results or not cached_candidates:
        return []

    hurdle = getattr(config, "HURDLE_RATE", 0.20)
    fit_min = getattr(config, "PORTFOLIO_FIT_MIN", 0.50)
    max_swaps = getattr(config, "DISCOVERY_MAX_SWAPS_PER_RUN", 3)
    swap_threshold = getattr(config, "SWAP_CANDIDATE_THRESHOLD", -0.10)

    # Find swap-eligible holdings — prioritise exit-flagged holdings first.
    # Exit reconciliation sets _exit_override=True and _exit_posterior on holdings
    # downgraded by stop/momentum/decay signals. These should be swapped before
    # merely low-scoring holdings, since the exit engine has identified active risk.
    exit_flagged = [
        r for r in results
        if r.get("_exit_override") and r.get("final_action", r.get("action")) in ("SELL", "STRONG SELL")
    ]
    # Sort exit-flagged by posterior score ascending (worst risk first)
    exit_flagged.sort(key=lambda r: r.get("_exit_posterior", r.get("aggregate_score", 0)))

    # Then add score-based candidates (not already in exit list)
    exit_tickers = {r["ticker"] for r in exit_flagged}
    sorted_by_score = sorted(results, key=lambda r: r.get("aggregate_score", 0))
    score_eligible = [
        r for r in sorted_by_score
        if r.get("aggregate_score", 0) < swap_threshold and r["ticker"] not in exit_tickers
    ]

    # Combine: exit-flagged first, then score-based
    swap_eligible = exit_flagged + score_eligible

    # Even if none qualify, always consider the single weakest
    if not swap_eligible and sorted_by_score:
        swap_eligible = [sorted_by_score[0]]

    # Sort candidates by final discovery decision rank descending (best first)
    sorted_candidates = sorted(
        cached_candidates,
        key=lambda c: (c.get("final_rank", c.get("aggregate_score", 0)), c.get("aggregate_score", 0)),
        reverse=True,
    )

    swap_recs = []
    used_candidates = set()
    used_holdings = set()

    for target in swap_eligible:
        if len(swap_recs) >= max_swaps:
            break

        target_ticker = target["ticker"]
        target_score = target.get("_exit_posterior", target.get("aggregate_score", 0))

        if target_ticker in used_holdings:
            continue

        for cand in sorted_candidates:
            cand_ticker = cand.get("ticker", "")
            if cand_ticker in used_candidates:
                continue

            cand_score = cand.get("final_rank", cand.get("aggregate_score", 0))
            cand_fit = cand.get("portfolio_fit_score", 0)
            cand_action = cand.get("action", "NEUTRAL")
            delta = cand_score - target_score

            passes_hurdle = delta >= hurdle
            passes_fit = cand_fit >= fit_min
            require_entry_ready = bool(getattr(config, "DISCOVERY_SWAPS_REQUIRE_ENTRY_READY", True))
            passes_entry_ready = (not require_entry_ready) or bool(candidate_entry_ready(cand))
            passes_action = cand_action in ("BUY", "STRONG BUY") and passes_entry_ready
            on_cooldown = is_on_cooldown(state, cand_ticker)
            recommended = passes_action and passes_hurdle and passes_fit and not on_cooldown

            # Log every evaluation for audit
            _log_decision("swap_eval", {
                "candidate": cand_ticker,
                "candidate_score": round(cand_score, 3),
                "candidate_action": cand_action,
                "candidate_fit": round(cand_fit, 3),
                "target": target_ticker,
                "target_score": round(target_score, 3),
                "delta": round(delta, 3),
                "passes_action": passes_action,
                "passes_entry_ready": passes_entry_ready,
                "passes_hurdle": passes_hurdle,
                "passes_fit": passes_fit,
                "on_cooldown": on_cooldown,
                "recommended": recommended,
            })

            if recommended:
                set_cooldown(state, cand_ticker)
                used_candidates.add(cand_ticker)
                used_holdings.add(target_ticker)
                swap_recs.append({
                    "candidate": cand,
                    "weakest_ticker": target_ticker,
                    "weakest_score": round(target_score, 3),
                    "score_delta": round(delta, 3),
                })
                break  # Move to next target holding

    return swap_recs


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_orchestrator(
    dry_run: bool = False,
    force_discovery: bool = False,
    portfolio_only: bool = False,
    discovery_lightweight_n: int | None = None,
    discovery_top_n: int | None = None,
    discovery_timeout: int | None = None,
    discovery_component_timeout: int | None = None,
    discovery_per_ticker_timeout: int | None = None,
    discovery_stage5b_timeout: int | None = None,
    discovery_challenge_reserve_n: int | None = None,
) -> dict:
    """Main orchestration entry point.

    Returns a summary dict for logging / testing.
    """
    start = time.time()
    max_runtime = getattr(config, "ORCHESTRATOR_MAX_RUNTIME", 3600)
    summary = {
        "dry_run": dry_run,
        "portfolio_ran": False,
        "discovery_ran": False,
        "alerts": [],
        "swap_recs": [],
        "discovery_recorded": 0,
        "discovery_strong_buys": [],
        "email_sent": False,
        "live_sanity_ok": None,
        "error": None,
    }

    def _check_timeout(stage: str):
        """Raise if total runtime exceeds max. Ensures process always completes."""
        elapsed = time.time() - start
        if elapsed > max_runtime:
            raise TimeoutError(f"Orchestrator timeout after {elapsed:.0f}s in {stage} "
                               f"(limit: {max_runtime}s). Partial results will be used.")

    _log_decision("run_start", {
        "dry_run": dry_run,
        "force_discovery": force_discovery,
        "portfolio_only": portfolio_only,
        "discovery_lightweight_n": discovery_lightweight_n,
        "discovery_top_n": discovery_top_n,
        "discovery_timeout": discovery_timeout,
        "discovery_component_timeout": discovery_component_timeout,
        "discovery_per_ticker_timeout": discovery_per_ticker_timeout,
        "discovery_stage5b_timeout": discovery_stage5b_timeout,
        "discovery_challenge_reserve_n": discovery_challenge_reserve_n,
    })

    # --- Paper trading: resolve yesterday's pending signals (T+1 fills) ---
    if _paper_trading_enabled_for_run(dry_run):
        try:
            _init_paper_db()
            n_resolved = resolve_pending_signals()
            if n_resolved:
                logger.info("Paper trading: resolved %d pending signals.", n_resolved)
        except Exception as e:
            logger.warning("Paper trading fill resolution failed: %s", e)
    elif dry_run and getattr(config, "PAPER_TRADING_ENABLED", False):
        logger.info("[DRY RUN] Paper trading ledger writes skipped.")

    # --- Discovery backtest: evaluate picks that are 90+ days old ---
    try:
        n_evaluated = evaluate_matured_signals()
        if n_evaluated:
            logger.info("Signal backtest: evaluated %d matured signal-horizon pairs.", n_evaluated)
    except Exception as e:
        logger.warning("Discovery backtest evaluation failed: %s", e)
    try:
        n_threshold_updates = _update_threshold_learner_from_matured_labels()
        if n_threshold_updates:
            logger.info("Threshold learner: marked %d profile-tagged TB outcomes processed.", n_threshold_updates)
    except Exception as e:
        logger.warning("Threshold learner update step failed: %s", e)

    # --- Load state and portfolio ---
    state = load_state()
    prune_expired_cooldowns(state)

    if getattr(config, "ML_DRIFT_MONITOR_ENABLED", True):
        try:
            from utils.drift_monitor import get_drift_status

            drift = get_drift_status(write=True)
            drift_payload = dict(drift.__dict__)
            previous = state.get("drift_status") or {}
            consecutive = int(previous.get("consecutive_alerts", 0) or 0)
            if drift.drift_detected:
                consecutive += 1
            else:
                consecutive = 0
            drift_payload["consecutive_alerts"] = consecutive
            state["drift_status"] = drift_payload
            summary["drift_status"] = drift_payload
            if drift.drift_detected:
                _log_decision("drift_alert", drift_payload)
                logger.warning("Drift monitor alert: %s", drift_payload)
            if consecutive >= 2:
                runtime_overrides = state.setdefault("runtime_overrides", {})
                runtime_overrides["ML_RANKER_MODE"] = "shadow"
                runtime_overrides["ML_RANKER_SHADOW_ONLY"] = True
                runtime_overrides["percentile_action_tightening"] = 0.5
                _log_decision("drift_debounce_confirmed", drift_payload)
                logger.warning("Drift confirmed on consecutive runs; ML override set to shadow.")

            # Threshold-learner hook: persist drift-active flag so the next
            # discovery run picks the conservative profile (Russo & Van Roy
            # 2014 — Thompson sampling with safety-first override).
            if getattr(config, "THRESHOLD_LEARNER_ENABLED", True):
                try:
                    from engine.threshold_learner import load_state as _load_tl_state
                    from engine.threshold_learner import save_state as _save_tl_state
                    tl_state = _load_tl_state()
                    tl_state.drift_active = bool(consecutive >= 2)
                    _save_tl_state(tl_state)
                except Exception as exc:
                    logger.debug("Threshold-learner drift sync failed: %s", exc)
        except Exception as e:
            logger.warning("Drift monitor failed: %s", e)

    try:
        holdings = load_portfolio()
    except Exception as e:
        logger.error("Failed to load portfolio: %s", e)
        summary["error"] = f"portfolio_load_failed: {e}"
        _log_decision("error", {"stage": "load_portfolio", "error": str(e)})
        return summary

    if not holdings:
        logger.warning("Portfolio is empty — nothing to analyse.")
        summary["error"] = "empty_portfolio"
        return summary

    # --- Step 1: Portfolio analysis ---
    logger.info("Running portfolio analysis on %d holdings...", len(holdings))
    try:
        results, risk_data, position_weights = _run_with_timeout(
            _run_portfolio_analysis,
            args=(holdings,),
            timeout_seconds=getattr(config, "PORTFOLIO_ANALYSIS_TIMEOUT", 300),
        )
        summary["portfolio_ran"] = True
        state["last_portfolio_run"] = datetime.now().isoformat()
        _log_decision("portfolio_done", {
            "n_holdings": len(results),
            "risk_score": risk_data.get("risk_score", 0),
        })
    except TimeoutError:
        logger.error("Portfolio analysis timed out. Aborting run.")
        summary["error"] = "portfolio_timeout"
        _log_decision("portfolio_timeout")
        save_state(state)
        return summary
    except Exception as e:
        logger.error("Portfolio analysis failed: %s", e)
        summary["error"] = f"portfolio_failed: {e}"
        _log_decision("error", {"stage": "portfolio_analysis", "error": str(e)})
        save_state(state)
        return summary

    _check_timeout("portfolio_analysis")

    # --- Step 2: VIX regime ---
    vix_regime = _get_regime()
    logger.info("VIX regime: %s (level %.1f, percentile %.0f%%)",
                vix_regime["regime_label"], vix_regime["vix_level"],
                vix_regime["vix_percentile"])

    # --- Step 2b: Portfolio optimisation ---
    portfolio_alloc = None
    try:
        from engine.portfolio_optimizer import optimize_portfolio as _opt_portfolio
        portfolio_alloc = _opt_portfolio(results, holdings, risk_data, position_weights, vix_regime)
        logger.info("Portfolio optimiser: expected return %.1f%%, vol %.1f%%, Sharpe %.2f, turnover %.1f%%",
                     portfolio_alloc.portfolio_expected_return * 100,
                     portfolio_alloc.portfolio_volatility * 100,
                     portfolio_alloc.portfolio_sharpe,
                     portfolio_alloc.turnover * 100)
        if portfolio_alloc.rebalance_trades:
            for t in portfolio_alloc.rebalance_trades:
                logger.info("  Rebalance: %s %s %+.1f%% (£%.0f)",
                             t["direction"], t["ticker"], t["delta_pct"], t["trade_value"])
        _log_decision("portfolio_optimised", {
            "expected_return": portfolio_alloc.portfolio_expected_return,
            "volatility": portfolio_alloc.portfolio_volatility,
            "sharpe": portfolio_alloc.portfolio_sharpe,
            "turnover": portfolio_alloc.turnover,
            "n_trades": len(portfolio_alloc.rebalance_trades),
            "method": portfolio_alloc.method,
        })
    except Exception as e:
        logger.warning("Portfolio optimisation failed: %s", e)

    # --- Step 2c: Record portfolio signals for comprehensive backtest ---
    try:
        # Get adaptive weights currently in use (for point-in-time logging)
        _active_weights = None
        try:
            from engine.discovery_backtest import get_adaptive_weights as _get_aw
            _active_weights = _get_aw(source="portfolio", horizon="90d")
        except Exception:
            pass
        if _active_weights is None:
            _active_weights = dict(config.WEIGHTS)

        n_bt = record_portfolio_signals(
            results, position_weights, vix_regime,
            optimizer_alloc=portfolio_alloc,
            pillar_weights=_active_weights,
        )
        if n_bt:
            logger.info("Recorded %d portfolio signals for backtest.", n_bt)
    except Exception as e:
        logger.warning("Failed to record portfolio signals for backtest: %s", e)

    # --- Step 3: Exit intelligence and final risk-adjusted alerts ---
    summary["exit_signals"] = []
    try:
        from engine.exit_engine import assess_exits
        exit_signals = assess_exits(results, holdings)
        for es in exit_signals:
            logger.info("EXIT SIGNAL [%s] %s — %s: %s",
                         es.severity.upper(), es.ticker, es.signal_type, es.message)
            _log_decision("exit_signal", {
                "ticker": es.ticker,
                "type": es.signal_type,
                "severity": es.severity,
                "message": es.message,
                "score": es.current_score,
            })
        reconcile_actions_with_exits(results, exit_signals)
        result_map = {r["ticker"]: r for r in results}
        summary["exit_signals"] = [
            exit_signal_to_dict(e, result_map.get(e.ticker))
            for e in exit_signals
        ]
        for es in exit_signals:
            r = result_map.get(es.ticker)
            if r and r.get("_exit_override"):
                logger.info(
                    "EXIT RECONCILE: %s %s -> %s "
                    "(prior=%.3f, exit_score=%.3f, penalty=%.3f, posterior=%.3f)",
                    es.ticker, r.get("base_action"), r.get("final_action"),
                    r.get("aggregate_score", 0) or 0,
                    es.exit_score,
                    r.get("_exit_penalty", 0) or 0,
                    r.get("_exit_posterior", 0) or 0,
                )
    except Exception as e:
        logger.warning("Exit intelligence failed: %s", e)

    # Final alerts should use the post-risk-adjustment action, not the alpha prior.
    alerts = [r for r in results if r.get("final_action", r.get("action")) in ("SELL", "STRONG SELL")]
    summary["alerts"] = [
        {
            "ticker": a["ticker"],
            "action": a.get("final_action", a.get("action")),
            "base_action": a.get("base_action", a.get("action")),
            "prior_score": a.get("aggregate_score"),
            "posterior_score": a.get("_exit_posterior"),
            "exit_score": a.get("exit_score"),
            "exit_penalty": a.get("_exit_penalty"),
            "current_price": a.get("current_price"),
            "structural_stop_loss": a.get("structural_stop_loss", a.get("stop_loss")),
            "trailing_exit_stop": a.get("trailing_exit_stop"),
        }
        for a in alerts
    ]
    for a in alerts:
        logger.info("FINAL ALERT: %s â€” %s (base: %s, prior=%.3f, posterior=%s)",
                    a["ticker"],
                    a.get("final_action", a.get("action")),
                    a.get("base_action", a.get("action")),
                    a.get("aggregate_score", 0) or 0,
                    a.get("_exit_posterior", "n/a"))
        _log_decision("alert_found", {
            "ticker": a["ticker"],
            "action": a.get("final_action", a.get("action")),
            "base_action": a.get("base_action", a.get("action")),
            "score": a.get("aggregate_score", 0),
            "posterior_score": a.get("_exit_posterior"),
        })

    # --- Paper trading: log SELL signals ---
    if _paper_trading_enabled_for_run(dry_run):
        for a in alerts:
            try:
                _log_paper_signal(
                    ticker=a["ticker"],
                    side="SELL",
                    source="portfolio_alert",
                    signal_price=a.get("current_price", 0),
                    quantity=a.get("quantity"),
                    score=a.get("aggregate_score"),
                    action=a.get("final_action", a.get("action")),
                )
            except Exception as e:
                logger.warning("Paper trading signal log failed for %s: %s", a["ticker"], e)

    _check_timeout("pre_discovery")

    # --- Step 4: Discovery (weekly or forced) ---
    if not portfolio_only:
        run_disc = should_run_discovery(state) or force_discovery
        if run_disc:
            _old_lightweight_n = None
            _old_top_n = None
            _old_component_timeout = None
            _old_per_ticker_timeout = None
            _old_stage5b_timeout = None
            _old_challenge_reserve_n = None
            if discovery_lightweight_n is not None:
                _old_lightweight_n = getattr(config, "DISCOVERY_TOP_N_LIGHTWEIGHT", None)
                config.DISCOVERY_TOP_N_LIGHTWEIGHT = max(1, int(discovery_lightweight_n))
                logger.info(
                    "Discovery bounded mode: DISCOVERY_TOP_N_LIGHTWEIGHT=%d (was %s)",
                    config.DISCOVERY_TOP_N_LIGHTWEIGHT,
                    _old_lightweight_n,
                )
            if discovery_top_n is not None:
                _old_top_n = getattr(config, "DISCOVERY_TOP_N_FULL_SCORE", None)
                config.DISCOVERY_TOP_N_FULL_SCORE = max(1, int(discovery_top_n))
                logger.info(
                    "Discovery bounded mode: DISCOVERY_TOP_N_FULL_SCORE=%d (was %s)",
                    config.DISCOVERY_TOP_N_FULL_SCORE,
                    _old_top_n,
                )
            if discovery_component_timeout is not None:
                _old_component_timeout = getattr(config, "SCORING_COMPONENT_TIMEOUT", None)
                config.SCORING_COMPONENT_TIMEOUT = max(1, int(discovery_component_timeout))
                logger.info(
                    "Discovery bounded mode: SCORING_COMPONENT_TIMEOUT=%d (was %s)",
                    config.SCORING_COMPONENT_TIMEOUT,
                    _old_component_timeout,
                )
            if discovery_per_ticker_timeout is not None:
                _old_per_ticker_timeout = getattr(config, "DISCOVERY_PER_TICKER_TIMEOUT", None)
                config.DISCOVERY_PER_TICKER_TIMEOUT = max(1, int(discovery_per_ticker_timeout))
                logger.info(
                    "Discovery bounded mode: DISCOVERY_PER_TICKER_TIMEOUT=%d (was %s)",
                    config.DISCOVERY_PER_TICKER_TIMEOUT,
                    _old_per_ticker_timeout,
                )
            if discovery_stage5b_timeout is not None:
                _old_stage5b_timeout = getattr(config, "DISCOVERY_STAGE5B_SCORING_TIMEOUT", None)
                config.DISCOVERY_STAGE5B_SCORING_TIMEOUT = max(1, int(discovery_stage5b_timeout))
                logger.info(
                    "Discovery bounded mode: DISCOVERY_STAGE5B_SCORING_TIMEOUT=%d (was %s)",
                    config.DISCOVERY_STAGE5B_SCORING_TIMEOUT,
                    _old_stage5b_timeout,
                )
            if discovery_challenge_reserve_n is not None:
                _old_challenge_reserve_n = getattr(config, "DISCOVERY_CHALLENGE_RESERVE_N", None)
                config.DISCOVERY_CHALLENGE_RESERVE_N = max(0, int(discovery_challenge_reserve_n))
                logger.info(
                    "Discovery bounded mode: DISCOVERY_CHALLENGE_RESERVE_N=%d (was %s)",
                    config.DISCOVERY_CHALLENGE_RESERVE_N,
                    _old_challenge_reserve_n,
                )
            # Refresh the dynamic universe supplement before assembly so
            # ETF-graduated tickers pass liquidity/mcap validation (rather
            # than being injected directly with thin metadata).
            if getattr(config, "DISCOVERY_RECONSTITUTE_UNIVERSE_ENABLED", True):
                if not _dynamic_universe_cache_fresh():
                    _reconstitute_dynamic_universe()
            else:
                logger.info("Dynamic universe reconstitution skipped by config")
            logger.info(
                "Running Global Discovery Engine v4 (multi-lens, %s deep-scored, may take 30-60 min at full size)...",
                getattr(config, "DISCOVERY_TOP_N_FULL_SCORE", "configured"),
            )
            timeout = int(discovery_timeout or getattr(config, "DISCOVERY_TIMEOUT", 900))
            try:
                disc_result = _run_with_timeout(
                    _run_discovery_pipeline,
                    args=(holdings, risk_data),
                    timeout_seconds=timeout,
                )
                if disc_result.error:
                    logger.warning("Discovery completed with error: %s", disc_result.error)
                    _log_decision("discovery_error", {"error": disc_result.error})
                else:
                    n_candidates = len(disc_result.candidates)
                    logger.info("Discovery complete: %d candidates in %.1fs",
                                n_candidates, disc_result.run_time_seconds)

                    # Persist results + record backtest picks (shared with UI button)
                    n_recorded = save_discovery_results(disc_result, state)
                    summary["discovery_recorded"] = n_recorded
                    summary["discovery_strong_buys"] = [
                        c.ticker for c in disc_result.candidates
                        if getattr(c, "action", None) == "STRONG BUY"
                    ]
                    summary["discovery_ran"] = True

                    _log_decision("discovery_done", {
                        "screened": disc_result.screened_count,
                        "after_momentum": disc_result.after_momentum_screen,
                        "after_filter": disc_result.after_quick_filter,
                        "after_corr": disc_result.after_corr_filter,
                        "after_rank": disc_result.after_quick_rank,
                        "fully_scored": disc_result.fully_scored,
                        "final_candidates": n_candidates,
                        "runtime_s": disc_result.run_time_seconds,
                        "stage_timings": getattr(disc_result, "stage_timings", {}),
                    })
            except TimeoutError:
                logger.warning("Discovery timed out after %ds. Using cached results.", timeout)
                _log_decision("discovery_timeout", {
                    "timeout_s": timeout,
                    "discovery_top_n": discovery_top_n,
                })
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                logger.error("Discovery failed: %s\n%s", e, tb)
                _log_decision("discovery_error", {
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": tb[-2000:],
                })
            finally:
                if _old_top_n is not None:
                    config.DISCOVERY_TOP_N_FULL_SCORE = _old_top_n
                if _old_lightweight_n is not None:
                    config.DISCOVERY_TOP_N_LIGHTWEIGHT = _old_lightweight_n
                if _old_component_timeout is not None:
                    config.SCORING_COMPONENT_TIMEOUT = _old_component_timeout
                if _old_per_ticker_timeout is not None:
                    config.DISCOVERY_PER_TICKER_TIMEOUT = _old_per_ticker_timeout
                if _old_stage5b_timeout is not None:
                    config.DISCOVERY_STAGE5B_SCORING_TIMEOUT = _old_stage5b_timeout
                if _old_challenge_reserve_n is not None:
                    config.DISCOVERY_CHALLENGE_RESERVE_N = _old_challenge_reserve_n
        else:
            logger.info("Discovery not scheduled (last run: %s). Using cached results.",
                        state.get("last_discovery_run", "never"))

    _skip_post_discovery = False
    try:
        _check_timeout("post_discovery")
    except TimeoutError:
        logger.warning("Timeout after discovery — saving state with results so far. "
                       "Skipping swap evaluation; will proceed to email/summary.")
        save_state(state)
        _skip_post_discovery = True

    # --- Step 5: Evaluate swap opportunities ---
    if _skip_post_discovery:
        swap_recs = []
        logger.info("Skipping swap evaluation due to timeout.")
    else:
        cached = get_cached_discovery(state) if not portfolio_only else []
        swap_recs = _evaluate_swaps(results, cached, state)
    summary["swap_recs"] = [
        {"candidate": s["candidate"]["ticker"], "delta": s["score_delta"]}
        for s in swap_recs
    ]

    # Persist cooldowns set during swap evaluation
    if swap_recs:
        save_state(state)
        for s in swap_recs:
            logger.info("SWAP: Sell %s (%.3f) -> Buy %s (%.3f), delta +%.3f",
                        s["weakest_ticker"], s["weakest_score"],
                        s["candidate"]["ticker"], s["candidate"]["aggregate_score"],
                        s["score_delta"])

        # --- Paper trading: log both legs of each swap ---
        if _paper_trading_enabled_for_run(dry_run):
            for s in swap_recs:
                cand = s["candidate"]
                # SELL leg — the weak holding being swapped out
                weak_result = next((r for r in results if r["ticker"] == s["weakest_ticker"]), None)
                try:
                    _log_paper_signal(
                        ticker=s["weakest_ticker"],
                        side="SELL",
                        source="discovery_swap",
                        signal_price=weak_result["current_price"] if weak_result else 0,
                        quantity=weak_result.get("quantity") if weak_result else None,
                        score=s["weakest_score"],
                        action="SELL",
                        swap_from=None,
                    )
                except Exception as e:
                    logger.warning("Paper signal (swap SELL) failed for %s: %s", s["weakest_ticker"], e)

                # BUY leg — the discovery candidate (fetch live price for signal)
                try:
                    from utils.data_fetch import get_current_price
                    cand_price = get_current_price(cand["ticker"]) or 0
                    _log_paper_signal(
                        ticker=cand["ticker"],
                        side="BUY",
                        source="discovery_swap",
                        signal_price=cand_price,
                        quantity=weak_result.get("quantity") if weak_result else None,
                        score=cand.get("aggregate_score"),
                        action=cand.get("action", "BUY"),
                        swap_from=s["weakest_ticker"],
                    )
                except Exception as e:
                    logger.warning("Paper signal (swap BUY) failed for %s: %s", cand["ticker"], e)

    # --- Step 6: Build and send email ---
    # Always send when discovery ran (user wants confirmation of completion).
    # Otherwise send only if there are alerts or swap recommendations.
    discovery_ran = summary.get("discovery_ran", False)
    has_exit_signals = bool(summary.get("exit_signals"))
    has_triggers = bool(alerts) or bool(swap_recs) or discovery_ran or has_exit_signals
    email_subject = None
    email_html = None

    if has_triggers:
        logger.info("Building alert email (alerts=%d, swaps=%d)...", len(alerts), len(swap_recs))
        subject, html = build_alert_email(
            results, risk_data, position_weights,
            vix_regime, alerts, swap_recs,
            dry_run=dry_run,
            optimizer_alloc=portfolio_alloc,
            discovery_candidates=get_cached_discovery(state),
            exit_signals=summary.get("exit_signals"),
            discovery_meta=state.get("cached_discovery_meta"),
            artifact_timestamps={
                "portfolio": (state.get("cached_portfolio") or {}).get("timestamp") or state.get("last_portfolio_run"),
                "discovery": state.get("last_discovery_run"),
                "optimizer": (state.get("cached_optimizer") or {}).get("timestamp"),
                "exit": ((state.get("cached_exit_signals") or {}).get("timestamp")),
            },
            discovery_ran=discovery_ran,
        )
        email_subject = subject
        email_html = html

        success = send_email(subject, html, dry_run=dry_run)
        summary["email_sent"] = success

        if dry_run:
            _log_decision("email_dry_run", {
                "subject": subject,
                "html_length": len(html),
                "alerts": len(alerts),
                "swaps": len(swap_recs),
            })
            logger.info("[DRY RUN] Email would be: %s", subject)
        elif success:
            state["last_email_sent"] = datetime.now().isoformat()
            _log_decision("email_sent", {"subject": subject})
        else:
            _log_decision("email_failed", {"subject": subject})
            logger.error("Email send failed — check SMTP credentials and network.")
    else:
        logger.info("No alerts or swap opportunities. Exiting silently.")
        _log_decision("no_action", {
            "n_alerts": 0,
            "n_swaps": 0,
            "weakest_ticker": (sorted(results, key=lambda r: r.get("aggregate_score", 0))[0]["ticker"]
                               if results else None),
            "weakest_score": (sorted(results, key=lambda r: r.get("aggregate_score", 0))[0].get("aggregate_score", 0)
                              if results else None),
        })

    try:
        sanity_report = _write_live_run_sanity_report(
            get_cached_discovery(state),
            dry_run=dry_run,
            discovery_ran=bool(discovery_ran),
            email_sent=bool(summary.get("email_sent")),
            email_subject=email_subject,
            email_html=email_html,
            n_recorded=int(summary.get("discovery_recorded") or 0),
        )
        summary["live_sanity_ok"] = bool(sanity_report.get("ok"))
        summary["live_sanity_warnings"] = sanity_report.get("warnings", [])
        if sanity_report.get("warnings"):
            logger.warning("Live run sanity warnings: %s", "; ".join(sanity_report["warnings"]))
        else:
            logger.info("Live run sanity check passed.")
    except Exception as e:
        summary["live_sanity_ok"] = False
        logger.warning("Failed to write live run sanity report: %s", e)

    # --- Cache all artifacts for instant dashboard load ---
    try:
        state["cached_portfolio"] = {
            "results": results,
            "risk_data": {k: v for k, v in risk_data.items() if k != "correlation_matrix"},
            "position_weights": position_weights,
            "vix_regime": vix_regime,
            "timestamp": datetime.now().isoformat(),
        }
        # Correlation matrix as nested list (numpy → JSON)
        if risk_data.get("correlation_matrix") is not None:
            try:
                cm = risk_data["correlation_matrix"]
                if hasattr(cm, "values"):  # pandas DataFrame
                    state["cached_portfolio"]["correlation_matrix"] = cm.values.tolist()
                elif hasattr(cm, "tolist"):  # numpy array
                    state["cached_portfolio"]["correlation_matrix"] = cm.tolist()
            except Exception:
                pass

        # Portfolio optimizer
        if portfolio_alloc:
            state["cached_optimizer"] = {
                "holdings": [
                    {
                        "ticker": h.ticker, "name": h.name,
                        "current_weight": h.current_weight,
                        "optimal_weight": h.optimal_weight,
                        "rebalance_delta": h.rebalance_delta,
                        "expected_return": h.expected_return,
                        "volatility": h.volatility,
                        "sharpe_contribution": h.sharpe_contribution,
                        "sector": h.sector, "currency": h.currency,
                        "action": h.action,
                        "aggregate_score": h.aggregate_score,
                        "fx_cost_if_rebalanced": h.fx_cost_if_rebalanced,
                    }
                    for h in portfolio_alloc.holdings
                ],
                "portfolio_expected_return": portfolio_alloc.portfolio_expected_return,
                "portfolio_volatility": portfolio_alloc.portfolio_volatility,
                "portfolio_sharpe": portfolio_alloc.portfolio_sharpe,
                "risk_free_rate": portfolio_alloc.risk_free_rate,
                "method": portfolio_alloc.method,
                "sector_weights": portfolio_alloc.sector_weights,
                "fx_exposure": portfolio_alloc.fx_exposure,
                "turnover": portfolio_alloc.turnover,
                "rebalance_trades": portfolio_alloc.rebalance_trades,
                "warnings": portfolio_alloc.warnings,
                "timestamp": datetime.now().isoformat(),
            }

        # Exit signals
        if summary.get("exit_signals"):
            state["cached_exit_signals"] = {
                "signals": summary["exit_signals"],
                "timestamp": datetime.now().isoformat(),
            }

        logger.info("Cached all artifacts for dashboard instant load.")
    except Exception as e:
        logger.warning("Failed to cache artifacts: %s", e)
    finally:
        # Always save state — ensures discovery results, cooldowns, and
        # portfolio cache are persisted even if artifact caching partially fails
        save_state(state)

    # Self-monitoring: analyze recent telemetry and emit recommendations.
    # Reads the decision log just written above; never mutates config.
    if getattr(config, "AUTO_TUNE_ENABLED", True):
        try:
            from engine import auto_tune
            report = auto_tune.run(
                window_days=int(getattr(config, "AUTO_TUNE_WINDOW_DAYS", 30)),
            )
            summary_line = auto_tune.summarize(report)
            logger.info(summary_line)
            _log_decision("auto_tune_done", {
                "n_recommendations": len(report.recommendations),
                "params": [r.param for r in report.recommendations],
                "summary": summary_line,
            })
        except Exception as e:
            logger.warning("auto_tune run failed: %s", e)

    elapsed = round(time.time() - start, 1)
    _log_decision("run_complete", {"elapsed_s": elapsed, "summary": summary})
    logger.info("Orchestrator finished in %.1fs", elapsed)

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ISA Portfolio Autonomous Orchestrator (v4.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python daily_orchestrator.py --dry-run --portfolio-only   Quick test, no email
  python daily_orchestrator.py --dry-run --force-discovery  Test full pipeline
  python daily_orchestrator.py                              Production run
""",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log decisions and build email but do not actually send it",
    )
    parser.add_argument(
        "--force-discovery", action="store_true",
        help="Run discovery even if not scheduled (overrides weekly frequency)",
    )
    parser.add_argument(
        "--portfolio-only", action="store_true",
        help="Skip discovery entirely — only evaluate current holdings",
    )
    parser.add_argument(
        "--discovery-lightweight-n", type=int, default=None,
        help="Temporarily cap Stage 5b medium-cost scoring to N candidates for bounded verification runs",
    )
    parser.add_argument(
        "--discovery-top-n", type=int, default=None,
        help="Temporarily cap Stage 6 full scoring to N candidates for bounded verification runs",
    )
    parser.add_argument(
        "--discovery-timeout", type=int, default=None,
        help="Temporarily override discovery timeout in seconds for this run",
    )
    parser.add_argument(
        "--discovery-component-timeout", type=int, default=None,
        help="Temporarily override per-component scoring timeout in seconds",
    )
    parser.add_argument(
        "--discovery-per-ticker-timeout", type=int, default=None,
        help="Temporarily override per-ticker deep-scoring timeout in seconds",
    )
    parser.add_argument(
        "--discovery-stage5b-timeout", type=int, default=None,
        help="Temporarily override Stage 5b quick-rank scoring timeout in seconds",
    )
    parser.add_argument(
        "--discovery-challenge-reserve-n", type=int, default=None,
        help="Temporarily override challenge reserve size for bounded discovery verification",
    )
    parser.add_argument(
        "--backfill-actions", action="store_true",
        help="Retroactively re-label signal_backtest action column for replay_pit_v1 rows "
             "using the current threshold function, then exit (no orchestrator run).",
    )
    args = parser.parse_args()

    # One-shot maintenance: action-label backfill for cold-start unlock.
    # Runs without acquiring the orchestrator lock since it's a pure DB UPDATE.
    if args.backfill_actions:
        _setup_logging(dry_run=False)
        from engine.discovery_backtest import backfill_panel_actions, _recompute_all_stats
        logger.info("Starting --backfill-actions one-shot maintenance...")
        summary = backfill_panel_actions(dry_run=False)
        logger.info("Backfill summary: %s", summary)
        logger.info("Triggering _recompute_all_stats to refresh action_calibration / pillar_effectiveness...")
        _recompute_all_stats()
        logger.info("--backfill-actions complete.")
        sys.exit(0)

    _setup_logging(dry_run=args.dry_run)

    if args.dry_run:
        logger.info("=== DRY RUN MODE — no emails will be sent ===")

    lock = None
    try:
        lock = _acquire_orchestrator_lock(
            dry_run=args.dry_run,
            force_discovery=args.force_discovery,
            portfolio_only=args.portfolio_only,
        )
        if lock.reclaimed:
            _log_decision("lock_reclaimed", {
                "path": str(lock.path),
                "previous": lock.reclaimed,
            })
        logger.info("Acquired orchestrator lock: %s", lock.path)

        summary = run_orchestrator(
            dry_run=args.dry_run,
            force_discovery=args.force_discovery,
            portfolio_only=args.portfolio_only,
            discovery_lightweight_n=args.discovery_lightweight_n,
            discovery_top_n=args.discovery_top_n,
            discovery_timeout=args.discovery_timeout,
            discovery_component_timeout=args.discovery_component_timeout,
            discovery_per_ticker_timeout=args.discovery_per_ticker_timeout,
            discovery_stage5b_timeout=args.discovery_stage5b_timeout,
            discovery_challenge_reserve_n=args.discovery_challenge_reserve_n,
        )
    except OrchestratorAlreadyRunning as e:
        md = e.metadata or {}
        logger.warning(
            "Another orchestrator run is already active (pid=%s, started=%s, host=%s). Exiting without running.",
            md.get("pid", "unknown"),
            md.get("started_at", "unknown"),
            md.get("hostname", "unknown"),
        )
        _log_decision("run_skipped_lock", {
            "active_run": md,
            "argv": list(sys.argv),
        })
        sys.exit(0)
    except KeyboardInterrupt:
        logger.warning("Orchestrator interrupted by user (SIGINT/Ctrl+C)")
        _log_decision("interrupted", {"reason": "KeyboardInterrupt"})
        sys.exit(130)
    except Exception as e:
        # Catch-all: no pipeline run should ever crash silently
        import traceback
        tb = traceback.format_exc()
        logger.critical("FATAL: Orchestrator crashed with unhandled exception:\n%s", tb)
        _log_decision("fatal_crash", {
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": tb[-2000:],  # Last 2000 chars of traceback
        })
        sys.exit(2)
    finally:
        if lock is not None:
            lock.release()

    # Exit code: 0 = success, 1 = error
    sys.exit(0 if summary.get("error") is None else 1)


if __name__ == "__main__":
    main()
