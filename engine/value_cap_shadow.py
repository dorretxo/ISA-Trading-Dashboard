"""Shadow diagnostics for valuation-cap and institutional-prior gate changes.

The live action gates stay untouched.  This module recomputes candidate gate
ceilings under narrow counterfactual knobs, then reports which names would move
and how mature replay/discovery labels have behaved in the same buckets.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import config as default_config
from engine.action_gates import build_context, evaluate_candidate


_RANK = {"MANUAL REVIEW": 0, "AVOID": 1, "NEUTRAL": 2, "BUY": 3, "STRONG BUY": 4}


class _ConfigOverlay:
    def __init__(self, base: Any, overrides: dict[str, Any]):
        self._base = base
        self._overrides = dict(overrides)

    def __getattr__(self, item: str) -> Any:
        if item in self._overrides:
            return self._overrides[item]
        return getattr(self._base, item)


def _get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def _label(value: Any, default: str = "UNKNOWN") -> str:
    out = str(value or default).upper()
    return out if out in _RANK else default


def _as_object(row: Any) -> Any:
    if isinstance(row, Mapping):
        payload = dict(row)
        if payload.get("qmj_factor_score") is None:
            try:
                from engine.factors import compute_factor_scores_from_result

                scores = compute_factor_scores_from_result(payload)
                if scores.get("qmj_factor_score") is not None:
                    payload["qmj_factor_score"] = scores.get("qmj_factor_score")
                    payload["qmj_component_count"] = scores.get("qmj_component_count")
                    payload["_qmj_shadow_recomputed"] = True
            except Exception:
                pass
        return SimpleNamespace(**payload)
    return row


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _active_profile(candidates: Iterable[Any], cfg: Any) -> str:
    counts: Counter[str] = Counter()
    for candidate in candidates:
        profile = str(_get(candidate, "threshold_profile", "") or "").lower()
        if profile and profile != "unknown":
            counts[profile] += 1
    if counts:
        return counts.most_common(1)[0][0]
    return str(getattr(cfg, "THRESHOLD_LEARNER_PROFILE", "moderate") or "moderate").lower()


def _profile_overlay(cfg: Any, profile: str) -> Any:
    try:
        from engine.threshold_learner import PROFILES
    except Exception:
        PROFILES = {}
    if profile in PROFILES:
        return _ConfigOverlay(cfg, dict(PROFILES.get(profile, {})))
    return cfg


def _prior_passes(candidate: Any, cfg: Any, *, coverage_min: float | None = None) -> bool:
    percentile = _finite(_get(candidate, "institutional_prior_percentile")) or 0.0
    confidence = _finite(_get(candidate, "institutional_prior_confidence")) or 0.0
    coverage = _finite(_get(candidate, "institutional_prior_coverage")) or 0.0
    return (
        percentile >= float(getattr(cfg, "INSTITUTIONAL_PRIOR_STRONG_BUY_PERCENTILE", 0.90))
        and confidence >= float(getattr(cfg, "INSTITUTIONAL_PRIOR_MIN_CONFIDENCE", 0.55))
        and coverage >= float(
            coverage_min
            if coverage_min is not None
            else getattr(cfg, "INSTITUTIONAL_PRIOR_MIN_COVERAGE", 0.45)
        )
    )


def _coverage_source(row: Any) -> str:
    source = str(_get(row, "institutional_prior_coverage_source") or "").strip()
    if source:
        return source
    if _finite(_get(row, "institutional_prior_coverage")) is not None:
        return "legacy_cached_prior_pipeline"
    return "unavailable"


def _serialize_eval(ev: dict) -> dict:
    return {
        "ceiling": ev.get("ceiling"),
        "reasons": [str(x) for x in _as_list(ev.get("reasons"))],
        "flags": dict(ev.get("flags") or {}),
    }


def annotate_value_cap_shadow(
    candidates: Iterable[Any],
    *,
    config_module=None,
) -> list[dict]:
    """Return per-candidate counterfactual annotations without mutating input."""
    cfg = config_module or default_config
    raw_candidates = list(candidates or [])
    if not raw_candidates:
        return []

    objects = [_as_object(candidate) for candidate in raw_candidates]
    profile = _active_profile(raw_candidates, cfg)
    gate_cfg = _profile_overlay(cfg, profile)
    valuation_cfg = _ConfigOverlay(
        gate_cfg,
        {
            "STRONG_BUY_VALUATION_QUALITY_OVERRIDE_ENABLED": True,
        },
    )

    current_ctx = build_context(objects, config_module=gate_cfg)
    valuation_ctx = build_context(objects, config_module=valuation_cfg)
    prior_cov_current = float(getattr(gate_cfg, "INSTITUTIONAL_PRIOR_MIN_COVERAGE", 0.45))
    prior_cov_shadow = float(getattr(gate_cfg, "INSTITUTIONAL_PRIOR_MIN_COVERAGE_SHADOW", 0.42))

    annotations: list[dict] = []
    for raw, obj in zip(raw_candidates, objects):
        ticker = str(_get(raw, "ticker", "") or "").upper()
        current_eval = evaluate_candidate(obj, context=current_ctx, config_module=gate_cfg)
        valuation_eval = evaluate_candidate(obj, context=valuation_ctx, config_module=valuation_cfg)
        stored_ceiling = _label(_get(raw, "action_gate_ceiling"), default=current_eval.get("ceiling") or "UNKNOWN")
        current_ceiling = _label(current_eval.get("ceiling"), default=stored_ceiling)
        valuation_ceiling = _label(valuation_eval.get("ceiling"), default=current_ceiling)

        current_prior_pass = _prior_passes(raw, gate_cfg, coverage_min=prior_cov_current)
        coverage_prior_pass = _prior_passes(raw, gate_cfg, coverage_min=prior_cov_shadow)
        current_gate_clear = current_ceiling == "STRONG BUY"
        valuation_gate_clear = valuation_ceiling == "STRONG BUY"
        current_gate_prior_clear = current_gate_clear and current_prior_pass
        valuation_a_clear = valuation_gate_clear and current_prior_pass
        prior_b_clear = current_gate_clear and coverage_prior_pass
        combined_clear = valuation_gate_clear and coverage_prior_pass

        valuation_rank_gain = _RANK.get(valuation_ceiling, 0) > _RANK.get(current_ceiling, 0)
        valuation_would_clear_gate = not current_gate_clear and valuation_gate_clear
        prior_coverage_gain = (not current_prior_pass) and coverage_prior_pass
        combined_gain = (not current_gate_prior_clear) and combined_clear

        if combined_gain and valuation_would_clear_gate and prior_coverage_gain:
            primary_bucket = "combined_valuation_and_prior_gain"
        elif valuation_would_clear_gate:
            primary_bucket = "valuation_gate_gain"
        elif prior_coverage_gain:
            primary_bucket = "prior_coverage_gain"
        elif valuation_rank_gain:
            primary_bucket = "valuation_ceiling_gain_not_strong"
        else:
            primary_bucket = "unchanged"

        qmj_pct = current_ctx.qmj_percentiles.get(str(_get(obj, "ticker", "") or ""))
        gpa_pct = current_ctx.gpa_percentiles.get(str(_get(obj, "ticker", "") or ""))
        sector = str(_get(raw, "sector", "") or "Unknown")
        row = {
            "ticker": ticker,
            "run_date": _get(raw, "run_date"),
            "source": _get(raw, "source"),
            "action": _get(raw, "action"),
            "sector": sector,
            "threshold_profile": profile,
            "final_rank": _get(raw, "final_rank"),
            "aggregate_score": _get(raw, "aggregate_score"),
            "sb_score": _get(raw, "sb_score"),
            "ready_contract_core_status": _get(raw, "ready_contract_core_status"),
            "ready_contract_status": _get(raw, "ready_contract_status"),
            "entry_stance": _get(raw, "entry_stance"),
            "ev_ebit": _get(raw, "ev_ebit"),
            "pe_forward": _get(raw, "pe_forward", _get(raw, "pe_ratio")),
            "revenue_growth": _get(raw, "revenue_growth"),
            "sector_median_revenue_growth": current_ctx.sector_median_revenue_growth.get(sector),
            "f_score": _get(raw, "f_score"),
            "qmj_factor_score": _get(obj, "qmj_factor_score", _get(raw, "qmj_factor_score")),
            "qmj_shadow_recomputed": bool(_get(obj, "_qmj_shadow_recomputed", False)),
            "qmj_percentile": qmj_pct,
            "gpa_percentile": gpa_pct,
            "stored_action_gate_ceiling": stored_ceiling,
            "current": _serialize_eval(current_eval),
            "valuation_a": _serialize_eval(valuation_eval),
            "current_prior_pass": current_prior_pass,
            "coverage_b_prior_pass": coverage_prior_pass,
            "institutional_prior_percentile": _get(raw, "institutional_prior_percentile"),
            "institutional_prior_confidence": _get(raw, "institutional_prior_confidence"),
            "institutional_prior_coverage": _get(raw, "institutional_prior_coverage"),
            "institutional_prior_coverage_source": _coverage_source(raw),
            "current_gate_prior_clear": current_gate_prior_clear,
            "valuation_a_gate_prior_clear": valuation_a_clear,
            "prior_b_gate_prior_clear": prior_b_clear,
            "combined_a_b_gate_prior_clear": combined_clear,
            "valuation_rank_gain": valuation_rank_gain,
            "valuation_would_clear_gate": valuation_would_clear_gate,
            "prior_coverage_gain": prior_coverage_gain,
            "combined_gain": combined_gain,
            "primary_bucket": primary_bucket,
            "would_change": valuation_rank_gain or prior_coverage_gain or combined_gain,
        }
        annotations.append(row)
    return annotations


def _sort_key(row: dict) -> tuple[float, float, float]:
    return (
        _finite(row.get("sb_score")) or -999.0,
        _finite(row.get("aggregate_score")) or -999.0,
        _finite(row.get("final_rank")) or -999.0,
    )


def _summarise_annotations(rows: list[dict]) -> dict:
    return {
        "total": len(rows),
        "primary_bucket_counts": dict(Counter(str(r.get("primary_bucket")) for r in rows)),
        "current_gate_prior_clear": sum(1 for r in rows if r.get("current_gate_prior_clear")),
        "valuation_a_gate_prior_clear": sum(1 for r in rows if r.get("valuation_a_gate_prior_clear")),
        "prior_b_gate_prior_clear": sum(1 for r in rows if r.get("prior_b_gate_prior_clear")),
        "combined_a_b_gate_prior_clear": sum(1 for r in rows if r.get("combined_a_b_gate_prior_clear")),
        "valuation_rank_gains": sum(1 for r in rows if r.get("valuation_rank_gain")),
        "valuation_gate_gains": sum(1 for r in rows if r.get("valuation_would_clear_gate")),
        "prior_coverage_gains": sum(1 for r in rows if r.get("prior_coverage_gain")),
        "combined_gains": sum(1 for r in rows if r.get("combined_gain")),
    }


def _outcome_summary(rows: list[dict], bucket_key: str) -> list[dict]:
    buckets: dict[str, dict] = {}
    sectors: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        bucket = str(row.get(bucket_key) or "UNKNOWN")
        item = buckets.setdefault(
            bucket,
            {
                "bucket": bucket,
                "n": 0,
                "wins": 0,
                "losses": 0,
                "flats": 0,
                "tb_returns": [],
                "stop_hits": 0,
                "target_hits": 0,
            },
        )
        item["n"] += 1
        label = int(_finite(row.get("tb_label")) or 0)
        if label > 0:
            item["wins"] += 1
        elif label < 0:
            item["losses"] += 1
        else:
            item["flats"] += 1
        ret = _finite(row.get("tb_return"))
        if ret is not None:
            item["tb_returns"].append(ret)
        if int(_finite(row.get("stop_hit")) or 0):
            item["stop_hits"] += 1
        if int(_finite(row.get("target_hit")) or 0):
            item["target_hits"] += 1
        sectors[bucket][str(row.get("sector") or "Unknown")] += 1

    out: list[dict] = []
    for bucket, item in buckets.items():
        decisive = item["wins"] + item["losses"]
        returns = item.pop("tb_returns")
        top_sector = sectors[bucket].most_common(1)
        item["hit_rate"] = round(item["wins"] / decisive, 4) if decisive else None
        item["avg_tb_return"] = round(sum(returns) / len(returns), 4) if returns else None
        item["worst_tb_return"] = round(min(returns), 4) if returns else None
        item["stop_hit_rate"] = round(item["stop_hits"] / item["n"], 4) if item["n"] else None
        item["target_hit_rate"] = round(item["target_hits"] / item["n"], 4) if item["n"] else None
        item["top_sector"] = top_sector[0][0] if top_sector else None
        item["top_sector_share"] = round(top_sector[0][1] / item["n"], 4) if top_sector and item["n"] else None
        out.append(item)
    out.sort(key=lambda r: (r["n"], r["bucket"]), reverse=True)
    return out


def _historical_shadow_outcomes(cfg: Any) -> dict:
    try:
        from engine.discovery_backtest import init_backtest_db
        from engine.paper_trading import _connect

        init_backtest_db()
        with _connect() as conn:
            valid = {str(row[1]) for row in conn.execute("PRAGMA table_info(signal_backtest)").fetchall()}
            wanted = [
                "ticker", "run_date", "source", "action", "sector", "final_rank",
                "aggregate_score", "ev_ebit", "pe_ratio", "revenue_growth",
                "f_score", "qmj_factor_score", "gpa_score",
                "institutional_prior_percentile", "institutional_prior_confidence",
                "institutional_prior_coverage", "ready_contract_status",
                "ready_contract_core_status", "entry_stance", "action_gate_ceiling",
                "threshold_profile", "tb_label", "tb_return", "stop_hit", "target_hit",
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
                    WHERE source IN ('discovery', 'replay_pit_v1')
                      AND tb_label IS NOT NULL
                    ORDER BY run_date DESC
                    LIMIT 5000
                    """
                ).fetchall()
            ]
    except Exception as exc:
        return {"available": False, "reason": str(exc)}

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("source") or ""), str(row.get("run_date") or "")[:10])].append(row)

    annotations: list[dict] = []
    for group_rows in grouped.values():
        annotations.extend(annotate_value_cap_shadow(group_rows, config_module=cfg))

    by_ticker_date = {
        (str(row.get("ticker") or "").upper(), str(row.get("run_date") or "")): row
        for row in rows
    }
    enriched: list[dict] = []
    for ann in annotations:
        source_row = by_ticker_date.get((ann.get("ticker"), str(ann.get("run_date") or ""))) or {}
        # annotate_value_cap_shadow does not carry run_date, so fall back by ticker
        if not source_row:
            source_row = next((r for r in rows if str(r.get("ticker") or "").upper() == ann.get("ticker")), {})
        merged = {**source_row, **ann}
        merged["valuation_a_bucket"] = (
            "valuation_a_gain" if ann.get("valuation_would_clear_gate")
            else "valuation_a_no_gain"
        )
        merged["prior_b_bucket"] = (
            "prior_b_gain" if ann.get("prior_coverage_gain")
            else "prior_b_no_gain"
        )
        merged["combined_a_b_bucket"] = (
            "combined_a_b_gain" if ann.get("combined_gain")
            else "combined_a_b_no_gain"
        )
        enriched.append(merged)

    return {
        "available": True,
        "sample": len(enriched),
        "uses": "source IN ('discovery','replay_pit_v1') rows with mature triple-barrier labels",
        "by_primary_bucket": _outcome_summary(enriched, "primary_bucket"),
        "by_valuation_a": _outcome_summary(enriched, "valuation_a_bucket"),
        "by_prior_b": _outcome_summary(enriched, "prior_b_bucket"),
        "by_combined_a_b": _outcome_summary(enriched, "combined_a_b_bucket"),
    }


def _historical_shadow_rows(cfg: Any, *, limit: int = 5000) -> list[dict]:
    try:
        from engine.discovery_backtest import init_backtest_db
        from engine.paper_trading import _connect

        init_backtest_db()
        with _connect() as conn:
            valid = {str(row[1]) for row in conn.execute("PRAGMA table_info(signal_backtest)").fetchall()}
            wanted = [
                "ticker", "run_date", "source", "action", "sector", "final_rank",
                "aggregate_score", "ev_ebit", "pe_ratio", "revenue_growth",
                "f_score", "qmj_factor_score", "gpa_score",
                "institutional_prior_percentile", "institutional_prior_confidence",
                "institutional_prior_coverage", "institutional_prior_coverage_source",
                "ready_contract_status", "ready_contract_core_status", "entry_stance",
                "action_gate_ceiling", "threshold_profile", "tb_label", "tb_return",
                "stop_hit", "target_hit",
            ]
            columns = [c for c in wanted if c in valid]
            if "tb_label" not in columns:
                return []
            rows = [
                dict(zip(columns, tuple(row)))
                for row in conn.execute(
                    f"""
                    SELECT {', '.join(columns)}
                    FROM signal_backtest
                    WHERE source IN ('discovery', 'replay_pit_v1')
                      AND tb_label IS NOT NULL
                    ORDER BY run_date DESC
                    LIMIT {int(limit)}
                    """
                ).fetchall()
            ]
    except Exception:
        return []

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("source") or ""), str(row.get("run_date") or "")[:10])].append(row)
    annotations: list[dict] = []
    by_key = {
        (str(row.get("ticker") or "").upper(), str(row.get("source") or ""), str(row.get("run_date") or "")): row
        for row in rows
    }
    for group_rows in grouped.values():
        for ann in annotate_value_cap_shadow(group_rows, config_module=cfg):
            source_row = by_key.get((
                str(ann.get("ticker") or "").upper(),
                str(ann.get("source") or ""),
                str(ann.get("run_date") or ""),
            ), {})
            annotations.append({**source_row, **ann})
    return annotations


def _summarise_analogue_rows(rows: list[dict]) -> dict:
    returns = [_finite(row.get("tb_return")) for row in rows]
    returns = [ret for ret in returns if ret is not None]
    wins = 0
    losses = 0
    flats = 0
    sectors: Counter[str] = Counter()
    for row in rows:
        label = int(_finite(row.get("tb_label")) or 0)
        if label > 0:
            wins += 1
        elif label < 0:
            losses += 1
        else:
            flats += 1
        sectors[str(row.get("sector") or "Unknown")] += 1
    decisive = wins + losses
    ordered_returns = sorted(returns)
    median_return = None
    if ordered_returns:
        mid = len(ordered_returns) // 2
        median_return = (
            ordered_returns[mid]
            if len(ordered_returns) % 2
            else 0.5 * (ordered_returns[mid - 1] + ordered_returns[mid])
        )
    top_sector = sectors.most_common(1)
    return {
        "n": len(rows),
        "wins": wins,
        "losses": losses,
        "flats": flats,
        "hit_rate": round(wins / decisive, 4) if decisive else None,
        "mean_return": round(sum(returns) / len(returns), 4) if returns else None,
        "median_return": round(median_return, 4) if median_return is not None else None,
        "worst_return": round(min(returns), 4) if returns else None,
        "max_drawdown_proxy": round(min(returns), 4) if returns else None,
        "top_sector": top_sector[0][0] if top_sector else None,
        "top_sector_share": round(top_sector[0][1] / len(rows), 4) if top_sector and rows else None,
        "directional_only": len(rows) < 20,
        "drawdown_note": "max_drawdown_proxy uses worst mature triple-barrier return; intraperiod path drawdown is not stored.",
    }


def _valuation_analogue_slices(targets: list[dict], cfg: Any) -> list[dict]:
    historical = _historical_shadow_rows(cfg)
    if not historical:
        return []
    pe_cap = float(getattr(cfg, "PE_FORWARD_STRONG_BUY_MAX", 30.0))
    pe_max = float(getattr(cfg, "STRONG_BUY_VALUATION_OVERRIDE_MAX_PE_FORWARD", 40.0))
    ev_cap = float(getattr(cfg, "EV_EBIT_STRONG_BUY_MAX", 30.0))
    ev_max = float(getattr(cfg, "STRONG_BUY_VALUATION_OVERRIDE_MAX_EV_EBIT", 35.0))
    out: list[dict] = []
    for target in targets[:10]:
        ticker = str(target.get("ticker") or "").upper()
        target_pe = _finite(target.get("pe_forward"))
        target_ev = _finite(target.get("ev_ebit"))
        rows: list[dict] = []
        for row in historical:
            if str(row.get("ticker") or "").upper() == ticker:
                continue
            if not row.get("valuation_would_clear_gate"):
                continue
            pe = _finite(row.get("pe_forward"))
            ev = _finite(row.get("ev_ebit"))
            comparable_pe = (
                target_pe is None
                or target_pe <= pe_cap
                or (pe is not None and pe_cap < pe <= pe_max)
            )
            comparable_ev = (
                target_ev is None
                or target_ev <= ev_cap
                or (ev is not None and ev_cap < ev <= ev_max)
            )
            if comparable_pe and comparable_ev:
                rows.append(row)
        rows.sort(key=lambda row: str(row.get("run_date") or ""), reverse=True)
        out.append({
            "ticker": ticker,
            "criteria": {
                "exclude_target_ticker": True,
                "uses_historical_asof_features": True,
                "requires_valuation_a_gain": True,
                "pe_forward_range": [pe_cap, pe_max] if target_pe and target_pe > pe_cap else None,
                "ev_ebit_range": [ev_cap, ev_max] if target_ev and target_ev > ev_cap else None,
            },
            "summary": _summarise_analogue_rows(rows),
            "examples": [
                {
                    "ticker": row.get("ticker"),
                    "run_date": row.get("run_date"),
                    "source": row.get("source"),
                    "sector": row.get("sector"),
                    "pe_forward": row.get("pe_forward"),
                    "ev_ebit": row.get("ev_ebit"),
                    "qmj_percentile": row.get("qmj_percentile"),
                    "f_score": row.get("f_score"),
                    "revenue_growth": row.get("revenue_growth"),
                    "tb_label": row.get("tb_label"),
                    "tb_return": row.get("tb_return"),
                }
                for row in rows[:20]
            ],
        })
    return out


def build_value_cap_shadow_report(
    candidates: Iterable[Any],
    *,
    config_module=None,
    include_historical: bool = True,
) -> dict:
    """Build the value-cap shadow report payload."""
    cfg = config_module or default_config
    annotations = annotate_value_cap_shadow(candidates, config_module=cfg)
    changed = [row for row in annotations if row.get("would_change")]
    changed.sort(key=_sort_key, reverse=True)
    valuation_rows = [row for row in annotations if row.get("valuation_rank_gain")]
    valuation_rows.sort(key=_sort_key, reverse=True)
    valuation_blocked_rows = [
        row for row in annotations
        if any(
            "EV/EBIT" in str(reason) or "Fwd P/E" in str(reason)
            for reason in row.get("current", {}).get("reasons", [])
        )
    ]
    valuation_blocked_rows.sort(key=_sort_key, reverse=True)
    prior_rows = [row for row in annotations if row.get("prior_coverage_gain")]
    prior_rows.sort(key=_sort_key, reverse=True)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "shadow_only",
        "live_impact": False,
        "knobs": {
            "valuation_a": {
                "description": "Narrow EV/EBIT + Fwd P/E quality-growth override",
                "live_enabled": bool(getattr(cfg, "STRONG_BUY_VALUATION_QUALITY_OVERRIDE_ENABLED", False)),
                "shadow_enabled": bool(getattr(cfg, "STRONG_BUY_VALUATION_QUALITY_OVERRIDE_SHADOW", True)),
                "qmj_floor": float(getattr(cfg, "STRONG_BUY_VALUATION_OVERRIDE_QMJ_FLOOR", 0.80)),
                "min_f_score": float(getattr(cfg, "STRONG_BUY_VALUATION_OVERRIDE_MIN_F_SCORE", 7)),
                "min_revenue_growth": float(getattr(cfg, "STRONG_BUY_VALUATION_OVERRIDE_MIN_REVENUE_GROWTH", 0.0)),
                "max_ev_ebit": float(getattr(cfg, "STRONG_BUY_VALUATION_OVERRIDE_MAX_EV_EBIT", 35.0)),
                "max_pe_forward": float(getattr(cfg, "STRONG_BUY_VALUATION_OVERRIDE_MAX_PE_FORWARD", 40.0)),
            },
            "prior_b": {
                "description": "Institutional-prior coverage floor counterfactual",
                "current_min_coverage": float(getattr(cfg, "INSTITUTIONAL_PRIOR_MIN_COVERAGE", 0.45)),
                "shadow_min_coverage": float(getattr(cfg, "INSTITUTIONAL_PRIOR_MIN_COVERAGE_SHADOW", 0.42)),
            },
        },
        "summary": _summarise_annotations(annotations),
        "changed_candidates": changed[:80],
        "valuation_a_candidates": valuation_rows[:80],
        "valuation_blocked_candidates": valuation_blocked_rows[:80],
        "prior_b_candidates": prior_rows[:80],
    }
    if include_historical:
        payload["historical_matured_outcomes"] = _historical_shadow_outcomes(cfg)
        payload["valuation_analogue_slices"] = _valuation_analogue_slices(valuation_rows, cfg)
    return payload
