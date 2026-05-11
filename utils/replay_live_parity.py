"""Shared replay/live parity comparison helpers."""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import config
from engine.canonical_scores import compute_ev_ebit_score, compute_f_score_score, compute_gpa_score
from engine.factors import compute_factor_scores_from_result


def finite_float(value) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def comparable_fields(fields: Iterable[str], valid_columns: set[str] | None = None) -> list[str]:
    excluded = set(getattr(config, "REPLAY_LIVE_PARITY_EXCLUDED_FIELDS", []) or [])
    out: list[str] = []
    seen: set[str] = set()
    for field in fields:
        name = str(field)
        if name in seen or name in excluded:
            continue
        if valid_columns is not None and name not in valid_columns:
            continue
        out.append(name)
        seen.add(name)
    return out


def adaptive_weight_fields(valid_columns: set[str] | None = None) -> list[str]:
    return comparable_fields(
        getattr(
            config,
            "REPLAY_LIVE_PARITY_ADAPTIVE_FIELDS",
            ("technical_score", "return_10d_prior", "return_30d_prior", "return_90d_prior", "vol_20d"),
        ),
        valid_columns,
    )


def field_tolerance(field: str, default: float) -> float:
    overrides = getattr(config, "REPLAY_LIVE_PARITY_FIELD_TOLERANCES", {}) or {}
    try:
        return float(overrides.get(field, default))
    except (TypeError, ValueError):
        return float(default)


def canonicalize_parity_row(row: Mapping) -> dict:
    """Normalize persisted rows before parity comparison.

    Discovery and replay sometimes persist derived factor scores that were
    produced by different call paths. Recomputing them from the stored raw
    inputs makes parity test the underlying feature snapshot instead of stale
    derived columns.
    """
    out = dict(row)
    zero_as_missing = set(getattr(config, "REPLAY_LIVE_PARITY_ZERO_AS_MISSING_FIELDS", []) or [])
    for field in zero_as_missing:
        if finite_float(out.get(field)) == 0.0:
            out[field] = None

    f_score = finite_float(out.get("f_score"))
    if f_score is not None and finite_float(out.get("f_score_score")) is None:
        out["f_score_score"] = compute_f_score_score(f_score)
        out["f_score_gate"] = bool(f_score >= 6)

    gpa = finite_float(out.get("gpa"))
    if gpa is not None and finite_float(out.get("gpa_score")) is None:
        out["gpa_score"] = compute_gpa_score(gpa)

    if finite_float(out.get("ev_ebit_score")) is None:
        ebit_yield = finite_float(out.get("ebit_yield"))
        ev_ebit = finite_float(out.get("ev_ebit"))
        if ebit_yield is None and ev_ebit is not None and ev_ebit > 0:
            ebit_yield = 1.0 / ev_ebit
        if ebit_yield is not None:
            out["ev_ebit_score"] = compute_ev_ebit_score(ebit_yield)

    try:
        factor_scores = compute_factor_scores_from_result(out)
    except Exception:
        factor_scores = {}

    support_fields = {
        "quality_factor_score": (
            "quality_score_fundamental", "gpa_score", "f_score_score", "gpa", "f_score",
        ),
        "qmj_factor_score": (
            "quality_score_fundamental", "gpa_score", "f_score_score", "gpa", "f_score",
            "earnings_stability", "leverage_factor_score",
        ),
        "value_factor_score": (
            "pe_ratio", "peg_ratio", "fcf_yield", "ev_ebit_score", "pb_score", "ps_score",
        ),
        "momentum_factor_score": (
            "momentum_score", "return_30d_prior", "return_90d_prior",
        ),
        "volatility_factor_score": ("vol_20d", "beta_90d", "beta"),
        "bab_factor_score": (
            "beta_90d", "beta", "volatility_factor_score", "vol_20d",
            "downside_vol_60d", "max_dd_252d",
        ),
        "turnover_cost_score": (
            "avg_dollar_volume", "market_cap", "vol_20d", "return_10d_prior", "return_90d_prior",
        ),
        "pead_factor_score": ("earnings_surprises", "estimate_revision", "earnings_beat_rate"),
        "sue_score": ("earnings_surprises",),
        "revision_momentum_3m": ("estimate_revision",),
    }
    for key, value in factor_scores.items():
        supported = any(out.get(field) is not None for field in support_fields.get(key, ()))
        if value is not None and (supported or out.get(key) is None):
            out[key] = value
    return out


def compare_parity_rows(
    live: Mapping,
    replay: Mapping,
    fields: Iterable[str],
    *,
    default_tolerance: float,
) -> dict:
    live_row = canonicalize_parity_row(live)
    replay_row = canonicalize_parity_row(replay)
    comparisons: list[dict] = []
    drift_fields: list[str] = []
    live_missing_fields: list[str] = []
    replay_missing_fields: list[str] = []

    for field in fields:
        live_val = finite_float(live_row.get(field))
        replay_val = finite_float(replay_row.get(field))
        if live_val is None or replay_val is None:
            status = "missing"
            delta = None
            if live_val is None:
                live_missing_fields.append(field)
            if replay_val is None:
                replay_missing_fields.append(field)
        else:
            delta = live_val - replay_val
            status = "drift" if abs(delta) > field_tolerance(field, default_tolerance) else "ok"
            if status == "drift":
                drift_fields.append(field)
        comparisons.append({
            "field": field,
            "live": live_val,
            "replay": replay_val,
            "delta": None if delta is None else round(delta, 4),
            "tolerance": field_tolerance(field, default_tolerance),
            "status": status,
        })

    return {
        "comparisons": comparisons,
        "drift_fields": drift_fields,
        "live_missing_fields": live_missing_fields,
        "replay_missing_fields": replay_missing_fields,
    }
