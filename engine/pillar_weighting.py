"""Shared controls for live pillar-weight learning.

The helpers in this module deliberately treat signed IC as directional
evidence. A negative IC can justify reducing a pillar toward the safety floor;
it should not be transformed into positive allocation pressure.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Mapping

import config

PILLARS: tuple[str, ...] = ("technical", "fundamental", "sentiment", "forecast")


def _finite(value, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def normalise_weights(weights: Mapping[str, float], *, pillars: tuple[str, ...] = PILLARS) -> dict[str, float]:
    values = {p: max(0.0, _finite(weights.get(p), 0.0)) for p in pillars}
    total = sum(values.values())
    if total <= 0:
        equal = 1.0 / max(1, len(pillars))
        return {p: equal for p in pillars}
    return {p: values[p] / total for p in pillars}


def blend_with_default_prior(
    weights: Mapping[str, float],
    *,
    blend: float | None = None,
    pillars: tuple[str, ...] = PILLARS,
) -> dict[str, float]:
    """Blend learned weights toward configured priors before live serving."""
    adaptive_blend = float(
        getattr(config, "ADAPTIVE_WEIGHTS_LIVE_BLEND", 0.60)
        if blend is None
        else blend
    )
    adaptive_blend = max(0.0, min(1.0, adaptive_blend))
    learned = normalise_weights(weights, pillars=pillars)
    prior = normalise_weights(getattr(config, "WEIGHTS", {}), pillars=pillars)
    mixed = {
        pillar: adaptive_blend * learned.get(pillar, 0.0) + (1.0 - adaptive_blend) * prior.get(pillar, 0.0)
        for pillar in pillars
    }
    return normalise_weights(mixed, pillars=pillars)


def positive_ic_allocation(
    ic_by_pillar: Mapping[str, float],
    *,
    min_positive_ic: float = 0.0,
    min_positive_pillars: int = 1,
    pillars: tuple[str, ...] = PILLARS,
) -> dict[str, float] | None:
    """Return normalized positive signed-IC evidence, or None if none exists."""
    raw: dict[str, float] = {}
    positive_count = 0
    for pillar in pillars:
        ic = _finite(ic_by_pillar.get(pillar), 0.0)
        if ic <= 0:
            raw[pillar] = 0.0
        elif min_positive_ic > 0:
            positive_count += 1
            raw[pillar] = max(ic, min_positive_ic)
        else:
            positive_count += 1
            raw[pillar] = ic
    if positive_count < max(1, int(min_positive_pillars)) or sum(raw.values()) <= 0:
        return None
    return normalise_weights(raw, pillars=pillars)


def _horizon_is_long(horizon: str | None) -> bool:
    if horizon is None:
        return True
    horizon_norm = str(horizon).strip().lower()
    long_horizons = tuple(str(v).lower() for v in getattr(config, "PILLAR_WEIGHT_LONG_HORIZONS", ("30d", "60d", "90d", "multi")))
    return horizon_norm in long_horizons


def _bounded_normalise(
    weights: Mapping[str, float],
    *,
    lower: Mapping[str, float],
    upper: Mapping[str, float],
    pillars: tuple[str, ...] = PILLARS,
) -> dict[str, float]:
    lows = {p: max(0.0, _finite(lower.get(p), 0.0)) for p in pillars}
    highs = {p: max(lows[p], _finite(upper.get(p), 1.0)) for p in pillars}
    low_sum = sum(lows.values())
    high_sum = sum(highs.values())
    if low_sum > 1.0:
        return normalise_weights(lows, pillars=pillars)
    if high_sum < 1.0:
        return normalise_weights(highs, pillars=pillars)

    remaining = set(pillars)
    fixed: dict[str, float] = {}
    raw = {p: max(0.0, _finite(weights.get(p), 0.0)) for p in pillars}

    for _ in range(len(pillars) + 1):
        target = 1.0 - sum(fixed.values())
        if not remaining:
            break
        raw_total = sum(raw[p] for p in remaining)
        if raw_total <= 0:
            equal = target / len(remaining)
            proposal = {p: equal for p in remaining}
        else:
            proposal = {p: target * raw[p] / raw_total for p in remaining}

        violations = []
        for pillar, value in proposal.items():
            if value < lows[pillar]:
                violations.append((pillar, lows[pillar]))
            elif value > highs[pillar]:
                violations.append((pillar, highs[pillar]))
        if not violations:
            fixed.update(proposal)
            break
        for pillar, value in violations:
            fixed[pillar] = value
            remaining.remove(pillar)

    for pillar in remaining:
        fixed.setdefault(pillar, lows[pillar])
    total = sum(fixed.values())
    if total <= 0:
        return normalise_weights(lows, pillars=pillars)
    if abs(total - 1.0) > 1e-8:
        # Last-resort numerical repair. The iterative allocation above should
        # already sum to one when the bounds are feasible; keep the repair tiny
        # so it cannot knowingly breach configured caps.
        slack = {p: max(0.0, highs[p] - fixed.get(p, 0.0)) for p in pillars}
        slack_total = sum(slack.values())
        delta = 1.0 - total
        if delta > 0 and slack_total > 0:
            for pillar in pillars:
                fixed[pillar] = fixed.get(pillar, 0.0) + delta * slack[pillar] / slack_total
        elif delta < 0:
            reducible = {p: max(0.0, fixed.get(p, 0.0) - lows[p]) for p in pillars}
            reducible_total = sum(reducible.values())
            if reducible_total > 0:
                for pillar in pillars:
                    fixed[pillar] = fixed.get(pillar, 0.0) + delta * reducible[pillar] / reducible_total
    return {p: max(lows[p], min(highs[p], float(fixed.get(p, lows[p])))) for p in pillars}


def apply_weight_guardrails(
    weights: Mapping[str, float],
    *,
    horizon: str | None = "90d",
    ic_by_pillar: Mapping[str, float] | None = None,
    pillars: tuple[str, ...] = PILLARS,
) -> dict[str, float]:
    """Apply floor/cap constraints to learned pillar weights."""
    floor = float(getattr(config, "PILLAR_WEIGHT_MIN_FLOOR", 0.03))
    max_single = float(getattr(config, "PILLAR_WEIGHT_MAX_SINGLE", 0.55))
    nonpositive_ic_max = max(floor, float(getattr(config, "PILLAR_WEIGHT_NONPOSITIVE_IC_MAX", max_single)))
    lower = {p: floor for p in pillars}
    upper = {p: max_single for p in pillars}

    positive_pillars: list[str] = []
    if ic_by_pillar is not None:
        for pillar in pillars:
            ic = _finite(ic_by_pillar.get(pillar), 0.0)
            if ic > 0:
                positive_pillars.append(pillar)
            else:
                upper[pillar] = min(upper[pillar], nonpositive_ic_max)

    if "forecast" in upper:
        upper["forecast"] = min(upper["forecast"], float(getattr(config, "PILLAR_WEIGHT_FORECAST_MAX", 0.30)))
        if ic_by_pillar is not None and _finite(ic_by_pillar.get("forecast"), 0.0) <= 0:
            upper["forecast"] = min(
                upper["forecast"],
                max(floor, float(getattr(config, "PILLAR_WEIGHT_FORECAST_MAX_WITH_NEGATIVE_IC", floor))),
            )
    if "sentiment" in upper and _horizon_is_long(horizon):
        upper["sentiment"] = min(upper["sentiment"], float(getattr(config, "PILLAR_WEIGHT_SENTIMENT_LONG_MAX", 0.08)))

    # If negative-IC caps and long-horizon sentiment/forecast caps make the box
    # infeasible, relax generic non-positive caps before breaking the single
    # pillar concentration cap. This keeps a passing adaptive model from
    # becoming a one-pillar bet.
    if ic_by_pillar is not None and sum(upper.values()) < 1.0 and positive_pillars:
        hard_capped = set()
        if "sentiment" in upper and _horizon_is_long(horizon):
            hard_capped.add("sentiment")
        if "forecast" in upper:
            hard_capped.add("forecast")
        relaxable = [
            p for p in pillars
            if p not in hard_capped and _finite(ic_by_pillar.get(p), 0.0) <= 0
        ]
        if not relaxable:
            relaxable = [p for p in pillars if p not in hard_capped]
        shortfall = 1.0 - sum(upper.values())
        for pillar in relaxable:
            upper[pillar] = min(1.0, upper[pillar] + shortfall / max(1, len(relaxable)))

    return {p: round(v, 4) for p, v in _bounded_normalise(weights, lower=lower, upper=upper, pillars=pillars).items()}


def pillar_parity_gate() -> tuple[bool, list[str], dict]:
    """Return whether replay/live parity is healthy enough for adaptive weights."""
    if not bool(getattr(config, "ADAPTIVE_WEIGHTS_GATE_ENABLED", True)):
        return True, [], {"enabled": False}
    if not bool(getattr(config, "ADAPTIVE_WEIGHTS_GATE_REQUIRE_PARITY", True)):
        return True, [], {"enabled": True, "parity_required": False}

    path = Path(getattr(config, "REPLAY_LIVE_PARITY_REPORT_PATH", "feature_cache/replay_live_parity_report.json"))
    if not path.exists():
        reason = f"parity report missing: {path}"
        return False, [reason], {"enabled": True, "available": False, "reason": reason}
    try:
        root_report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        reason = f"parity report unreadable: {exc}"
        return False, [reason], {"enabled": True, "available": False, "reason": reason}
    if not isinstance(root_report, dict):
        reason = "parity report must be a JSON object"
        return False, [reason], {"enabled": True, "available": False, "reason": reason}
    report = root_report.get("adaptive_weight_parity") if isinstance(root_report, dict) else None
    if not isinstance(report, dict):
        report = root_report

    blockers: list[str] = []
    if report.get("available") is False:
        blockers.append(f"parity report unavailable: {report.get('reason') or 'unknown'}")

    sample = int(report.get("sample") or 0)
    available_pairs = int(report.get("available_pairs") or 0)
    missing_replay = int(report.get("missing_replay") or 0)
    stale_replay = int(report.get("stale_replay") or 0)
    drifted = int(report.get("drifted_tickers") or 0)
    denom = sample or max(available_pairs + missing_replay + stale_replay, available_pairs, 1)
    pair_denom = max(available_pairs, 1)

    min_available = float(getattr(config, "ML_RANKER_PARITY_MIN_AVAILABLE_RATIO", 0.70))
    max_missing = float(getattr(config, "ML_RANKER_PARITY_MAX_MISSING_REPLAY_RATIO", 0.20))
    max_stale = float(getattr(config, "ML_RANKER_PARITY_MAX_STALE_RATIO", 0.10))
    max_drifted = float(getattr(config, "ML_RANKER_PARITY_MAX_DRIFTED_PAIR_RATIO", 0.25))
    available_ratio = available_pairs / max(denom, 1)
    missing_ratio = missing_replay / max(denom, 1)
    stale_ratio = stale_replay / max(denom, 1)
    drifted_ratio = drifted / pair_denom

    if sample <= 0:
        blockers.append("parity sample missing")
    if available_ratio < min_available:
        blockers.append(f"available_pairs_ratio={available_ratio:.0%}<{min_available:.0%}")
    if missing_ratio > max_missing:
        blockers.append(f"missing_replay_ratio={missing_ratio:.0%}>{max_missing:.0%}")
    if stale_ratio > max_stale:
        blockers.append(f"stale_replay_ratio={stale_ratio:.0%}>{max_stale:.0%}")
    if drifted_ratio > max_drifted:
        blockers.append(f"drifted_pair_ratio={drifted_ratio:.0%}>{max_drifted:.0%}")

    generated_at = report.get("generated_at") or root_report.get("generated_at")
    max_age_hours = float(getattr(config, "ML_RANKER_PARITY_MAX_AGE_HOURS", 48))
    age_hours = None
    if generated_at:
        try:
            generated = datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
            now = datetime.now(tz=generated.tzinfo) if generated.tzinfo else datetime.now()
            age_hours = (now - generated).total_seconds() / 3600.0
            if age_hours > max_age_hours:
                blockers.append(f"parity_report_age={age_hours:.1f}h>{max_age_hours:.1f}h")
        except Exception:
            blockers.append("parity report generated_at invalid")
    else:
        blockers.append("parity report generated_at missing")

    critical_fields = list(getattr(config, "ADAPTIVE_WEIGHTS_PARITY_CRITICAL_FIELDS", []) or [])
    replay_missing = report.get("replay_missing_field_counts") or {}
    max_critical_missing = float(getattr(config, "ML_RANKER_PARITY_MAX_CRITICAL_MISSING_RATIO", 0.15))
    critical_missing: dict[str, float] = {}
    for field in critical_fields:
        missing_count = int(replay_missing.get(field) or 0)
        ratio = missing_count / pair_denom
        if ratio > max_critical_missing:
            critical_missing[field] = ratio
    if critical_missing:
        top = ", ".join(f"{field}={ratio:.0%}" for field, ratio in list(critical_missing.items())[:5])
        blockers.append(f"critical replay missingness too high: {top}")

    summary = {
        "enabled": True,
        "path": str(path),
        "sample": sample,
        "available_pairs": available_pairs,
        "available_ratio": round(available_ratio, 4),
        "missing_replay_ratio": round(missing_ratio, 4),
        "stale_replay_ratio": round(stale_ratio, 4),
        "drifted_pair_ratio": round(drifted_ratio, 4),
        "age_hours": None if age_hours is None else round(age_hours, 2),
        "critical_missing": {k: round(v, 4) for k, v in critical_missing.items()},
        "blockers": blockers,
    }
    return not blockers, blockers, summary
