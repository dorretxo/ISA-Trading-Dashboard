"""Self-monitoring & parameter-tuning recommendations.

Reads telemetry (orchestrator_log.jsonl, DiscoveryResult funnel stats,
paper_trading.db) and produces a recommendation report. Does NOT mutate
config — the design is "propose, don't apply" so decision-support ethos
and backtest reproducibility are preserved. The user reviews the report
and chooses whether to edit config.py.

v1 scope (deliberately narrow):
  - COOLDOWN_DAYS                — churn vs. binding analysis
  - DISCOVERY_DOLLAR_VOLUME_FLOORS — stage-2 survival rates per region
  - HURDLE_RATE / PORTFOLIO_FIT_MIN — swap-evaluation pass rates

Explicitly OUT of scope: scoring weights, action thresholds, universe
mcap floor. Changing those mid-stream invalidates historical evaluation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import config

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_LOG_PATH = _ROOT / getattr(config, "ORCHESTRATOR_LOG_FILE", "orchestrator_log.jsonl")
_REPORT_PATH = _ROOT / "feature_cache" / "auto_tune_report.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    param: str
    current_value: Any
    suggested_value: Any
    severity: str           # "info" | "warn" | "action"
    rationale: str
    evidence: dict[str, Any] = field(default_factory=dict)
    confidence: str = "medium"  # "low" | "medium" | "high"


@dataclass
class TuningReport:
    generated_at: str
    window_days: int
    metrics: dict[str, Any] = field(default_factory=dict)
    recommendations: list[Recommendation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Log reading
# ---------------------------------------------------------------------------

def _read_log_events(
    window_days: int,
    wanted_events: set[str] | None = None,
) -> list[dict]:
    """Return parsed JSONL events within the window. Resilient to bad lines."""
    if not _LOG_PATH.exists():
        return []
    cutoff = datetime.now() - timedelta(days=window_days)
    out: list[dict] = []
    try:
        with open(_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if wanted_events and rec.get("event") not in wanted_events:
                    continue
                ts_raw = rec.get("ts")
                if not ts_raw:
                    continue
                try:
                    ts = datetime.fromisoformat(ts_raw)
                except ValueError:
                    continue
                if ts < cutoff:
                    continue
                out.append(rec)
    except OSError as e:
        logger.warning("auto_tune: failed to read log: %s", e)
    return out


def _read_cached_discovery() -> list[dict]:
    path = _ROOT / getattr(config, "ORCHESTRATOR_STATE_FILE", "orchestrator_state.json")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        cached = state.get("cached_discovery") or []
        return cached if isinstance(cached, list) else []
    except Exception:
        return []


def _is_core_ready_candidate(candidate: Any) -> bool:
    if isinstance(candidate, dict):
        return str(
            candidate.get("ready_contract_core_status")
            or candidate.get("ready_contract_status")
            or ""
        ).upper() == "PASS"
    return str(
        getattr(candidate, "ready_contract_core_status", None)
        or getattr(candidate, "ready_contract_status", "")
        or ""
    ).upper() == "PASS"


def _candidate_meta_probs(candidates: list[Any], *, core_ready_only: bool = False) -> list[float]:
    out: list[float] = []
    for c in candidates:
        if core_ready_only and not _is_core_ready_candidate(c):
            continue
        if isinstance(c, dict):
            value = c.get("meta_label_proba", c.get("meta_success_prob", c.get("meta_prob")))
        else:
            value = getattr(c, "meta_success_prob", getattr(c, "meta_prob", None))
        try:
            prob = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 <= prob <= 1.0:
            out.append(prob)
    return out


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[idx]


def _meta_threshold_from_probs(values: list[float]) -> tuple[float, dict[str, Any]]:
    static = float(getattr(config, "META_LABEL_STRONG_BUY_MIN_PROB", 0.60))
    evidence: dict[str, Any] = {
        "static_threshold": static,
        "dynamic_enabled": bool(getattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_THRESHOLD", False)),
        "count": len(values),
    }
    if not getattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_THRESHOLD", False):
        evidence["threshold"] = static
        return static, evidence
    min_count = int(getattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_MIN_CANDIDATES", 50))
    if len(values) < min_count:
        evidence["threshold"] = static
        evidence["reason"] = f"insufficient live meta probabilities ({len(values)}<{min_count})"
        return static, evidence

    pct = float(getattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_PCTILE", 0.90))
    floor = float(getattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_FLOOR", 0.55))
    multiplier = float(getattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_MULTIPLIER", 0.92))
    pctl = _percentile(values, pct)
    p50 = _percentile(values, 0.50)
    p95 = _percentile(values, 0.95)
    count_at_static = sum(1 for v in values if v >= static)
    compressed = pctl is not None and pctl < floor and count_at_static == 0
    mode = "percentile_floor"
    effective_floor = floor
    if compressed:
        compressed_floor = float(getattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_COMPRESSED_FLOOR", 0.45))
        compressed_pct = float(getattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_COMPRESSED_PCTILE", 0.95))
        compressed_tail = _percentile(values, compressed_pct)
        suggested = max(compressed_floor, compressed_tail or pctl or static)
        effective_floor = compressed_floor
        mode = "compressed_tail"
    else:
        suggested = max(floor, (pctl or static) * multiplier)
    threshold = round(min(0.95, max(0.01, suggested)), 4)
    evidence.update({
        "threshold": threshold,
        "floor": floor,
        "effective_floor": effective_floor,
        "mode": mode,
        "percentile": pct,
        "percentile_value": pctl,
        "multiplier": multiplier,
        "min": min(values),
        "p50": p50,
        "p90": _percentile(values, 0.90),
        "p95": p95,
        "max": max(values),
        "count_at_static_threshold": count_at_static,
        "count_at_dynamic_threshold": sum(1 for v in values if v >= threshold),
    })
    return threshold, evidence


def get_meta_strong_buy_min_prob(candidates: list[Any] | None = None) -> float:
    """Return the active meta-label gate without mutating config."""
    values = _candidate_meta_probs(candidates or _read_cached_discovery())
    threshold, _ = _meta_threshold_from_probs(values)
    return threshold


def _core_ready_meta_threshold_from_probs(values: list[float], fallback_threshold: float) -> tuple[float, dict[str, Any]]:
    evidence: dict[str, Any] = {
        "fallback_threshold": fallback_threshold,
        "dynamic_enabled": bool(getattr(config, "META_LABEL_CORE_READY_DYNAMIC_THRESHOLD", True)),
        "count": len(values),
    }
    if not getattr(config, "META_LABEL_CORE_READY_DYNAMIC_THRESHOLD", True):
        evidence["threshold"] = fallback_threshold
        return fallback_threshold, evidence

    min_count = int(getattr(config, "META_LABEL_CORE_READY_DYNAMIC_MIN_CANDIDATES", 2))
    if len(values) < min_count:
        evidence["threshold"] = fallback_threshold
        evidence["reason"] = f"insufficient core-ready meta probabilities ({len(values)}<{min_count})"
        return fallback_threshold, evidence

    pct = float(getattr(config, "META_LABEL_CORE_READY_DYNAMIC_PCTILE", 0.50))
    floor = float(getattr(config, "META_LABEL_CORE_READY_DYNAMIC_FLOOR", 0.45))
    multiplier = float(getattr(config, "META_LABEL_CORE_READY_DYNAMIC_MULTIPLIER", 0.98))
    max_threshold = float(getattr(config, "META_LABEL_CORE_READY_DYNAMIC_MAX_THRESHOLD", fallback_threshold))
    pctl = _percentile(values, pct)
    suggested = max(floor, (pctl or fallback_threshold) * multiplier)
    threshold = round(min(max_threshold, max(0.01, suggested)), 4)
    evidence.update({
        "threshold": threshold,
        "floor": floor,
        "max_threshold": max_threshold,
        "percentile": pct,
        "percentile_value": pctl,
        "multiplier": multiplier,
        "min": min(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "max": max(values),
        "count_at_threshold": sum(1 for v in values if v >= threshold),
    })
    return threshold, evidence


def get_core_ready_meta_strong_buy_min_prob(candidates: list[Any] | None = None) -> float:
    """Return a meta gate calibrated only on candidates that pass core readiness."""
    pool = candidates or _read_cached_discovery()
    fallback = get_meta_strong_buy_min_prob(pool)
    values = _candidate_meta_probs(pool, core_ready_only=True)
    threshold, _ = _core_ready_meta_threshold_from_probs(values, fallback)
    return threshold


def analyze_meta_threshold(events: list[dict]) -> tuple[dict, list[Recommendation]]:
    """Recommend the active STRONG BUY meta gate from the live probability distribution."""
    cached = _read_cached_discovery()
    values = _candidate_meta_probs(cached)
    threshold, evidence = _meta_threshold_from_probs(values)
    core_threshold, core_evidence = _core_ready_meta_threshold_from_probs(
        _candidate_meta_probs(cached, core_ready_only=True),
        threshold,
    )
    evidence["core_ready_threshold"] = core_threshold
    evidence["core_ready_evidence"] = core_evidence
    current = float(getattr(config, "META_LABEL_STRONG_BUY_MIN_PROB", 0.60))
    recs: list[Recommendation] = []
    if evidence.get("dynamic_enabled") and evidence.get("count", 0) >= int(getattr(config, "META_LABEL_STRONG_BUY_DYNAMIC_MIN_CANDIDATES", 50)):
        if abs(threshold - current) >= 0.01:
            recs.append(Recommendation(
                param="META_LABEL_STRONG_BUY_MIN_PROB_ACTIVE",
                current_value=current,
                suggested_value=threshold,
                severity="info",
                rationale=(
                    "Live meta-label probabilities are compressed relative to the static gate. "
                    "Use the dynamic percentile gate for serving while keeping config as the "
                    "fallback reference."
                ),
                evidence=evidence,
                confidence="medium",
            ))
    return evidence, recs


# ---------------------------------------------------------------------------
# Analyzers — each returns (metrics_dict, [Recommendation])
# ---------------------------------------------------------------------------

def analyze_cooldown_pressure(events: list[dict]) -> tuple[dict, list[Recommendation]]:
    """How often is the cooldown gate the reason a swap is blocked?

    Heuristics (evidence-based, conservative):
      - Hit-rate >30% of swap_evals → likely too long; suggest -7 days.
      - Hit-rate <5% over 30d with ≥20 evals → likely not binding; flag, no change.
      - Otherwise informational.
    """
    swap_evals = [e for e in events if e.get("event") == "swap_eval"]
    total = len(swap_evals)
    if total == 0:
        return ({"swap_evals": 0}, [])

    on_cd = sum(1 for e in swap_evals if e.get("details", {}).get("on_cooldown"))
    hit_rate = on_cd / total
    metrics = {
        "swap_evals": total,
        "blocked_by_cooldown": on_cd,
        "cooldown_hit_rate": round(hit_rate, 3),
    }

    current = int(getattr(config, "COOLDOWN_DAYS", 14))
    recs: list[Recommendation] = []

    if hit_rate > 0.30 and total >= 20:
        new = max(7, current - 7)
        if new != current:
            recs.append(Recommendation(
                param="COOLDOWN_DAYS",
                current_value=current,
                suggested_value=new,
                severity="warn",
                rationale=(
                    f"{on_cd}/{total} swap evaluations ({hit_rate:.0%}) were blocked by "
                    "cooldown over the observation window. High churn signal; a shorter "
                    "cooldown may be letting better candidates wait too long."
                ),
                evidence=metrics,
                confidence="medium",
            ))
    elif hit_rate < 0.05 and total >= 20:
        recs.append(Recommendation(
            param="COOLDOWN_DAYS",
            current_value=current,
            suggested_value=current,
            severity="info",
            rationale=(
                f"Cooldown fired on only {on_cd}/{total} evaluations ({hit_rate:.1%}). "
                "Not binding. Keep as-is; no churn problem to solve."
            ),
            evidence=metrics,
            confidence="high",
        ))
    return metrics, recs


def analyze_hurdle_pressure(events: list[dict]) -> tuple[dict, list[Recommendation]]:
    """Proportion of swap_evals failing the score-delta or fit checks.

    - If >95% fail hurdle → likely too strict OR market regime unfavourable.
      Recommend -0.05, clamped at 0.10.
    - If <30% fail hurdle AND recommendations are flowing normally → no change.
    """
    swap_evals = [e for e in events if e.get("event") == "swap_eval"]
    total = len(swap_evals)
    if total == 0:
        return ({"swap_evals": 0}, [])

    fail_hurdle = sum(1 for e in swap_evals if not e.get("details", {}).get("passes_hurdle", True))
    fail_fit = sum(1 for e in swap_evals if not e.get("details", {}).get("passes_fit", True))
    recommended = sum(1 for e in swap_evals if e.get("details", {}).get("recommended"))

    hurdle_fail_rate = fail_hurdle / total
    fit_fail_rate = fail_fit / total

    metrics = {
        "swap_evals": total,
        "hurdle_fail_rate": round(hurdle_fail_rate, 3),
        "fit_fail_rate": round(fit_fail_rate, 3),
        "recommended_count": recommended,
    }

    current_hurdle = float(getattr(config, "HURDLE_RATE", 0.20))
    current_fit = float(getattr(config, "PORTFOLIO_FIT_MIN", 0.50))
    recs: list[Recommendation] = []

    if hurdle_fail_rate > 0.95 and recommended == 0 and total >= 15:
        suggested = round(max(0.10, current_hurdle - 0.05), 3)
        if suggested != current_hurdle:
            recs.append(Recommendation(
                param="HURDLE_RATE",
                current_value=current_hurdle,
                suggested_value=suggested,
                severity="warn",
                rationale=(
                    f"{fail_hurdle}/{total} ({hurdle_fail_rate:.0%}) of candidates failed "
                    "the score-delta hurdle and zero swaps were recommended. The hurdle "
                    "may be unreachable in the current regime."
                ),
                evidence=metrics,
                confidence="low",
            ))

    if fit_fail_rate > 0.80 and total >= 15:
        suggested = round(max(0.30, current_fit - 0.05), 3)
        recs.append(Recommendation(
            param="PORTFOLIO_FIT_MIN",
            current_value=current_fit,
            suggested_value=suggested,
            severity="info",
            rationale=(
                f"{fail_fit}/{total} ({fit_fail_rate:.0%}) of candidates failed portfolio-fit. "
                "Considering a small relaxation to surface more swap options. "
                "Verify portfolio diversification still holds after any change."
            ),
            evidence=metrics,
            confidence="low",
        ))
    return metrics, recs


def analyze_funnel_health(events: list[dict]) -> tuple[dict, list[Recommendation]]:
    """Stage-by-stage survival from discovery_done events. Flags collapsing
    funnels (possible over-tight gates) and suspicious widening."""
    done = [e for e in events if e.get("event") == "discovery_done"]
    if not done:
        return ({}, [])

    latest = done[-1].get("details", {})
    screened = int(latest.get("screened", 0) or 0)
    after_mom = int(latest.get("after_momentum", 0) or 0)
    after_filter = int(latest.get("after_filter", 0) or 0)
    after_rank = int(latest.get("after_rank", 0) or 0)
    fully_scored = int(latest.get("fully_scored", 0) or 0)
    final = int(latest.get("final_candidates", 0) or 0)

    def pct(a: int, b: int) -> float:
        return round(a / b, 3) if b > 0 else 0.0

    metrics = {
        "screened": screened,
        "after_momentum": after_mom,
        "after_filter": after_filter,
        "after_rank": after_rank,
        "fully_scored": fully_scored,
        "final_candidates": final,
        "survival_stage2": pct(after_mom, screened),
        "survival_stage3": pct(after_filter, after_mom),
        "survival_stage5a": pct(after_rank, after_filter),
    }

    recs: list[Recommendation] = []

    # Stage 2 (dollar-vol gate): if survival <20%, the floor may be too tight.
    if screened >= 500 and pct(after_mom, screened) < 0.20:
        floors = dict(getattr(config, "DISCOVERY_DOLLAR_VOLUME_FLOORS", {}) or {})
        suggested = {k: int(v * 0.75) for k, v in floors.items()}
        recs.append(Recommendation(
            param="DISCOVERY_DOLLAR_VOLUME_FLOORS",
            current_value=floors,
            suggested_value=suggested,
            severity="warn",
            rationale=(
                f"Stage-2 survival is {metrics['survival_stage2']:.0%} "
                f"({after_mom}/{screened}). Dollar-volume floors may be cutting too "
                "deep. Consider a 25% reduction across regions."
            ),
            evidence=metrics,
            confidence="medium",
        ))

    # Final list too thin → upstream gates choking quality candidates.
    if final < 10 and screened >= 500:
        recs.append(Recommendation(
            param="(funnel-wide)",
            current_value=final,
            suggested_value=None,
            severity="warn",
            rationale=(
                f"Only {final} final candidates from {screened} screened. Worth "
                "reviewing Stage 2 / Stage 5a preselect percentages before tuning "
                "individual gates."
            ),
            evidence=metrics,
            confidence="low",
        ))
    return metrics, recs


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(window_days: int = 30, write_report: bool = True) -> TuningReport:
    """Run all analyzers over the given window and produce a TuningReport.

    Never mutates config. Writes a JSON report to feature_cache/ for audit
    and dashboard consumption. Safe to call from any context.
    """
    report = TuningReport(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        window_days=window_days,
    )

    wanted = {"swap_eval", "discovery_done"}
    events = _read_log_events(window_days=window_days, wanted_events=wanted)

    if not events:
        report.notes.append(
            "No relevant telemetry events in the observation window. "
            "Log-based auto-tune will become useful after a few days of daily discovery runs."
        )

    for name, fn in (
        ("cooldown", analyze_cooldown_pressure),
        ("hurdle", analyze_hurdle_pressure),
        ("funnel", analyze_funnel_health),
        ("meta_threshold", analyze_meta_threshold),
    ):
        try:
            metrics, recs = fn(events)
            report.metrics[name] = metrics
            report.recommendations.extend(recs)
        except Exception as e:
            logger.warning("auto_tune analyzer %s failed: %s", name, e)
            report.notes.append(f"Analyzer '{name}' failed: {e}")

    if write_report:
        _write_report(report)
    return report


def _write_report(report: TuningReport) -> None:
    try:
        from utils.atomic_io import atomic_write_json
        _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": report.generated_at,
            "window_days": report.window_days,
            "metrics": report.metrics,
            "recommendations": [asdict(r) for r in report.recommendations],
            "notes": report.notes,
        }
        atomic_write_json(_REPORT_PATH, payload, indent=2)
    except Exception as e:
        logger.warning("auto_tune: failed to write report: %s", e)


def summarize(report: TuningReport) -> str:
    """One-line log summary for the orchestrator output."""
    n_recs = len(report.recommendations)
    if n_recs == 0:
        return "auto_tune: no recommendations"
    by_sev: dict[str, int] = {}
    for r in report.recommendations:
        by_sev[r.severity] = by_sev.get(r.severity, 0) + 1
    parts = [f"{k}={v}" for k, v in sorted(by_sev.items())]
    return f"auto_tune: {n_recs} recommendation(s) [{', '.join(parts)}]"
