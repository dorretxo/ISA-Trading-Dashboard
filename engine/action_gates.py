"""Combined Tier 1-4 action-gate orchestrator.

This module wires the four gate evaluators (distress, value-cap, quality-floor,
momentum) together and applies the strictest action ceiling to a batch of
ScoredCandidate objects.

It intentionally has *no* hard imports of the discovery dataclass — it accepts
any object with the relevant attributes — so it is unit-testable in isolation
and the discovery module can call it without a circular import.

Sleeve coverage and limit-price suggestion are also handled here.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from engine.distress import evaluate_distress_gates
from engine.momentum_gates import (
    evaluate_momentum_gates,
    suggest_limit_price,
)
from engine.value_caps import (
    compute_sector_medians,
    cross_sectional_percentile,
    evaluate_quality_floors,
    evaluate_value_caps,
)

logger = logging.getLogger(__name__)


_RANK = {"STRONG BUY": 4, "BUY": 3, "NEUTRAL": 2, "AVOID": 1, "MANUAL REVIEW": 0}


def _strictest(*labels: str) -> str:
    return min(labels, key=lambda x: _RANK.get(x, 4))


def _f(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


@dataclass
class GateContext:
    """Pre-computed batch-level context shared across candidates."""
    sector_median_ev_ebit: dict = field(default_factory=dict)
    qmj_percentiles: dict = field(default_factory=dict)        # ticker -> percentile [0,1]
    gpa_percentiles: dict = field(default_factory=dict)


def build_context(candidates: Iterable[Any]) -> GateContext:
    """One-shot scan over the batch to build sector medians and percentiles."""
    rows: list[dict] = []
    qmj_values: list[float | None] = []
    gpa_values: list[float | None] = []
    tickers: list[str] = []

    for c in candidates:
        ticker = str(getattr(c, "ticker", "") or "")
        sector = str(getattr(c, "sector", "") or "Unknown")
        ev_ebit = _f(getattr(c, "ev_ebit", None))
        rows.append({"ticker": ticker, "sector": sector, "ev_ebit": ev_ebit})
        qmj_values.append(_f(getattr(c, "qmj_factor_score", None)))
        # Use gpa as proxy when gpa_score isn't present
        gpa_values.append(_f(getattr(c, "gpa_score", None)) or _f(getattr(c, "gpa", None)))
        tickers.append(ticker)

    sec_med = compute_sector_medians(rows, key="ev_ebit", sector_key="sector")
    qmj_pcts = cross_sectional_percentile(qmj_values)
    gpa_pcts = cross_sectional_percentile(gpa_values)

    return GateContext(
        sector_median_ev_ebit=sec_med,
        qmj_percentiles={t: p for t, p in zip(tickers, qmj_pcts)},
        gpa_percentiles={t: p for t, p in zip(tickers, gpa_pcts)},
    )


def evaluate_candidate(candidate: Any, *, context: GateContext, config_module=None) -> dict:
    """Apply Tier 1-4 gates to a single candidate.  Returns a result dict."""
    cfg = config_module
    if cfg is None:
        import config as cfg    # type: ignore[no-redef]

    if not getattr(cfg, "ACTION_GATES_ENABLED", True):
        return {
            "ceiling": "STRONG BUY", "reasons": [], "flags": {},
            "limit_price": None, "limit_price_method": None, "limit_price_rationale": None,
        }

    ticker = str(getattr(candidate, "ticker", "") or "")
    sector = str(getattr(candidate, "sector", "") or "")
    qmj_pct = context.qmj_percentiles.get(ticker)
    gpa_pct = context.gpa_percentiles.get(ticker)
    sec_med_ev_ebit = context.sector_median_ev_ebit.get(sector)

    # Tier 1
    t1 = evaluate_distress_gates(
        altman_z=_f(getattr(candidate, "altman_z", None)),
        beneish_m=_f(getattr(candidate, "beneish_m", None)),
        f_score=getattr(candidate, "f_score", None),
        f_score_coverage=_f(getattr(candidate, "f_score_coverage", None)),
        accruals_factor_score=_f(getattr(candidate, "accruals_factor_score", None)),
        investment_factor_score=_f(getattr(candidate, "investment_factor_score", None)),
        net_debt_ebitda=_f(getattr(candidate, "net_debt_ebitda", None)),
        config_module=cfg,
    )

    # Tier 2 — value caps.  pe_forward fallback to trailing pe_ratio when forward
    # is not directly available on the candidate (yfinance doesn't always expose
    # forwardPE).  This is conservative — trailing P/E >= forward P/E in growth
    # names, so the cap binds at least as tightly.
    pe_forward = _f(getattr(candidate, "pe_forward", None)) or _f(getattr(candidate, "pe_ratio", None))
    t2 = evaluate_value_caps(
        ev_ebit=_f(getattr(candidate, "ev_ebit", None)),
        ev_sales=_f(getattr(candidate, "ev_sales", None)),
        pe_forward=pe_forward,
        eps_growth_3y_cagr=_f(getattr(candidate, "eps_growth_3y_cagr", None)),
        sector_median_ev_ebit=sec_med_ev_ebit,
        qmj_percentile=qmj_pct,
        config_module=cfg,
    )

    # Tier 3 — quality floors
    t3 = evaluate_quality_floors(
        gpa_percentile=gpa_pct,
        qmj_percentile=qmj_pct,
        roic=_f(getattr(candidate, "roic", None)),
        wacc=_f(getattr(candidate, "wacc", None)),
        op_margin_yoy_delta=_f(getattr(candidate, "op_margin_yoy_delta", None)),
        config_module=cfg,
    )

    # Tier 4 — momentum sanity
    t4 = evaluate_momentum_gates(
        rsi=_f(getattr(candidate, "rsi", None)),
        stretch_200dma=_f(getattr(candidate, "price_vs_sma200_stretch", None)),
        entry_stance=getattr(candidate, "entry_stance", None),
        realized_vol_pctile=_f(getattr(candidate, "realized_vol_pctile", None)),
        config_module=cfg,
    )

    ceiling = _strictest(t1.action_ceiling, t2.action_ceiling, t3.action_ceiling, t4.action_ceiling)
    reasons: list = []
    flags: dict = {}
    for r in (t1, t2, t3, t4):
        reasons.extend(r.reasons)
        flags.update(r.flags)

    # Limit-price suggestion when momentum gates demand a pullback
    limit_price = None
    limit_method = None
    limit_rationale = None
    if t4.needs_pullback:
        try:
            cp = _f(getattr(candidate, "current_price", None)) or _f(getattr(candidate, "entry_price", None))
            sup_levels = getattr(candidate, "support_levels", {}) or {}
            sup = None
            if isinstance(sup_levels, dict):
                # Use the highest support level below current price
                for v in sup_levels.values():
                    fv = _f(v)
                    if fv is not None and (sup is None or fv > sup):
                        sup = fv
            sug = suggest_limit_price(
                current_price=cp,
                sma_200=_f(getattr(candidate, "sma_200", None)),
                support_level=sup,
                atr=_f(getattr(candidate, "atr", None)),
                config_module=cfg,
            )
            if sug is not None:
                limit_price = sug.limit_price
                limit_method = sug.method
                limit_rationale = sug.rationale
        except Exception as exc:    # pragma: no cover — defensive
            logger.debug("Limit-price suggestion failed for %s: %s", ticker, exc)

    return {
        "ceiling": ceiling,
        "reasons": reasons,
        "flags": flags,
        "limit_price": limit_price,
        "limit_price_method": limit_method,
        "limit_price_rationale": limit_rationale,
    }


def apply_action_gates(candidates: list, *, config_module=None) -> dict:
    """Apply Tier 1-4 gates to a batch and mutate candidates in place.

    Returns a summary dict with per-tier failure counts (used by the
    threshold-learner and the digest UI).
    """
    cfg = config_module
    if cfg is None:
        import config as cfg    # type: ignore[no-redef]

    if not candidates or not getattr(cfg, "ACTION_GATES_ENABLED", True):
        return {"applied": 0, "downgrades": 0, "limits": 0}

    ctx = build_context(candidates)
    summary = {"applied": 0, "downgrades": 0, "limits": 0,
               "tier_fail_counts": {}}

    for c in candidates:
        ev = evaluate_candidate(c, context=ctx, config_module=cfg)
        c.action_gate_ceiling = ev["ceiling"]
        c.action_gate_reasons = ev["reasons"]
        c.action_gate_flags = ev["flags"]
        c.limit_price = ev["limit_price"]
        c.limit_price_method = ev["limit_price_method"]
        c.limit_price_rationale = ev["limit_price_rationale"]

        summary["applied"] += 1
        if ev["ceiling"] != "STRONG BUY":
            summary["downgrades"] += 1
        if ev["limit_price"] is not None:
            summary["limits"] += 1
        for k, v in ev["flags"].items():
            if v == "fail":
                summary["tier_fail_counts"][k] = summary["tier_fail_counts"].get(k, 0) + 1

    return summary


# ---------------------------------------------------------------------------
# Helpers used by the percentile-action assigner
# ---------------------------------------------------------------------------

def cap_action(percentile_action: str, gate_ceiling: str) -> str:
    """Return the strictest of the two action labels."""
    return _strictest(percentile_action, gate_ceiling)
