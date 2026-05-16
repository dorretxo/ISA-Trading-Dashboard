"""Tier 2 value-discipline caps and Tier 3 quality floors.

Hard ceilings on multiples that have never empirically produced 5-year alpha
(Greenblatt 2006; Loughran & Wellman 2011; Asness, Liew, Pedersen, Thapar 2020;
Lakonishok, Shleifer, Vishny 1994), and quality floors required to receive a
STRONG BUY label (Novy-Marx 2013; Asness, Frazzini, Pedersen 2019; Damodaran).

Both gate sets share one return type — :class:`ValueQualityGateResult` — so the
caller can apply them in a single ``_cap_action_label`` pass.

Sector-relative fallback: if a candidate's sector is large enough (>= 8 names),
the cap is ``min(absolute_cap, 1.5 * sector_median)``; otherwise the absolute
cap applies.  Missing data never blocks — the gate is skipped and recorded as
"skip" in the flags.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValueQualityGateResult:
    action_ceiling: str = "STRONG BUY"
    reasons: list = field(default_factory=list)
    flags: dict = field(default_factory=dict)


_RANK = {"STRONG BUY": 4, "BUY": 3, "NEUTRAL": 2, "AVOID": 1, "MANUAL REVIEW": 0}


def _cap(current: str, candidate: str) -> str:
    return candidate if _RANK.get(candidate, 4) < _RANK.get(current, 4) else current


def _f(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _quality_growth_override_allowed(
    *,
    qmj_percentile: float | None,
    f_score: float | None,
    revenue_growth: float | None,
    sector_median_revenue_growth: float | None,
    config_module,
) -> tuple[bool, list[str]]:
    """Return whether a rich multiple earns the shadowable quality-growth waiver."""
    cfg = config_module
    if not getattr(cfg, "STRONG_BUY_VALUATION_QUALITY_OVERRIDE_ENABLED", False):
        return False, ["quality-growth valuation override disabled"]

    reasons: list[str] = []
    qmj = _f(qmj_percentile)
    qmj_floor = float(getattr(cfg, "STRONG_BUY_VALUATION_OVERRIDE_QMJ_FLOOR", 0.80))
    if qmj is None or qmj < qmj_floor:
        reasons.append(
            "QMJ pctile missing"
            if qmj is None
            else f"QMJ pctile {qmj:.0%} below override floor ({qmj_floor:.0%})"
        )

    fs = _f(f_score)
    min_f = float(getattr(cfg, "STRONG_BUY_VALUATION_OVERRIDE_MIN_F_SCORE", 7))
    if fs is None or fs < min_f:
        reasons.append(
            "F-score missing"
            if fs is None
            else f"F-score {fs:.0f}/9 below override floor ({min_f:.0f})"
        )

    growth = _f(revenue_growth)
    min_growth = float(getattr(cfg, "STRONG_BUY_VALUATION_OVERRIDE_MIN_REVENUE_GROWTH", 0.0))
    if growth is None or growth < min_growth:
        reasons.append(
            "Revenue growth missing"
            if growth is None
            else f"Revenue growth {growth:.0%} below override floor ({min_growth:.0%})"
        )

    if getattr(cfg, "STRONG_BUY_VALUATION_OVERRIDE_REQUIRE_SECTOR_GROWTH", True):
        sector_growth = _f(sector_median_revenue_growth)
        sector_growth_multiplier = float(
            getattr(cfg, "STRONG_BUY_VALUATION_OVERRIDE_SECTOR_GROWTH_MULTIPLIER", 1.0)
        )
        if sector_growth is None:
            reasons.append("Sector median revenue growth missing")
        elif growth is None or growth < sector_growth * sector_growth_multiplier:
            required = sector_growth * sector_growth_multiplier
            reasons.append(
                "Revenue growth missing"
                if growth is None else (
                    f"Revenue growth {growth:.0%} below sector median "
                    f"x{sector_growth_multiplier:.2g} ({required:.0%})"
                )
            )

    return not reasons, reasons


# ---------------------------------------------------------------------------
# Tier 2 — value caps
# ---------------------------------------------------------------------------

def evaluate_value_caps(
    *,
    ev_ebit: float | None,
    ev_sales: float | None,
    pe_forward: float | None,
    eps_growth_3y_cagr: float | None = None,
    sector_median_ev_ebit: float | None = None,
    revenue_growth: float | None = None,
    sector_median_revenue_growth: float | None = None,
    qmj_percentile: float | None = None,
    f_score: float | None = None,
    config_module=None,
) -> ValueQualityGateResult:
    """Apply EV/EBIT, EV/Sales and forward-P/E caps.

    A high QMJ percentile lifts the EV/Sales cap (quality compounders earn a
    multiple).  A high forward 3-year EPS CAGR lifts the P/E cap (LSV 1994).
    """
    cfg = config_module
    if cfg is None:
        import config as cfg    # type: ignore[no-redef]

    result = ValueQualityGateResult()
    if not getattr(cfg, "ACTION_GATES_ENABLED", True):
        return result

    # EV/EBIT — global cap with sector-relative tightening
    if getattr(cfg, "EV_EBIT_GATE_ENABLED", True):
        v = _f(ev_ebit)
        if v is not None and v > 0:
            global_strong = float(getattr(cfg, "EV_EBIT_STRONG_BUY_MAX", 30.0))
            global_buy = float(getattr(cfg, "EV_EBIT_BUY_MAX", 50.0))
            sec_med = _f(sector_median_ev_ebit)
            sector_strong = (sec_med * 1.5) if sec_med and sec_med > 0 else None
            cap_strong = min([c for c in (global_strong, sector_strong) if c is not None])

            if v > global_buy:
                result.action_ceiling = _cap(result.action_ceiling, "NEUTRAL")
                result.reasons.append(f"EV/EBIT {v:.1f}x above absolute ceiling ({global_buy:.0f}x)")
                result.flags["ev_ebit"] = "fail"
            elif v > cap_strong:
                override_max = float(getattr(cfg, "STRONG_BUY_VALUATION_OVERRIDE_MAX_EV_EBIT", 35.0))
                override_enabled = bool(getattr(cfg, "STRONG_BUY_VALUATION_QUALITY_OVERRIDE_ENABLED", False))
                override_ok, override_blockers = _quality_growth_override_allowed(
                    qmj_percentile=qmj_percentile,
                    f_score=f_score,
                    revenue_growth=revenue_growth,
                    sector_median_revenue_growth=sector_median_revenue_growth,
                    config_module=cfg,
                )
                if v <= override_max and override_ok:
                    result.flags["ev_ebit"] = "pass-quality-growth-override"
                    result.reasons.append(
                        f"EV/EBIT {v:.1f}x above {cap_strong:.0f}x but clears quality-growth override"
                    )
                else:
                    result.action_ceiling = _cap(result.action_ceiling, "BUY")
                    msg = f"EV/EBIT {v:.1f}x above STRONG-BUY cap ({cap_strong:.0f}x"
                    msg += " sector-relative)" if sector_strong is not None and sector_strong < global_strong else ")"
                    if v > override_max:
                        msg += f"; above override ceiling ({override_max:.0f}x)"
                    elif override_enabled and override_blockers:
                        msg += f"; override blocked: {'; '.join(override_blockers[:3])}"
                    result.reasons.append(msg)
                    result.flags["ev_ebit"] = "borderline"
            else:
                result.flags["ev_ebit"] = "pass"
        else:
            result.flags["ev_ebit"] = "skip"

    # EV/Sales — softer cap, lifted for high-QMJ names
    v = _f(ev_sales)
    if v is not None and v > 0:
        cap = float(getattr(cfg, "EV_SALES_STRONG_BUY_MAX", 8.0))
        qmj_pct = _f(qmj_percentile)
        qmj_override = float(getattr(cfg, "EV_SALES_QMJ_OVERRIDE_PCTILE", 0.80))
        if v > cap:
            if qmj_pct is not None and qmj_pct >= qmj_override:
                result.flags["ev_sales"] = "pass-qmj-override"
                result.reasons.append(
                    f"EV/Sales {v:.1f}x above {cap:.0f}x but QMJ pctile {qmj_pct:.0%} clears override"
                )
            else:
                result.action_ceiling = _cap(result.action_ceiling, "BUY")
                result.reasons.append(f"EV/Sales {v:.1f}x above STRONG-BUY cap ({cap:.0f}x)")
                result.flags["ev_sales"] = "fail"
        else:
            result.flags["ev_sales"] = "pass"
    else:
        result.flags["ev_sales"] = "skip"

    # Forward P/E with growth override
    v = _f(pe_forward)
    if v is not None and v > 0:
        cap = float(getattr(cfg, "PE_FORWARD_STRONG_BUY_MAX", 30.0))
        growth = _f(eps_growth_3y_cagr)
        growth_override = float(getattr(cfg, "PE_FORWARD_GROWTH_OVERRIDE_CAGR", 0.25))
        if v > cap:
            if growth is not None and growth >= growth_override:
                result.flags["pe_forward"] = "pass-growth-override"
                result.reasons.append(
                    f"Fwd P/E {v:.1f}x above {cap:.0f}x but EPS CAGR {growth:.0%} clears override"
                )
            else:
                override_max = float(getattr(cfg, "STRONG_BUY_VALUATION_OVERRIDE_MAX_PE_FORWARD", 40.0))
                override_enabled = bool(getattr(cfg, "STRONG_BUY_VALUATION_QUALITY_OVERRIDE_ENABLED", False))
                override_ok, override_blockers = _quality_growth_override_allowed(
                    qmj_percentile=qmj_percentile,
                    f_score=f_score,
                    revenue_growth=revenue_growth,
                    sector_median_revenue_growth=sector_median_revenue_growth,
                    config_module=cfg,
                )
                if v <= override_max and override_ok:
                    result.flags["pe_forward"] = "pass-quality-growth-override"
                    result.reasons.append(
                        f"Fwd P/E {v:.1f}x above {cap:.0f}x but clears quality-growth override"
                    )
                else:
                    result.action_ceiling = _cap(result.action_ceiling, "BUY")
                    msg = f"Fwd P/E {v:.1f}x above STRONG-BUY cap ({cap:.0f}x)"
                    if v > override_max:
                        msg += f"; above override ceiling ({override_max:.0f}x)"
                    elif override_enabled and override_blockers:
                        msg += f"; override blocked: {'; '.join(override_blockers[:3])}"
                    result.reasons.append(msg)
                    result.flags["pe_forward"] = "fail"
        else:
            result.flags["pe_forward"] = "pass"
    else:
        result.flags["pe_forward"] = "skip"

    return result


# ---------------------------------------------------------------------------
# Tier 3 — quality floors
# ---------------------------------------------------------------------------

def evaluate_quality_floors(
    *,
    gpa_percentile: float | None,
    qmj_percentile: float | None,
    roic: float | None,
    wacc: float | None,
    op_margin_yoy_delta: float | None,    # As decimal, e.g. -0.023 for -230 bps
    config_module=None,
) -> ValueQualityGateResult:
    """Quality floors — must clear, not merely contribute to a score.

    Source citations live in the relevant config flags.  Each floor is
    independently togglable; missing data never blocks (recorded as "skip").
    """
    cfg = config_module
    if cfg is None:
        import config as cfg    # type: ignore[no-redef]

    result = ValueQualityGateResult()
    if not getattr(cfg, "ACTION_GATES_ENABLED", True):
        return result

    # Gross-Profits-to-Assets percentile — Novy-Marx 2013
    if getattr(cfg, "GPA_GATE_ENABLED", True):
        gpa = _f(gpa_percentile)
        if gpa is not None:
            floor = float(getattr(cfg, "GPA_PCTILE_STRONG_BUY_MIN", 0.25))
            if gpa < floor:
                result.action_ceiling = _cap(result.action_ceiling, "BUY")
                result.reasons.append(f"GPA pctile {gpa:.0%} below STRONG-BUY floor ({floor:.0%})")
                result.flags["gpa_pctile"] = "fail"
            else:
                result.flags["gpa_pctile"] = "pass"
        else:
            result.flags["gpa_pctile"] = "skip"

    # ROIC vs WACC spread
    if getattr(cfg, "ROIC_WACC_GATE_ENABLED", True):
        r = _f(roic)
        w = _f(wacc)
        if r is not None and w is not None:
            spread_bps = (r - w) * 10_000.0
            min_bps = float(getattr(cfg, "ROIC_VS_WACC_STRONG_BUY_MIN_BPS", 200.0))
            if spread_bps < min_bps:
                result.action_ceiling = _cap(result.action_ceiling, "BUY")
                result.reasons.append(
                    f"ROIC-WACC spread {spread_bps:+.0f}bps below {min_bps:.0f}bps"
                )
                result.flags["roic_wacc"] = "fail"
            else:
                result.flags["roic_wacc"] = "pass"
        else:
            result.flags["roic_wacc"] = "skip"

    # QMJ percentile
    if getattr(cfg, "QMJ_GATE_ENABLED", True):
        q = _f(qmj_percentile)
        if q is not None:
            floor = float(getattr(cfg, "QMJ_PCTILE_STRONG_BUY_MIN", 0.50))
            if q < floor:
                result.action_ceiling = _cap(result.action_ceiling, "BUY")
                result.reasons.append(f"QMJ pctile {q:.0%} below STRONG-BUY floor ({floor:.0%})")
                result.flags["qmj_pctile"] = "fail"
            else:
                result.flags["qmj_pctile"] = "pass"
        else:
            result.flags["qmj_pctile"] = "skip"

    # YoY operating-margin direction
    delta = _f(op_margin_yoy_delta)
    if delta is not None:
        floor_bps = float(getattr(cfg, "OP_MARGIN_YOY_DELTA_FLOOR_BPS", -150.0))
        delta_bps = delta * 10_000.0
        if delta_bps < floor_bps:
            result.action_ceiling = _cap(result.action_ceiling, "BUY")
            result.reasons.append(
                f"Op-margin YoY {delta_bps:+.0f}bps below floor {floor_bps:+.0f}bps"
            )
            result.flags["op_margin_yoy"] = "fail"
        else:
            result.flags["op_margin_yoy"] = "pass"
    else:
        result.flags["op_margin_yoy"] = "skip"

    return result


# ---------------------------------------------------------------------------
# Sector-median helper (called once per discovery batch, not per ticker)
# ---------------------------------------------------------------------------

def compute_sector_medians(rows: list[dict], *, key: str = "ev_ebit", sector_key: str = "sector") -> dict[str, float]:
    """Return {sector: median(value)} skipping non-finite, non-positive values."""
    buckets: dict[str, list[float]] = {}
    for row in rows or []:
        sec = row.get(sector_key) or "Unknown"
        val = row.get(key)
        try:
            num = float(val)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(num) or num <= 0:
            continue
        buckets.setdefault(sec, []).append(num)
    out: dict[str, float] = {}
    for sec, vals in buckets.items():
        if len(vals) >= 8:    # Need a meaningful sample
            vals.sort()
            mid = len(vals) // 2
            out[sec] = (vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid]))
    return out


def cross_sectional_percentile(values: list[float | None]) -> list[float | None]:
    """Return per-row percentile rank in [0, 1] — None where the input is None."""
    finite_idx = [i for i, v in enumerate(values) if v is not None and math.isfinite(v)]
    if len(finite_idx) < 3:
        return [None] * len(values)
    finite_vals = sorted(values[i] for i in finite_idx)
    pct: list[float | None] = [None] * len(values)
    n = len(finite_vals)
    for i in finite_idx:
        v = values[i]
        # Empirical CDF — fraction of values <= v
        # Linear scan is fine; n is at most a few hundred per discovery batch.
        rank = sum(1 for x in finite_vals if x <= v)
        pct[i] = rank / n
    return pct
