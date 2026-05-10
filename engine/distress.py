"""Distress and earnings-manipulation gates.

Implements:

- Altman Z-Score (Altman 1968, *J. Finance* 23(4); Altman & Hotchkiss 2010).
  Z > 2.99 = safe; 1.81 < Z < 2.99 = grey; Z < 1.81 = distress.

- Beneish M-Score (Beneish 1999, *Financial Analysts J.* 55(5)).  M > -1.78
  flags likely earnings manipulators; famously caught Enron a year early.

Both are computed defensively — when input statements are missing fields the
function returns ``None`` and the gate is treated as "cannot evaluate" (data
coverage is recorded so the threshold-learner can debias).  The functions
work with both yfinance ``info``-style payloads and FMP balance-sheet /
income-statement lists.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from engine.fscore_utils import f_score_min_coverage, normalize_f_score_coverage
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (kept local so this module has no other engine dependencies)
# ---------------------------------------------------------------------------

def _f(value: Any) -> float | None:
    """Return value as a finite float, else None."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def _first(*values: Any) -> float | None:
    """First finite numeric in *values*."""
    for v in values:
        out = _f(v)
        if out is not None:
            return out
    return None


def _statement_field(statement: dict | None, *keys: str) -> float | None:
    if not statement:
        return None
    for k in keys:
        if k in statement:
            v = _f(statement[k])
            if v is not None:
                return v
    return None


# ---------------------------------------------------------------------------
# Altman Z-Score
# ---------------------------------------------------------------------------

@dataclass
class AltmanResult:
    z: float | None
    coverage: float        # Fraction of 5 components that resolved
    components: dict = field(default_factory=dict)
    note: str | None = None


def compute_altman_z(
    info: dict | None,
    balance_sheet_statements: list[dict] | None = None,
    income_statements: list[dict] | None = None,
) -> AltmanResult:
    """Compute the original Altman (1968) public-firm Z-Score.

    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

    where:
      X1 = working capital / total assets
      X2 = retained earnings / total assets
      X3 = EBIT / total assets
      X4 = market cap / total liabilities
      X5 = sales / total assets

    Parameters
    ----------
    info :
        yfinance-style dict.  Keys consulted: ``totalAssets``,
        ``totalCurrentAssets``, ``totalCurrentLiabilities``,
        ``retainedEarnings``, ``ebit``, ``marketCap``, ``totalLiab``,
        ``totalDebt`` (fallback), ``totalRevenue``.
    balance_sheet_statements :
        FMP-style list (most recent first).  Used as fallback for any
        missing balance-sheet item.
    income_statements :
        FMP-style list.  Used as fallback for revenue / EBIT.
    """
    info = info or {}
    bs0 = (balance_sheet_statements or [None])[0]
    is0 = (income_statements or [None])[0]

    total_assets = _first(
        info.get("totalAssets"),
        _statement_field(bs0, "totalAssets"),
    )
    if not total_assets or total_assets <= 0:
        return AltmanResult(z=None, coverage=0.0, note="missing total assets")

    current_assets = _first(
        info.get("totalCurrentAssets"),
        _statement_field(bs0, "totalCurrentAssets"),
    )
    current_liab = _first(
        info.get("totalCurrentLiabilities"),
        _statement_field(bs0, "totalCurrentLiabilities"),
    )
    working_capital = (
        current_assets - current_liab
        if current_assets is not None and current_liab is not None
        else None
    )

    retained_earnings = _first(
        info.get("retainedEarnings"),
        _statement_field(bs0, "retainedEarnings"),
    )

    ebit = _first(
        info.get("ebit"),
        _statement_field(is0, "ebit", "operatingIncome"),
    )

    market_cap = _f(info.get("marketCap"))
    total_liab = _first(
        info.get("totalLiab"),
        _statement_field(bs0, "totalLiabilities", "totalLiab"),
        # Last-resort: total debt (under-counts but better than None)
        info.get("totalDebt"),
    )

    sales = _first(
        info.get("totalRevenue"),
        _statement_field(is0, "revenue", "totalRevenue"),
    )

    components = {
        "x1_wc_ta": (working_capital / total_assets) if working_capital is not None else None,
        "x2_re_ta": (retained_earnings / total_assets) if retained_earnings is not None else None,
        "x3_ebit_ta": (ebit / total_assets) if ebit is not None else None,
        "x4_mc_tl": (market_cap / total_liab) if market_cap is not None and total_liab and total_liab > 0 else None,
        "x5_sales_ta": (sales / total_assets) if sales is not None else None,
    }
    coverage = sum(1 for v in components.values() if v is not None) / 5.0
    if coverage < 3 / 5:    # Need at least 3 of 5 components
        return AltmanResult(z=None, coverage=coverage, components=components, note="insufficient coverage")

    coeffs = {"x1_wc_ta": 1.2, "x2_re_ta": 1.4, "x3_ebit_ta": 3.3, "x4_mc_tl": 0.6, "x5_sales_ta": 1.0}
    # Pro-rate missing-component weight to keep scale comparable
    present_weight = sum(coeffs[k] for k, v in components.items() if v is not None)
    total_weight = sum(coeffs.values())
    z_raw = sum(coeffs[k] * v for k, v in components.items() if v is not None)
    z = z_raw * (total_weight / present_weight) if present_weight > 0 else None

    return AltmanResult(z=z, coverage=coverage, components=components)


def altman_zone(z: float | None) -> str:
    """Return 'safe' | 'grey' | 'distress' | 'unknown' for an Altman Z."""
    if z is None or not math.isfinite(z):
        return "unknown"
    if z >= 2.99:
        return "safe"
    if z >= 1.81:
        return "grey"
    return "distress"


# ---------------------------------------------------------------------------
# Beneish M-Score
# ---------------------------------------------------------------------------

@dataclass
class BeneishResult:
    m: float | None
    coverage: float
    components: dict = field(default_factory=dict)
    note: str | None = None


def compute_beneish_m(
    info: dict | None,
    balance_sheet_statements: list[dict] | None = None,
    income_statements: list[dict] | None = None,
    cash_flow_statements: list[dict] | None = None,
) -> BeneishResult:
    """Compute the 8-variable Beneish (1999) M-Score.

    M = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
        + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI

    Requires both current and prior-year balance-sheet & income-statement.
    Returns ``None`` (with a coverage flag) when the prior period is missing,
    rather than raising — most stocks pass through here, and missing data
    must not silently block them.
    """
    bs = balance_sheet_statements or []
    inc = income_statements or []
    cf = cash_flow_statements or []
    if len(bs) < 2 or len(inc) < 2:
        return BeneishResult(m=None, coverage=0.0, note="needs current + prior period")

    bs_t, bs_p = bs[0], bs[1]
    inc_t, inc_p = inc[0], inc[1]
    cf_t = cf[0] if cf else None

    def _bs(s: dict, *k: str) -> float | None:
        return _statement_field(s, *k)

    def _inc(s: dict, *k: str) -> float | None:
        return _statement_field(s, *k)

    rev_t = _inc(inc_t, "revenue", "totalRevenue")
    rev_p = _inc(inc_p, "revenue", "totalRevenue")
    cogs_t = _inc(inc_t, "costOfRevenue", "costOfGoodsSold")
    cogs_p = _inc(inc_p, "costOfRevenue", "costOfGoodsSold")
    sga_t = _inc(inc_t, "sellingGeneralAndAdministrativeExpenses", "sga")
    sga_p = _inc(inc_p, "sellingGeneralAndAdministrativeExpenses", "sga")
    dep_t = _inc(inc_t, "depreciationAndAmortization")
    dep_p = _inc(inc_p, "depreciationAndAmortization")
    ni_t = _inc(inc_t, "netIncome")

    rec_t = _bs(bs_t, "netReceivables", "accountsReceivables")
    rec_p = _bs(bs_p, "netReceivables", "accountsReceivables")
    ta_t = _bs(bs_t, "totalAssets")
    ta_p = _bs(bs_p, "totalAssets")
    ppe_t = _bs(bs_t, "propertyPlantEquipmentNet", "propertyPlantEquipment")
    ppe_p = _bs(bs_p, "propertyPlantEquipmentNet", "propertyPlantEquipment")
    ca_t = _bs(bs_t, "totalCurrentAssets")
    ca_p = _bs(bs_p, "totalCurrentAssets")
    tl_t = _bs(bs_t, "totalLiabilities", "totalLiab")
    tl_p = _bs(bs_p, "totalLiabilities", "totalLiab")
    cl_t = _bs(bs_t, "totalCurrentLiabilities")
    cl_p = _bs(bs_p, "totalCurrentLiabilities")
    ltd_t = _bs(bs_t, "longTermDebt")
    ltd_p = _bs(bs_p, "longTermDebt")

    cfo_t = _statement_field(cf_t, "operatingCashFlow", "netCashProvidedByOperatingActivities") if cf_t else None

    def _safe_div(a, b):
        if a is None or b is None or b == 0:
            return None
        return a / b

    components: dict[str, float | None] = {}

    # DSRI — Days Sales in Receivables Index
    if rec_t is not None and rec_p is not None and rev_t and rev_p:
        components["DSRI"] = _safe_div(rec_t / rev_t, rec_p / rev_p)
    else:
        components["DSRI"] = None

    # GMI — Gross Margin Index (prior / current)
    if rev_t and rev_p and cogs_t is not None and cogs_p is not None:
        gm_t = (rev_t - cogs_t) / rev_t
        gm_p = (rev_p - cogs_p) / rev_p
        components["GMI"] = _safe_div(gm_p, gm_t)
    else:
        components["GMI"] = None

    # AQI — Asset Quality Index: non-current non-PPE assets / total assets, current vs prior
    if ta_t and ta_p and ppe_t is not None and ppe_p is not None and ca_t is not None and ca_p is not None:
        aq_t = 1.0 - (ca_t + ppe_t) / ta_t
        aq_p = 1.0 - (ca_p + ppe_p) / ta_p
        components["AQI"] = _safe_div(aq_t, aq_p)
    else:
        components["AQI"] = None

    # SGI — Sales Growth Index
    components["SGI"] = _safe_div(rev_t, rev_p)

    # DEPI — Depreciation Index: prior / current of dep / (dep + PPE)
    if dep_t is not None and dep_p is not None and ppe_t is not None and ppe_p is not None:
        depi_t_denom = dep_t + ppe_t
        depi_p_denom = dep_p + ppe_p
        if depi_t_denom and depi_p_denom:
            components["DEPI"] = _safe_div(dep_p / depi_p_denom, dep_t / depi_t_denom)
        else:
            components["DEPI"] = None
    else:
        components["DEPI"] = None

    # SGAI — SGA Index
    if sga_t is not None and sga_p is not None and rev_t and rev_p:
        components["SGAI"] = _safe_div(sga_t / rev_t, sga_p / rev_p)
    else:
        components["SGAI"] = None

    # TATA — Total Accruals to Total Assets
    if ni_t is not None and cfo_t is not None and ta_t:
        components["TATA"] = (ni_t - cfo_t) / ta_t
    elif ni_t is not None and ta_t and ca_t is not None and ca_p is not None and cl_t is not None and cl_p is not None:
        # Balance-sheet approximation when CFO unavailable
        d_ca = ca_t - ca_p
        d_cl = cl_t - cl_p
        components["TATA"] = (d_ca - d_cl) / ta_t
    else:
        components["TATA"] = None

    # LVGI — Leverage Index
    if tl_t is not None and tl_p is not None and ta_t and ta_p:
        components["LVGI"] = _safe_div(tl_t / ta_t, tl_p / ta_p)
    else:
        components["LVGI"] = None

    coeffs = {
        "DSRI": 0.92, "GMI": 0.528, "AQI": 0.404, "SGI": 0.892,
        "DEPI": 0.115, "SGAI": -0.172, "TATA": 4.679, "LVGI": -0.327,
    }
    present = [k for k, v in components.items() if v is not None and math.isfinite(v)]
    coverage = len(present) / len(coeffs)
    if coverage < 5 / 8:    # Need at least 5 of 8
        return BeneishResult(m=None, coverage=coverage, components=components, note="insufficient coverage")

    contrib = sum(coeffs[k] * components[k] for k in present)
    # Scale up to compensate for missing components
    weight_present = sum(abs(coeffs[k]) for k in present)
    weight_total = sum(abs(c) for c in coeffs.values())
    contrib_scaled = contrib * (weight_total / weight_present) if weight_present else contrib
    m = -4.84 + contrib_scaled

    return BeneishResult(m=m, coverage=coverage, components=components)


def beneish_likely_manipulator(m: float | None) -> bool:
    """True if M-score crosses Beneish's manipulation threshold."""
    if m is None or not math.isfinite(m):
        return False
    return m > -1.78


# ---------------------------------------------------------------------------
# Combined Tier-1 evaluator — returns the action ceiling
# ---------------------------------------------------------------------------

@dataclass
class DistressGateResult:
    """Per-candidate Tier-1 gate evaluation.

    Attributes
    ----------
    action_ceiling :
        Highest action label this candidate may receive based on Tier-1
        gates: "STRONG BUY" (no block), "BUY", "NEUTRAL", "AVOID".
    reasons :
        List of human-readable reasons for any downgrade.
    flags :
        Structured per-gate decisions (used by the threshold learner and
        for UI breakdown).
    """
    action_ceiling: str = "STRONG BUY"
    reasons: list = field(default_factory=list)
    flags: dict = field(default_factory=dict)


_RANK = {"STRONG BUY": 4, "BUY": 3, "NEUTRAL": 2, "AVOID": 1, "MANUAL REVIEW": 0}


def _cap(current: str, candidate: str) -> str:
    if _RANK.get(candidate, 4) < _RANK.get(current, 4):
        return candidate
    return current


def evaluate_distress_gates(
    *,
    altman_z: float | None,
    beneish_m: float | None,
    f_score: int | float | None,
    f_score_coverage: float | None,
    accruals_factor_score: float | None,
    investment_factor_score: float | None,    # asset-growth proxy
    net_debt_ebitda: float | None,
    config_module=None,
) -> DistressGateResult:
    """Apply all Tier-1 gates and return the strictest action ceiling.

    Each gate is independently toggleable via config.  Missing data never
    blocks (the gate is skipped); only confirmed failures downgrade.
    """
    cfg = config_module
    if cfg is None:
        import config as cfg    # type: ignore[no-redef]

    result = DistressGateResult()

    if not getattr(cfg, "ACTION_GATES_ENABLED", True):
        return result

    # Altman Z
    if getattr(cfg, "ALTMAN_Z_GATE_ENABLED", True) and altman_z is not None:
        z_strong = float(getattr(cfg, "ALTMAN_Z_STRONG_BUY_MIN", 2.6))
        z_buy = float(getattr(cfg, "ALTMAN_Z_BUY_MIN", 1.81))
        if altman_z < z_buy:
            result.action_ceiling = _cap(result.action_ceiling, "NEUTRAL")
            result.reasons.append(f"Altman Z={altman_z:.2f} in distress zone (<{z_buy})")
            result.flags["altman_z"] = "fail"
        elif altman_z < z_strong:
            result.action_ceiling = _cap(result.action_ceiling, "BUY")
            result.reasons.append(f"Altman Z={altman_z:.2f} in grey zone (<{z_strong})")
            result.flags["altman_z"] = "borderline"
        else:
            result.flags["altman_z"] = "pass"
    else:
        result.flags["altman_z"] = "skip"

    # Beneish M
    if getattr(cfg, "BENEISH_M_GATE_ENABLED", True) and beneish_m is not None:
        m_strong = float(getattr(cfg, "BENEISH_M_STRONG_BUY_MAX", -2.22))
        m_buy = float(getattr(cfg, "BENEISH_M_BUY_MAX", -1.78))
        if beneish_m > m_buy:
            result.action_ceiling = _cap(result.action_ceiling, "NEUTRAL")
            result.reasons.append(f"Beneish M={beneish_m:.2f} likely-manipulator (>{m_buy})")
            result.flags["beneish_m"] = "fail"
        elif beneish_m > m_strong:
            result.action_ceiling = _cap(result.action_ceiling, "BUY")
            result.reasons.append(f"Beneish M={beneish_m:.2f} elevated (>{m_strong})")
            result.flags["beneish_m"] = "borderline"
        else:
            result.flags["beneish_m"] = "pass"
    else:
        result.flags["beneish_m"] = "skip"

    # Piotroski F-Score
    f = _f(f_score)
    fcov = normalize_f_score_coverage(f_score_coverage)
    fcov_min = f_score_min_coverage(cfg)
    if f is not None and fcov is not None and fcov >= fcov_min:
        f_strong = float(getattr(cfg, "F_SCORE_STRONG_BUY_MIN", 6))
        f_buy = float(getattr(cfg, "F_SCORE_BUY_MIN", 5))
        if f < f_buy:
            result.action_ceiling = _cap(result.action_ceiling, "NEUTRAL")
            result.reasons.append(f"Piotroski F={int(f)}/9 weak fundamentals (<{int(f_buy)})")
            result.flags["f_score"] = "fail"
        elif f < f_strong:
            result.action_ceiling = _cap(result.action_ceiling, "BUY")
            result.reasons.append(f"Piotroski F={int(f)}/9 below STRONG-BUY floor (<{int(f_strong)})")
            result.flags["f_score"] = "borderline"
        else:
            result.flags["f_score"] = "pass"
    else:
        result.flags["f_score"] = "skip"

    # Accruals (high accruals = very negative score)
    if getattr(cfg, "ACCRUALS_GATE_ENABLED", True) and accruals_factor_score is not None:
        acc_min = float(getattr(cfg, "ACCRUALS_FACTOR_STRONG_BUY_MIN", -0.60))
        if accruals_factor_score < acc_min:
            result.action_ceiling = _cap(result.action_ceiling, "BUY")
            result.reasons.append(f"Accruals factor {accruals_factor_score:+.2f} below {acc_min:+.2f}")
            result.flags["accruals"] = "fail"
        else:
            result.flags["accruals"] = "pass"
    else:
        result.flags["accruals"] = "skip"

    # Asset growth (investment_factor_score: low growth = positive; very-negative = very-high growth)
    if investment_factor_score is not None:
        ag_min = float(getattr(cfg, "ASSET_GROWTH_FACTOR_STRONG_BUY_MIN", -0.60))
        if investment_factor_score < ag_min:
            result.action_ceiling = _cap(result.action_ceiling, "BUY")
            result.reasons.append(f"Asset growth factor {investment_factor_score:+.2f} below {ag_min:+.2f}")
            result.flags["asset_growth"] = "fail"
        else:
            result.flags["asset_growth"] = "pass"
    else:
        result.flags["asset_growth"] = "skip"

    # Net debt / EBITDA
    nde = _f(net_debt_ebitda)
    if nde is not None:
        cap = float(getattr(cfg, "NET_DEBT_EBITDA_STRONG_BUY_MAX", 3.0))
        if nde > cap:
            result.action_ceiling = _cap(result.action_ceiling, "BUY")
            result.reasons.append(f"Net debt/EBITDA {nde:.1f}x above {cap:.1f}x")
            result.flags["net_debt_ebitda"] = "fail"
        else:
            result.flags["net_debt_ebitda"] = "pass"
    else:
        result.flags["net_debt_ebitda"] = "skip"

    return result
