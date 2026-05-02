"""End-to-end integration test for engine/action_gates.py.

Verifies that the SEB SA and Trustpilot snapshots — which both topped the
screener as STRONG BUY in the April 2026 review — are correctly downgraded
once the Tier 1-4 gates are wired in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from engine.action_gates import (
    apply_action_gates,
    build_context,
    cap_action,
    evaluate_candidate,
)


# ---------------------------------------------------------------------------
# Lightweight stand-in for ScoredCandidate.  Only the attributes the gate
# orchestrator reads are required.
# ---------------------------------------------------------------------------

@dataclass
class _Cand:
    ticker: str
    sector: str = "Industrials"
    action: str = "STRONG BUY"
    aggregate_score: float = 0.5
    # Distress
    altman_z: float | None = None
    beneish_m: float | None = None
    f_score: int | None = None
    f_score_coverage: float = 1.0
    accruals_factor_score: float | None = None
    investment_factor_score: float | None = None
    net_debt_ebitda: float | None = None
    # Value
    ev_ebit: float | None = None
    ev_sales: float | None = None
    pe_ratio: float | None = None
    pe_forward: float | None = None
    eps_growth_3y_cagr: float | None = None
    # Quality
    qmj_factor_score: float | None = None
    gpa: float | None = None
    gpa_score: float | None = None
    roic: float | None = None
    wacc: float | None = None
    op_margin_yoy_delta: float | None = None
    # Momentum
    rsi: float | None = None
    price_vs_sma200_stretch: float | None = None
    entry_stance: str = "Ready"
    realized_vol_pctile: float | None = None
    sma_200: float | None = None
    current_price: float | None = None
    atr: float | None = None
    support_levels: dict = field(default_factory=dict)
    # State that gate may mutate
    action_gate_ceiling: str = "STRONG BUY"
    action_gate_reasons: list = field(default_factory=list)
    action_gate_flags: dict = field(default_factory=dict)
    limit_price: float | None = None
    limit_price_method: str | None = None
    limit_price_rationale: str | None = None
    strong_buy_eligible: bool = True
    ready_contract_reasons: list = field(default_factory=list)
    ready_contract_status: str = "PASS"


def _seb_candidate() -> _Cand:
    """SEB SA snapshot from the April 2026 framework review."""
    return _Cand(
        ticker="SK.PA", sector="Consumer Cyclical",
        altman_z=1.72, f_score=5, f_score_coverage=1.0,
        accruals_factor_score=-0.20, investment_factor_score=0.10,
        net_debt_ebitda=2.7,
        ev_ebit=10.2, ev_sales=0.65, pe_ratio=9.1, pe_forward=9.1,
        qmj_factor_score=0.10, gpa=0.42, roic=0.071, wacc=0.059,
        op_margin_yoy_delta=-0.023,
        rsi=69, price_vs_sma200_stretch=0.05, entry_stance="Ready",
        sma_200=51.0, current_price=53.65,
    )


def _trst_candidate() -> _Cand:
    """Trustpilot snapshot from the April 2026 framework review."""
    return _Cand(
        ticker="TRST.L", sector="Communication Services",
        altman_z=4.5, f_score=8, f_score_coverage=1.0,
        accruals_factor_score=0.30, investment_factor_score=0.15,
        net_debt_ebitda=-1.0,
        ev_ebit=78.0, ev_sales=4.8, pe_ratio=49.0, pe_forward=49.0,
        eps_growth_3y_cagr=0.20,
        qmj_factor_score=0.85, gpa=1.69, roic=0.20, wacc=0.10,
        op_margin_yoy_delta=0.05,
        rsi=70, price_vs_sma200_stretch=0.35, entry_stance="Pullback Preferred",
        sma_200=185.0, current_price=251.0, atr=8.0,
        support_levels={"placement": 214.0},
    )


# ---------------------------------------------------------------------------
# SEB regression — distress + F-score floor block STRONG BUY
# ---------------------------------------------------------------------------

def test_seb_profile_blocks_strong_buy_via_distress():
    candidates = [_seb_candidate()]
    summary = apply_action_gates(candidates)
    c = candidates[0]
    # Altman Z=1.72 (distress) -> NEUTRAL ceiling
    assert c.action_gate_ceiling == "NEUTRAL"
    assert any("Altman" in r for r in c.action_gate_reasons)
    assert any("Piotroski" in r for r in c.action_gate_reasons)
    assert summary["downgrades"] == 1


# ---------------------------------------------------------------------------
# TRST regression — value cap + entry stance block STRONG BUY
# ---------------------------------------------------------------------------

def test_trst_profile_blocks_strong_buy_via_value_cap_and_pullback():
    candidates = [_trst_candidate()]
    apply_action_gates(candidates)
    c = candidates[0]
    # EV/EBIT 78x exceeds BUY cap (50x) -> NEUTRAL ceiling
    assert c.action_gate_ceiling == "NEUTRAL"
    assert any("EV/EBIT" in r for r in c.action_gate_reasons)
    # Pullback Preferred should also surface
    assert any("Entry stance" in r for r in c.action_gate_reasons)
    # Limit-price suggestion produced because momentum gate flagged pullback
    assert c.limit_price is not None
    assert c.limit_price < c.current_price


# ---------------------------------------------------------------------------
# Healthy candidate keeps its STRONG BUY label
# ---------------------------------------------------------------------------

def test_clean_candidate_unaffected():
    c = _Cand(
        ticker="GOOD", sector="Technology",
        altman_z=4.5, f_score=8, f_score_coverage=1.0,
        accruals_factor_score=0.20, investment_factor_score=0.15,
        net_debt_ebitda=0.5,
        ev_ebit=15.0, ev_sales=3.5, pe_ratio=22.0, pe_forward=22.0,
        qmj_factor_score=0.80, gpa=0.6, roic=0.20, wacc=0.10,
        op_margin_yoy_delta=0.02,
        rsi=58, price_vs_sma200_stretch=0.10, entry_stance="Ready",
    )
    candidates = [c]
    apply_action_gates(candidates)
    assert c.action_gate_ceiling == "STRONG BUY"
    assert c.action == "STRONG BUY"    # was set as default; stays put


# ---------------------------------------------------------------------------
# Cap-action helper
# ---------------------------------------------------------------------------

def test_cap_action_picks_strictest():
    assert cap_action("STRONG BUY", "BUY") == "BUY"
    assert cap_action("BUY", "STRONG BUY") == "BUY"
    assert cap_action("STRONG BUY", "NEUTRAL") == "NEUTRAL"
    assert cap_action("STRONG BUY", "STRONG BUY") == "STRONG BUY"


# ---------------------------------------------------------------------------
# Batch context computes percentiles only when sample is large enough
# ---------------------------------------------------------------------------

def test_build_context_handles_small_batch():
    cs = [_Cand(ticker="A"), _Cand(ticker="B")]
    ctx = build_context(cs)
    # All percentile dicts have entries but values are None for tiny batches
    assert "A" in ctx.qmj_percentiles
    assert ctx.qmj_percentiles["A"] is None    # too few non-null QMJs


# ---------------------------------------------------------------------------
# Apply-gates respects the master toggle
# ---------------------------------------------------------------------------

def test_apply_action_gates_skipped_when_disabled():
    class _Cfg:
        ACTION_GATES_ENABLED = False
    candidates = [_seb_candidate()]
    summary = apply_action_gates(candidates, config_module=_Cfg)
    assert summary["applied"] == 0
    # No mutation
    assert candidates[0].action_gate_ceiling == "STRONG BUY"


# ---------------------------------------------------------------------------
# Regression — missing distress data must NOT be treated as a failure.  The
# initial wiring used safe_float() with default=0.0 which silently flipped
# every None into a fail (0.0 < 1.81 distress threshold; 0.0 > -1.78
# manipulator threshold).  This test pins the correct contract.
# ---------------------------------------------------------------------------

def test_candidate_with_no_distress_data_is_not_downgraded():
    c = _Cand(
        ticker="DARK", sector="Industrials",
        altman_z=None, beneish_m=None, f_score=None, f_score_coverage=0.0,
        accruals_factor_score=None, investment_factor_score=None,
        net_debt_ebitda=None,
        ev_ebit=None, ev_sales=None, pe_ratio=None, pe_forward=None,
        qmj_factor_score=None, gpa=None, roic=None, wacc=None,
        op_margin_yoy_delta=None,
        rsi=None, price_vs_sma200_stretch=None, entry_stance="Ready",
    )
    apply_action_gates([c])
    assert c.action_gate_ceiling == "STRONG BUY", (
        f"Empty/missing data must not block STRONG BUY (got {c.action_gate_ceiling}; "
        f"reasons={c.action_gate_reasons})"
    )
    # Every gate should record "skip", never "fail"
    for k, v in c.action_gate_flags.items():
        assert v != "fail", f"Gate '{k}' marked fail despite missing data"
