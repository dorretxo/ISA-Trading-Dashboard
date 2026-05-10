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
    f_score_coverage: float | None = 1.0
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
    qmj_component_count: int | None = None
    pit_source: str | None = None
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
    is_parabolic: bool = False
    r_r_ratio: float | None = None
    ready_contract_core_status: str = "FAIL"
    ready_contract_score: float | None = None
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


def test_core_ready_candidate_can_soft_pass_moderate_stretch():
    c = _Cand(
        ticker="READY", sector="Technology",
        altman_z=4.5, f_score=8, f_score_coverage=1.0,
        net_debt_ebitda=0.5,
        ev_ebit=18.0, pe_ratio=24.0, pe_forward=24.0,
        qmj_factor_score=0.90, gpa=0.6, roic=0.20, wacc=0.10,
        rsi=62, price_vs_sma200_stretch=0.33, entry_stance="Ready",
        r_r_ratio=2.5, ready_contract_core_status="PASS", ready_contract_score=1.0,
    )

    class _Cfg:
        ACTION_GATES_ENABLED = True
        ACTION_GATES_CORE_READY_STRETCH_OVERRIDE_ENABLED = True
        ACTION_GATES_CORE_READY_STRETCH_MAX = 0.40
        ACTION_GATES_CORE_READY_STRETCH_MIN_RR = 2.0
        ACTION_GATES_CORE_READY_STRETCH_MIN_SCORE = 0.95
        STRETCH_200DMA_STRONG_BUY_MAX = 0.25
        RSI_STRONG_BUY_MAX = 70
        RSI_NEUTRAL_CAP = 80
        ENTRY_STANCE_BLOCKS_STRONG_BUY = True
        MOMENTUM_VOL_SCALING_ENABLED = True

    apply_action_gates([c], config_module=_Cfg)

    assert c.action_gate_ceiling == "STRONG BUY"
    assert c.action_gate_flags["stretch_200dma"] == "soft_pass"
    assert not any("200-DMA exceeds cap" in reason for reason in c.action_gate_reasons)


def test_core_ready_stretch_override_respects_rsi_warning():
    c = _Cand(
        ticker="HOT", sector="Technology",
        altman_z=4.5, f_score=8, f_score_coverage=1.0,
        ev_ebit=18.0, pe_ratio=24.0, pe_forward=24.0,
        qmj_factor_score=0.90, gpa=0.6,
        rsi=76, price_vs_sma200_stretch=0.33, entry_stance="Ready",
        r_r_ratio=2.5, ready_contract_core_status="PASS", ready_contract_score=1.0,
    )

    class _Cfg:
        ACTION_GATES_ENABLED = True
        ACTION_GATES_CORE_READY_STRETCH_OVERRIDE_ENABLED = True
        ACTION_GATES_CORE_READY_STRETCH_MAX = 0.40
        ACTION_GATES_CORE_READY_STRETCH_MIN_RR = 2.0
        ACTION_GATES_CORE_READY_STRETCH_MIN_SCORE = 0.95
        STRETCH_200DMA_STRONG_BUY_MAX = 0.25
        RSI_STRONG_BUY_MAX = 70
        RSI_NEUTRAL_CAP = 80
        ENTRY_STANCE_BLOCKS_STRONG_BUY = True
        MOMENTUM_VOL_SCALING_ENABLED = True

    apply_action_gates([c], config_module=_Cfg)

    assert c.action_gate_ceiling == "BUY"
    assert c.action_gate_flags["rsi"] == "borderline"


def test_clean_candidate_remains_strong_buy_under_conservative_profile():
    from engine.threshold_learner import apply_overlay
    import config

    c = _Cand(
        ticker="CLEANCON", sector="Technology",
        altman_z=5.0, f_score=9, f_score_coverage=1.0,
        accruals_factor_score=0.50, investment_factor_score=0.30,
        net_debt_ebitda=0.2,
        ev_ebit=16.0, ev_sales=3.5, pe_ratio=20.0, pe_forward=20.0,
        eps_growth_3y_cagr=0.18,
        qmj_factor_score=0.95, gpa=0.85, gpa_score=0.90,
        roic=0.28, wacc=0.10, op_margin_yoy_delta=0.04,
        rsi=58, price_vs_sma200_stretch=0.12, entry_stance="Ready",
        r_r_ratio=2.5, ready_contract_core_status="PASS", ready_contract_score=1.0,
    )

    apply_action_gates([c], config_module=apply_overlay(config, "conservative"))

    assert c.action_gate_ceiling == "STRONG BUY"
    assert c.action_gate_reasons == []


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


def test_quality_percentiles_use_sector_bucket_when_available():
    industrials = [
        _Cand(
            ticker=f"IND{i}",
            sector="Industrials",
            qmj_factor_score=0.10 + i * 0.10,
            gpa_score=0.10 + i * 0.10,
            altman_z=4.0,
            f_score=7,
            f_score_coverage=1.0,
            ev_ebit=14.0,
            pe_forward=18.0,
            entry_stance="Ready",
        )
        for i in range(8)
    ]
    technology = [
        _Cand(
            ticker=f"TECH{i}",
            sector="Technology",
            qmj_factor_score=0.90 + i * 0.10,
            gpa_score=0.90 + i * 0.10,
            altman_z=4.0,
            f_score=7,
            f_score_coverage=1.0,
            ev_ebit=14.0,
            pe_forward=18.0,
            entry_stance="Ready",
        )
        for i in range(8)
    ]

    apply_action_gates(industrials + technology)
    target = industrials[-1]

    assert target.action_gate_ceiling == "STRONG BUY"
    assert target.action_gate_flags["qmj_pctile"] == "pass"
    assert target.action_gate_flags["gpa_pctile"] == "pass"


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

def test_candidate_with_no_distress_data_is_not_downgraded_when_quality_is_evaluable():
    c = _Cand(
        ticker="DARK", sector="Industrials",
        altman_z=None, beneish_m=None, f_score=None, f_score_coverage=None,
        accruals_factor_score=None, investment_factor_score=None,
        net_debt_ebitda=None,
        ev_ebit=None, ev_sales=None, pe_ratio=None, pe_forward=None,
        qmj_factor_score=None, gpa=None, roic=0.14, wacc=0.07,
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


def test_all_core_fundamental_quality_missing_caps_fmp_style_to_neutral():
    c = _Cand(
        ticker="BLIND", sector="Industrials",
        altman_z=None, beneish_m=None, f_score=None, f_score_coverage=None,
        accruals_factor_score=None, investment_factor_score=None,
        net_debt_ebitda=None,
        ev_ebit=None, ev_sales=None, pe_ratio=None, pe_forward=None,
        qmj_factor_score=None, gpa=None, roic=None, wacc=None,
        op_margin_yoy_delta=None,
        rsi=None, price_vs_sma200_stretch=None, entry_stance="Ready",
    )

    apply_action_gates([c])

    assert c.action_gate_ceiling == "NEUTRAL"
    assert c.action_gate_flags["fundamental_coverage"] == "fail"
    assert c.action_gate_flags["fundamental_coverage_evidence"] == "fmp_partial"
    assert any("FMP-style fundamentals missing" in r for r in c.action_gate_reasons)


def test_source_aware_non_us_balance_only_quality_caps_to_buy():
    c = _Cand(
        ticker="THIN.L", sector="Industrials",
        pit_source="yfinance_quarterly",
        qmj_component_count=1,
        altman_z=None, beneish_m=None, f_score=None, f_score_coverage=None,
        accruals_factor_score=None, investment_factor_score=None,
        net_debt_ebitda=None,
        ev_ebit=None, ev_sales=None, pe_ratio=None, pe_forward=None,
        qmj_factor_score=None, gpa=0.12, roic=None, wacc=None,
        op_margin_yoy_delta=None,
        rsi=None, price_vs_sma200_stretch=None, entry_stance="Ready",
    )

    apply_action_gates([c])

    assert c.action_gate_ceiling == "BUY"
    assert c.action_gate_flags["fundamental_coverage"] == "fail"
    assert c.action_gate_flags["fundamental_coverage_evidence"] == "yfinance_balance_only"
    assert any("Thin yfinance quality evidence" in r for r in c.action_gate_reasons)


def test_source_aware_fmp_style_yfinance_balance_only_caps_to_neutral():
    c = _Cand(
        ticker="PLAIN", sector="Industrials",
        pit_source="yfinance_info",
        qmj_component_count=1,
        altman_z=None, beneish_m=None, f_score=None, f_score_coverage=None,
        accruals_factor_score=None, investment_factor_score=None,
        net_debt_ebitda=None,
        ev_ebit=None, ev_sales=None, pe_ratio=None, pe_forward=None,
        qmj_factor_score=None, gpa=0.12, roic=None, wacc=None,
        op_margin_yoy_delta=None,
        rsi=None, price_vs_sma200_stretch=None, entry_stance="Ready",
    )

    apply_action_gates([c])

    assert c.action_gate_ceiling == "NEUTRAL"
    assert c.action_gate_flags["fundamental_coverage"] == "fail"
    assert c.action_gate_flags["fundamental_coverage_evidence"] == "yfinance_balance_only"


def test_source_aware_non_us_with_two_quality_components_uses_normal_gates():
    c = _Cand(
        ticker="COVERED.L", sector="Industrials",
        pit_source="yfinance_quarterly",
        qmj_component_count=2,
        altman_z=None, beneish_m=None, f_score=7, f_score_coverage=1.0,
        accruals_factor_score=None, investment_factor_score=None,
        net_debt_ebitda=None,
        ev_ebit=None, ev_sales=None, pe_ratio=None, pe_forward=None,
        qmj_factor_score=0.8, gpa=0.45, roic=None, wacc=None,
        op_margin_yoy_delta=None,
        rsi=None, price_vs_sma200_stretch=None, entry_stance="Ready",
    )

    apply_action_gates([c])

    assert c.action_gate_ceiling == "STRONG BUY"
    assert c.action_gate_flags.get("fundamental_coverage") != "fail"
    assert c.action_gate_flags["fundamental_coverage_evidence"] == "yfinance_partial"


def test_source_aware_gate_can_fall_back_to_legacy_buy_cap():
    class _Cfg:
        SOURCE_AWARE_COVERAGE_ENABLED = False
        FUNDAMENTAL_COVERAGE_GATE_ENABLED = True
        FUNDAMENTAL_COVERAGE_FAIL_CAP = "BUY"

    c = _Cand(
        ticker="BLIND", sector="Industrials",
        altman_z=None, beneish_m=None, f_score=None, f_score_coverage=None,
        accruals_factor_score=None, investment_factor_score=None,
        net_debt_ebitda=None,
        ev_ebit=None, ev_sales=None, pe_ratio=None, pe_forward=None,
        qmj_factor_score=None, gpa=None, roic=None, wacc=None,
        op_margin_yoy_delta=None,
        rsi=None, price_vs_sma200_stretch=None, entry_stance="Ready",
    )

    apply_action_gates([c], config_module=_Cfg)

    assert c.action_gate_ceiling == "BUY"
    assert c.action_gate_flags["fundamental_coverage"] == "fail"
    assert "fundamental_coverage_evidence" not in c.action_gate_flags
