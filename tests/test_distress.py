"""Tests for engine/distress.py — Altman Z, Beneish M, and the combined gate."""

from __future__ import annotations

import math

import pytest

from engine.distress import (
    altman_zone,
    beneish_likely_manipulator,
    compute_altman_z,
    compute_beneish_m,
    evaluate_distress_gates,
)


# ---------------------------------------------------------------------------
# Altman Z
# ---------------------------------------------------------------------------

def test_altman_z_safe_zone_basic():
    """Healthy balance sheet should land safely above 2.99."""
    info = {
        "totalAssets": 1_000.0,
        "totalCurrentAssets": 400.0,
        "totalCurrentLiabilities": 100.0,
        "retainedEarnings": 350.0,
        "ebit": 200.0,
        "marketCap": 1_500.0,
        "totalLiab": 300.0,
        "totalRevenue": 1_200.0,
    }
    res = compute_altman_z(info)
    assert res.z is not None
    assert res.coverage == pytest.approx(1.0)
    assert res.z > 2.99
    assert altman_zone(res.z) == "safe"


def test_altman_z_distress_zone_basic():
    """Heavy leverage + low retained earnings + thin margins => distress."""
    info = {
        "totalAssets": 1_000.0,
        "totalCurrentAssets": 200.0,
        "totalCurrentLiabilities": 250.0,    # negative working capital
        "retainedEarnings": 20.0,
        "ebit": 30.0,
        "marketCap": 200.0,
        "totalLiab": 800.0,                  # high leverage
        "totalRevenue": 600.0,
    }
    res = compute_altman_z(info)
    assert res.z is not None
    assert res.z < 1.81
    assert altman_zone(res.z) == "distress"


def test_altman_z_handles_missing_market_cap():
    info = {
        "totalAssets": 1_000.0,
        "totalCurrentAssets": 400.0,
        "totalCurrentLiabilities": 100.0,
        "retainedEarnings": 200.0,
        "ebit": 100.0,
        "totalLiab": 400.0,
        "totalRevenue": 800.0,
    }
    res = compute_altman_z(info)
    assert res.z is not None
    assert res.coverage == pytest.approx(4 / 5)


def test_altman_z_returns_none_when_data_too_thin():
    res = compute_altman_z({"totalAssets": 100.0})
    assert res.z is None


def test_altman_zone_unknown_for_none():
    assert altman_zone(None) == "unknown"
    assert altman_zone(float("nan")) == "unknown"


# ---------------------------------------------------------------------------
# Beneish M
# ---------------------------------------------------------------------------

def _make_steady_statements():
    """Steady-state company — DSRI ~1, GMI ~1, etc.  M should be very negative."""
    inc = [
        {"revenue": 1100.0, "costOfRevenue": 700.0, "sellingGeneralAndAdministrativeExpenses": 100.0,
         "depreciationAndAmortization": 50.0, "netIncome": 200.0},
        {"revenue": 1000.0, "costOfRevenue": 640.0, "sellingGeneralAndAdministrativeExpenses": 90.0,
         "depreciationAndAmortization": 50.0, "netIncome": 180.0},
    ]
    bs = [
        {"netReceivables": 110.0, "totalAssets": 2000.0, "propertyPlantEquipmentNet": 800.0,
         "totalCurrentAssets": 600.0, "totalLiabilities": 800.0, "totalCurrentLiabilities": 200.0,
         "longTermDebt": 400.0},
        {"netReceivables": 100.0, "totalAssets": 1900.0, "propertyPlantEquipmentNet": 780.0,
         "totalCurrentAssets": 580.0, "totalLiabilities": 760.0, "totalCurrentLiabilities": 195.0,
         "longTermDebt": 380.0},
    ]
    cf = [{"operatingCashFlow": 220.0}]
    return inc, bs, cf


def test_beneish_steady_state_below_manipulator_threshold():
    inc, bs, cf = _make_steady_statements()
    res = compute_beneish_m(info={}, balance_sheet_statements=bs,
                            income_statements=inc, cash_flow_statements=cf)
    assert res.m is not None
    assert res.m < -1.78
    assert not beneish_likely_manipulator(res.m)


def test_beneish_aggressive_revenue_pump_flags_manipulator():
    """Receivables and revenue spike together while CFO collapses."""
    inc = [
        {"revenue": 2000.0, "costOfRevenue": 700.0, "sellingGeneralAndAdministrativeExpenses": 100.0,
         "depreciationAndAmortization": 50.0, "netIncome": 600.0},
        {"revenue": 1000.0, "costOfRevenue": 640.0, "sellingGeneralAndAdministrativeExpenses": 90.0,
         "depreciationAndAmortization": 50.0, "netIncome": 180.0},
    ]
    bs = [
        {"netReceivables": 600.0, "totalAssets": 2200.0, "propertyPlantEquipmentNet": 800.0,
         "totalCurrentAssets": 800.0, "totalLiabilities": 1100.0, "totalCurrentLiabilities": 250.0,
         "longTermDebt": 600.0},
        {"netReceivables": 100.0, "totalAssets": 1900.0, "propertyPlantEquipmentNet": 780.0,
         "totalCurrentAssets": 580.0, "totalLiabilities": 760.0, "totalCurrentLiabilities": 195.0,
         "longTermDebt": 380.0},
    ]
    cf = [{"operatingCashFlow": 50.0}]    # very low CFO vs net income => high TATA
    res = compute_beneish_m(info={}, balance_sheet_statements=bs,
                            income_statements=inc, cash_flow_statements=cf)
    assert res.m is not None
    assert res.m > -1.78    # crosses the manipulator threshold
    assert beneish_likely_manipulator(res.m)


def test_beneish_returns_none_without_prior_period():
    res = compute_beneish_m(info={}, balance_sheet_statements=[{"totalAssets": 1.0}],
                            income_statements=[{"revenue": 1.0}])
    assert res.m is None


# ---------------------------------------------------------------------------
# Combined gate
# ---------------------------------------------------------------------------

def test_evaluate_distress_gates_all_pass_keeps_strong_buy():
    res = evaluate_distress_gates(
        altman_z=3.5, beneish_m=-2.5, f_score=8, f_score_coverage=1.0,
        accruals_factor_score=0.2, investment_factor_score=0.1, net_debt_ebitda=1.5,
    )
    assert res.action_ceiling == "STRONG BUY"
    assert res.flags["altman_z"] == "pass"
    assert res.flags["f_score"] == "pass"


def test_evaluate_distress_gates_seb_profile_blocks_strong_buy():
    """SEB SA snapshot: Z=1.72, F=5/9, net debt/EBITDA 2.7x."""
    res = evaluate_distress_gates(
        altman_z=1.72, beneish_m=-2.4, f_score=5, f_score_coverage=1.0,
        accruals_factor_score=0.0, investment_factor_score=0.0, net_debt_ebitda=2.7,
    )
    # Z=1.72 is *below* 1.81 (distress) -> NEUTRAL
    assert res.action_ceiling == "NEUTRAL"
    assert any("Altman" in r for r in res.reasons)
    assert any("Piotroski" in r for r in res.reasons)


def test_evaluate_distress_gates_grey_zone_caps_at_buy():
    res = evaluate_distress_gates(
        altman_z=2.2, beneish_m=-2.5, f_score=7, f_score_coverage=1.0,
        accruals_factor_score=0.0, investment_factor_score=0.0, net_debt_ebitda=1.5,
    )
    assert res.action_ceiling == "BUY"
    assert any("grey zone" in r for r in res.reasons)


def test_evaluate_distress_gates_skips_when_data_missing():
    """No data should mean no downgrade — debiased by the threshold learner over time."""
    res = evaluate_distress_gates(
        altman_z=None, beneish_m=None, f_score=None, f_score_coverage=0.0,
        accruals_factor_score=None, investment_factor_score=None, net_debt_ebitda=None,
    )
    assert res.action_ceiling == "STRONG BUY"
    assert res.flags["altman_z"] == "skip"


def test_evaluate_distress_gates_high_net_debt_caps_at_buy():
    res = evaluate_distress_gates(
        altman_z=3.0, beneish_m=-2.5, f_score=7, f_score_coverage=1.0,
        accruals_factor_score=0.0, investment_factor_score=0.0, net_debt_ebitda=4.5,
    )
    assert res.action_ceiling == "BUY"
    assert any("Net debt" in r for r in res.reasons)
