"""Tests for engine/value_caps.py — Tier-2 value caps and Tier-3 quality floors."""

from __future__ import annotations

import pytest

from engine.value_caps import (
    compute_sector_medians,
    cross_sectional_percentile,
    evaluate_quality_floors,
    evaluate_value_caps,
)


# ---------------------------------------------------------------------------
# Tier 2 — value caps
# ---------------------------------------------------------------------------

def test_value_caps_clean_pass():
    res = evaluate_value_caps(
        ev_ebit=12.0, ev_sales=2.5, pe_forward=18.0,
        eps_growth_3y_cagr=0.10, sector_median_ev_ebit=14.0, qmj_percentile=0.60,
    )
    assert res.action_ceiling == "STRONG BUY"
    assert all(v == "pass" for k, v in res.flags.items())


def test_value_caps_trustpilot_profile_blocks_strong_buy():
    """TRST.L: EV/EBIT 78x, EV/Sales 4.8x, fwd P/E 49x — caps at NEUTRAL."""
    res = evaluate_value_caps(
        ev_ebit=78.0, ev_sales=4.8, pe_forward=49.0,
        eps_growth_3y_cagr=0.20, sector_median_ev_ebit=20.0, qmj_percentile=0.65,
    )
    assert res.action_ceiling == "NEUTRAL"
    assert any("EV/EBIT" in r for r in res.reasons)


def test_value_caps_ev_ebit_grey_zone_caps_at_buy():
    res = evaluate_value_caps(
        ev_ebit=35.0, ev_sales=3.0, pe_forward=22.0, sector_median_ev_ebit=18.0,
    )
    assert res.action_ceiling == "BUY"
    assert any("STRONG-BUY cap" in r for r in res.reasons)


def test_value_caps_ev_sales_lifted_by_high_qmj():
    res = evaluate_value_caps(
        ev_ebit=15.0, ev_sales=10.0, pe_forward=22.0, qmj_percentile=0.85,
    )
    assert res.action_ceiling == "STRONG BUY"
    assert res.flags["ev_sales"] == "pass-qmj-override"


def test_value_caps_pe_lifted_by_high_growth():
    res = evaluate_value_caps(
        ev_ebit=15.0, ev_sales=4.0, pe_forward=40.0, eps_growth_3y_cagr=0.30,
    )
    assert res.action_ceiling == "STRONG BUY"
    assert res.flags["pe_forward"] == "pass-growth-override"


def test_value_caps_skip_on_missing_data():
    res = evaluate_value_caps(ev_ebit=None, ev_sales=None, pe_forward=None)
    assert res.action_ceiling == "STRONG BUY"
    assert res.flags["ev_ebit"] == "skip"
    assert res.flags["ev_sales"] == "skip"
    assert res.flags["pe_forward"] == "skip"


def test_value_caps_sector_relative_tightens_global():
    """When sector median is lower than the global cap, the sector-relative
    cap binds first."""
    res = evaluate_value_caps(
        ev_ebit=22.0, ev_sales=3.0, pe_forward=22.0,
        sector_median_ev_ebit=12.0,    # 1.5x = 18x, tighter than 30x global
    )
    assert res.action_ceiling == "BUY"
    assert any("sector-relative" in r for r in res.reasons)


# ---------------------------------------------------------------------------
# Tier 3 — quality floors
# ---------------------------------------------------------------------------

def test_quality_floors_clean_pass():
    res = evaluate_quality_floors(
        gpa_percentile=0.65, qmj_percentile=0.70, roic=0.18, wacc=0.09,
        op_margin_yoy_delta=0.02,
    )
    assert res.action_ceiling == "STRONG BUY"


def test_quality_floors_seb_profile_blocks():
    """SEB: ROIC-WACC = 7.1% - 5.9% = 120bps (< 200bps), op-margin -230bps."""
    res = evaluate_quality_floors(
        gpa_percentile=0.55, qmj_percentile=0.55,
        roic=0.071, wacc=0.059, op_margin_yoy_delta=-0.023,
    )
    assert res.action_ceiling == "BUY"
    assert any("ROIC" in r for r in res.reasons)
    assert any("Op-margin" in r for r in res.reasons)


def test_quality_floors_low_gpa_blocks_strong_buy():
    res = evaluate_quality_floors(
        gpa_percentile=0.15, qmj_percentile=0.60, roic=0.20, wacc=0.10,
        op_margin_yoy_delta=0.0,
    )
    assert res.action_ceiling == "BUY"
    assert any("GPA" in r for r in res.reasons)


def test_quality_floors_skip_on_missing():
    res = evaluate_quality_floors(
        gpa_percentile=None, qmj_percentile=None, roic=None, wacc=None,
        op_margin_yoy_delta=None,
    )
    assert res.action_ceiling == "STRONG BUY"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_compute_sector_medians_skips_thin_sectors():
    rows = [{"sector": "Tech", "ev_ebit": v} for v in (10, 12, 14, 16, 18, 20, 22, 24)]
    rows.extend([{"sector": "Tiny", "ev_ebit": 5}, {"sector": "Tiny", "ev_ebit": 7}])
    medians = compute_sector_medians(rows)
    assert "Tech" in medians
    assert "Tiny" not in medians
    assert medians["Tech"] == pytest.approx(17.0)


def test_cross_sectional_percentile_rank():
    pct = cross_sectional_percentile([1, 2, 3, 4, None, 5])
    assert pct[0] == pytest.approx(1 / 5)
    assert pct[-1] == pytest.approx(5 / 5)
    assert pct[4] is None
