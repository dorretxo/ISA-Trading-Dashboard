"""Unit tests for engine.enterprise_factors.

Covers the mathematical core of items #1, #2, #4, #6, #7, #8 of the
enterprise-factor roadmap.  No yfinance/network access — purely synthetic.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from engine.enterprise_factors import (
    REGIME_FACTOR_TILTS,
    barroso_santa_clara_scale,
    compute_ev_ebit,
    compute_gpa,
    compute_piotroski_f_score,
    enterprise_value,
    orthogonalise,
    regime_factor_tilt,
    residual_momentum_12_1,
    sector_neutral_zscore,
)


# ---------------------------------------------------------------------------
# EV / EBIT
# ---------------------------------------------------------------------------

def test_enterprise_value_uses_yfinance_field_when_present():
    info = {"enterpriseValue": 1_500_000_000, "marketCap": 1_000_000_000, "totalDebt": 600_000_000, "totalCash": 100_000_000}
    assert enterprise_value(info) == 1_500_000_000


def test_enterprise_value_falls_back_to_components():
    info = {"marketCap": 1_000_000_000, "totalDebt": 600_000_000, "totalCash": 100_000_000}
    assert enterprise_value(info) == 1_500_000_000


def test_enterprise_value_returns_none_on_missing():
    assert enterprise_value({}) is None


def test_ev_ebit_basic():
    info = {"enterpriseValue": 1_000_000_000, "ebit": 100_000_000, "ebitda": 150_000_000}
    out = compute_ev_ebit(info)
    assert out["ev_ebit"] == 10.0
    assert out["ev_ebitda"] == pytest.approx(6.6667, rel=1e-3)
    assert out["ebit_yield"] == pytest.approx(0.10)
    # anchor at 10% yield => score ~ 0
    assert abs(out["ev_ebit_score"]) < 1e-9


def test_ev_ebit_deep_value_clips_to_one():
    info = {"enterpriseValue": 1_000_000_000, "ebit": 500_000_000}
    out = compute_ev_ebit(info)
    assert out["ev_ebit_score"] == 1.0  # clipped


def test_ev_ebit_negative_ebit_returns_none():
    info = {"enterpriseValue": 1_000_000_000, "ebit": -50_000_000}
    out = compute_ev_ebit(info)
    assert out["ev_ebit"] is None


# ---------------------------------------------------------------------------
# Piotroski F-score
# ---------------------------------------------------------------------------

def _period(**overrides):
    base = {
        "net_income": 100.0,
        "operating_cashflow": 150.0,
        "total_assets": 1000.0,
        "long_term_debt": 200.0,
        "current_assets": 500.0,
        "current_liabilities": 200.0,
        "shares_outstanding": 100.0,
        "gross_profit": 300.0,
        "revenue": 800.0,
    }
    base.update(overrides)
    return base


def test_f_score_all_pass_equals_nine():
    latest = _period(
        net_income=120,
        operating_cashflow=200,  # > NI
        long_term_debt=150,       # down from 200
        current_assets=600, current_liabilities=150,  # curr ratio up
        shares_outstanding=100,   # same
        gross_profit=360, revenue=900,  # GM up, turnover up
    )
    prior = _period()
    res = compute_piotroski_f_score(latest, prior)
    assert res["f_score"] == 9
    assert res["f_score_gate"] is True
    assert res["f_score_score"] == 1.0


def test_f_score_all_fail_equals_zero():
    # All checks must fail. Prior had revenue=800, assets=1000 (turnover 0.8).
    # Latest: revenue=600 (turnover 0.6 < 0.8), assets=1000 unchanged.
    latest = _period(
        net_income=-10, operating_cashflow=-20,
        long_term_debt=400,
        current_assets=300, current_liabilities=400,
        shares_outstanding=200,
        gross_profit=50, revenue=600,  # GM 0.083 < prior 0.375; turnover 0.6 < 0.8
    )
    prior = _period()
    res = compute_piotroski_f_score(latest, prior)
    assert res["f_score"] == 0
    assert res["f_score_gate"] is False


def test_f_score_gate_requires_six():
    # Construct exactly 5 passes -> gate False
    latest = _period(net_income=-10, operating_cashflow=150, long_term_debt=150)
    prior = _period()
    res = compute_piotroski_f_score(latest, prior)
    assert res["f_score"] < 6 or res["f_score_gate"] is (res["f_score"] >= 6)


def test_f_score_missing_returns_none():
    res = compute_piotroski_f_score(None, None)
    assert res["f_score"] is None
    assert res["f_score_gate"] is False


# ---------------------------------------------------------------------------
# GPA
# ---------------------------------------------------------------------------

def test_gpa_basic():
    info = {"grossProfits": 300, "totalAssets": 1000}
    out = compute_gpa(info)
    assert out["gpa"] == 0.30
    assert abs(out["gpa_score"]) < 1e-9


def test_gpa_missing():
    assert compute_gpa({})["gpa"] is None


# ---------------------------------------------------------------------------
# Residual momentum
# ---------------------------------------------------------------------------

def test_residual_momentum_zero_when_stock_equals_market():
    idx = pd.bdate_range("2023-01-01", periods=300)
    rng = np.random.default_rng(42)
    mkt = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(idx)))
    close = pd.Series(mkt, index=idx)
    market = pd.Series(mkt, index=idx)
    res = residual_momentum_12_1(close, market)
    assert res is not None
    # A β=1 perfect-match series has zero residual
    assert abs(res) < 1e-6


def test_residual_momentum_non_zero_with_idiosyncratic():
    idx = pd.bdate_range("2023-01-01", periods=300)
    rng = np.random.default_rng(123)
    mkt_ret = rng.normal(0, 0.01, len(idx))
    stock_ret = mkt_ret + rng.normal(0.001, 0.005, len(idx))  # +10bp daily alpha
    mkt = 100 * np.cumprod(1 + mkt_ret)
    stk = 100 * np.cumprod(1 + stock_ret)
    res = residual_momentum_12_1(pd.Series(stk, index=idx), pd.Series(mkt, index=idx))
    assert res is not None and res > 0  # positive alpha


def test_residual_momentum_insufficient_history():
    idx = pd.bdate_range("2023-01-01", periods=30)
    close = pd.Series(np.ones(30), index=idx)
    assert residual_momentum_12_1(close, close) is None


def test_barroso_santa_clara_down_scales_in_high_vol():
    # realised vol 36% vs 12% target → scale = 1/3
    out = barroso_santa_clara_scale(1.0, 0.36, target_vol=0.12)
    assert out == pytest.approx(1.0 / 3.0, rel=1e-9)


def test_barroso_santa_clara_clips_at_cap():
    # realised vol 1% vs 12% → scale would be 12, clipped at 3
    out = barroso_santa_clara_scale(1.0, 0.01, target_vol=0.12, cap=3.0)
    assert out == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Sector-neutral z-score
# ---------------------------------------------------------------------------

def test_sector_neutral_zscore_zero_mean():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    sectors = ["A"] * 5 + ["B"] * 5
    z = sector_neutral_zscore(values, sectors, min_peers=3)
    assert abs(float(np.mean(z))) < 1e-9


def test_sector_neutral_zscore_clips_to_winsor():
    # Single massive outlier
    values = [0.0] * 9 + [1000.0]
    sectors = ["A"] * 10
    z = sector_neutral_zscore(values, sectors, winsor=3.0, min_peers=3)
    assert float(np.max(np.abs(z))) <= 3.0 + 1e-9


def test_sector_neutral_zscore_handles_empty():
    z = sector_neutral_zscore([], [])
    assert z.shape == (0,)


def test_sector_neutral_zscore_handles_missing():
    values = [1.0, None, 3.0, float("nan"), 5.0, 6.0]
    sectors = ["A"] * 6
    z = sector_neutral_zscore(values, sectors, min_peers=3)
    assert z.shape == (6,)
    assert math.isfinite(float(z[1])) and math.isfinite(float(z[3]))


# ---------------------------------------------------------------------------
# Orthogonalisation
# ---------------------------------------------------------------------------

def test_orthogonalise_recovers_zero_when_target_is_linear_combo():
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=50)
    x2 = rng.normal(size=50)
    y = 2.0 * x1 - 3.0 * x2 + 1.5
    resid = orthogonalise(y, x1, x2)
    # Residual should be ~0
    assert float(np.max(np.abs(resid))) < 1e-9


def test_orthogonalise_preserves_orthogonal_component():
    rng = np.random.default_rng(11)
    x1 = rng.normal(size=100)
    extra = rng.normal(size=100)
    y = 2.0 * x1 + extra
    resid = orthogonalise(y, x1)
    # Residual should strongly correlate with `extra`
    c = np.corrcoef(resid, extra)[0, 1]
    assert c > 0.9


def test_orthogonalise_insufficient_rows_returns_zeros():
    resid = orthogonalise([1.0, 2.0], [0.5, 1.0])
    assert np.allclose(resid, 0.0)


# ---------------------------------------------------------------------------
# Regime tilts
# ---------------------------------------------------------------------------

def test_regime_tilt_normalises_to_one():
    base = {"value": 0.2, "quality": 0.2, "momentum": 0.2, "low_vol": 0.2, "f_score": 0.2}
    for reg in ("BULL", "NEUTRAL", "BEAR"):
        out = regime_factor_tilt(base, reg)
        assert abs(sum(out.values()) - 1.0) < 1e-9


def test_regime_tilt_bear_reduces_momentum():
    base = {"value": 0.2, "quality": 0.2, "momentum": 0.2, "low_vol": 0.2, "f_score": 0.2}
    bear = regime_factor_tilt(base, "BEAR")
    bull = regime_factor_tilt(base, "BULL")
    assert bear["momentum"] < bull["momentum"]
    assert bear["quality"] > bull["quality"]
    assert bear["low_vol"] > bull["low_vol"]


def test_regime_tilt_unknown_label_falls_back_to_neutral():
    base = {"value": 0.5, "quality": 0.5}
    out = regime_factor_tilt(base, "MYSTERY")
    assert abs(sum(out.values()) - 1.0) < 1e-9
    # Should equal neutral normalisation (50/50)
    assert abs(out["value"] - 0.5) < 1e-9
