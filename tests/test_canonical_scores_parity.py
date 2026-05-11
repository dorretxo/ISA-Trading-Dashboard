from __future__ import annotations

import pandas as pd
import pytest

from engine import historical_replay as replay
from engine.canonical_scores import (
    compute_ev_ebit_score,
    compute_f_score_score,
    compute_fcf_yield_score,
    compute_gpa_score,
)
from engine.factors import compute_factor_scores_from_result
from engine.institutional_prior import _component_scores
from utils.replay_live_parity import canonicalize_parity_row


def test_canonical_scalar_scores_midpoints_and_caps():
    assert compute_f_score_score(4.5) == 0.0
    assert compute_f_score_score(7.5) == 1.0
    assert compute_f_score_score(1.5) == -1.0
    assert compute_f_score_score(9) == 1.0

    assert compute_gpa_score(0.30) == 0.0
    assert compute_gpa_score(0.50) == 1.0
    assert compute_gpa_score(0.10) == pytest.approx(-1.0)

    assert compute_ev_ebit_score(0.10) == 0.0
    assert compute_ev_ebit_score(0.20) == 1.0
    assert compute_ev_ebit_score(0.00) == pytest.approx(-1.0)

    assert compute_fcf_yield_score(0.03) == 0.0
    assert compute_fcf_yield_score(0.11) == 1.0
    assert compute_fcf_yield_score(-0.05) == pytest.approx(-1.0)


def test_factor_canonicalization_is_idempotent():
    row = {
        "ticker": "ABC",
        "quality_score_fundamental": 0.2,
        "gpa": 0.40,
        "f_score": 7,
        "f_score_coverage": 1.0,
        "pe_ratio": 10.0,
        "fcf_yield": 0.07,
        "ev_ebit_score": 0.25,
        "momentum_score": 0.75,
        "vol_20d": 0.20,
    }

    once = canonicalize_parity_row(row)
    twice = canonicalize_parity_row(once)

    for key in ("f_score_score", "gpa_score", "quality_factor_score", "qmj_factor_score", "value_factor_score"):
        assert once[key] == twice[key]


def test_replay_fundamentals_use_same_factor_bundle(monkeypatch):
    monkeypatch.setattr(
        replay,
        "_PIT_STORE_CACHE",
        {
            "tickers": {
                "ABC": {
                    "2023-03-31": {
                        "_accepted_date": "2023-04-20",
                        "revenue": 900.0,
                        "net_income": 80.0,
                        "operating_cashflow": 120.0,
                        "capital_expenditure": -30.0,
                        "gross_profit": 260.0,
                        "total_assets": 950.0,
                        "total_debt": 80.0,
                        "cash": 20.0,
                        "ebit": 90.0,
                        "eps": 1.80,
                        "shares_outstanding": 100.0,
                    },
                    "2024-03-31": {
                        "_source": "fmp",
                        "_accepted_date": "2024-04-20",
                        "revenue": 1000.0,
                        "net_income": 100.0,
                        "operating_cashflow": 140.0,
                        "capital_expenditure": -40.0,
                        "gross_profit": 300.0,
                        "total_assets": 1000.0,
                        "total_debt": 100.0,
                        "cash": 25.0,
                        "ebit": 125.0,
                        "eps": 2.0,
                        "shares_outstanding": 100.0,
                    },
                }
            }
        },
    )

    features = replay._fundamental_features("ABC", pd.Timestamp("2024-06-30"), signal_price=10.0)
    expected = compute_factor_scores_from_result(features)

    assert features["quality_factor_score"] == expected["quality_factor_score"]
    assert features["qmj_factor_score"] == expected["qmj_factor_score"]
    assert features["value_factor_score"] == expected["value_factor_score"]


def test_institutional_prior_uses_canonical_f_score_scale():
    components, _coverage = _component_scores({"f_score": 6, "f_score_coverage": 1.0})
    expected_health = (compute_f_score_score(6) + 0.0 - (0.5 / 1.5) + 0.0) / 4.0
    assert components["health"] == pytest.approx(expected_health)
