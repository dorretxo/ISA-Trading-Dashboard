from __future__ import annotations

from engine import discovery


def test_replay_like_stage2_features_populate_private_pit_fields():
    out = {}

    discovery._apply_replay_like_stage2_features(
        out,
        {
            "f_score": 8,
            "f_score_coverage": 0.9,
            "f_score_score": 1.0,
            "gpa": 0.48,
            "gross_profitability": 0.48,
            "gpa_score": 0.9,
            "quality_score_fundamental": 0.42,
            "value_factor_score": 0.31,
            "pe_ratio": 14.0,
            "fcf_yield": 0.06,
            "ebit_yield": 0.15,
            "ev_ebit_score": 0.5,
        },
    )

    assert out["_f_score"] == 8
    assert out["_f_score_coverage"] == 0.9
    assert out["_gpa"] == 0.48
    assert out["_gross_profitability"] == 0.48
    assert out["_pit_quality_score"] == 0.42
    assert out["_pit_value_score"] == 0.31
    assert out["_pe_ratio"] == 14.0
    assert out["_fcf_yield"] == 0.06
    assert out["_pit_ebit_yield"] == 0.15
    assert out["_ev_ebit_score"] == 0.5


def test_pit_factor_overrides_make_pit_values_authoritative():
    result = {
        "f_score": 3,
        "f_score_score": -0.5,
        "f_score_gate": False,
        "gpa": 0.10,
        "gross_profitability": 0.10,
        "gpa_score": -1.0,
        "quality_score_fundamental": -0.75,
        "pe_ratio": 80.0,
    }
    candidate = {
        "_f_score": 8,
        "_f_score_coverage": 1.0,
        "_gpa": 0.50,
        "_pit_quality_score": 0.35,
        "_pe_ratio": 12.0,
        "_pit_ebit_yield": 0.15,
    }

    out = discovery._apply_pit_factor_overrides(result, candidate)

    assert out["f_score"] == 8
    assert out["f_score_score"] == 1.0
    assert out["f_score_gate"] is True
    assert out["gpa"] == 0.50
    assert out["gross_profitability"] == 0.50
    assert out["_gross_profitability"] == 0.50
    assert out["gpa_score"] == 1.0
    assert out["quality_score_fundamental"] == 0.35
    assert out["pe_ratio"] == 12.0
    assert round(out["ev_ebit_score"], 4) == 0.5


def test_pit_factor_overrides_derive_pe_from_pit_earnings_yield():
    result = {"pe_ratio": 80.0}
    candidate = {"_pit_earnings_yield": 0.05}

    out = discovery._apply_pit_factor_overrides(result, candidate)

    assert out["pe_ratio"] == 20.0


def test_stage5b_metadata_fills_without_clobbering_pit_fields():
    candidate = {
        "_pe_ratio": 14.0,
        "_gross_profitability": 0.48,
        "_fcf_to_assets": None,
    }

    discovery._fill_missing_candidate_field(candidate, "_pe_ratio", 80.0)
    discovery._fill_missing_candidate_field(candidate, "_gross_profitability", 0.10)
    discovery._fill_missing_candidate_field(candidate, "_fcf_to_assets", 0.06)

    assert candidate["_pe_ratio"] == 14.0
    assert candidate["_gross_profitability"] == 0.48
    assert candidate["_fcf_to_assets"] == 0.06
