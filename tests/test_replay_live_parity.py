from __future__ import annotations

import pytest

import config
from utils.replay_live_parity import canonicalize_parity_row, comparable_fields, compare_parity_rows


def test_parity_comparison_uses_field_specific_tolerance(monkeypatch):
    monkeypatch.setattr(config, "REPLAY_LIVE_PARITY_EXCLUDED_FIELDS", [])
    monkeypatch.setattr(config, "REPLAY_LIVE_PARITY_FIELD_TOLERANCES", {"r_r_ratio": 0.75})

    compared = compare_parity_rows(
        {"r_r_ratio": 2.5},
        {"r_r_ratio": 2.0},
        ["r_r_ratio"],
        default_tolerance=0.15,
    )

    assert compared["drift_fields"] == []
    assert compared["comparisons"][0]["tolerance"] == pytest.approx(0.75)


def test_parity_canonicalizes_factor_scores_from_raw_inputs(monkeypatch):
    monkeypatch.setattr(config, "REPLAY_LIVE_PARITY_EXCLUDED_FIELDS", [])

    compared = compare_parity_rows(
        {
            "quality_factor_score": 0.0,
            "quality_score_fundamental": 0.0,
            "f_score": 9,
            "gpa": 0.50,
        },
        {
            "quality_factor_score": 0.0,
            "quality_score_fundamental": 0.0,
            "f_score": 9,
            "gpa": 0.50,
        },
        ["quality_factor_score", "qmj_factor_score"],
        default_tolerance=0.15,
    )

    assert compared["drift_fields"] == []
    assert compared["comparisons"][0]["live"] == pytest.approx(compared["comparisons"][0]["replay"])
    assert compared["comparisons"][1]["live"] == pytest.approx(compared["comparisons"][1]["replay"])


def test_comparable_fields_drops_configured_exclusions(monkeypatch):
    monkeypatch.setattr(config, "REPLAY_LIVE_PARITY_EXCLUDED_FIELDS", ["forecast_score"])

    assert comparable_fields(["technical_score", "forecast_score"]) == ["technical_score"]


def test_canonicalize_derives_ev_ebit_score_from_ev_ebit():
    row = canonicalize_parity_row({"ev_ebit": 5.0})

    assert row["ev_ebit_score"] == pytest.approx(1.0)


def test_canonicalize_nulls_live_only_inputs_before_derived_recompute(monkeypatch):
    """peg_ratio is yfinance-.info-only and has no PIT analog.  The
    canonicalizer must null it before recomputing derived factor scores
    so live and replay rows use the same component basis."""
    monkeypatch.setattr(
        config,
        "REPLAY_LIVE_PARITY_LIVE_ONLY_INPUTS_FOR_DERIVED",
        ["peg_ratio"],
    )

    row = canonicalize_parity_row({
        "peg_ratio": 1.0,
        "pe_ratio": 15.0,
        "ev_ebit_score": 0.5,
    })

    assert row["peg_ratio"] is None
    # value_factor_score still derivable from pe + ev_ebit components.
    assert row.get("value_factor_score") is not None


def test_canonicalize_value_factor_score_equal_when_only_peg_differs(monkeypatch):
    """Live row has peg_ratio, replay row does not.  After the canonicalizer
    nulls peg, the recomputed value_factor_score must match within tolerance.
    This is the regression pin for the live-only PEG leakage bug."""
    monkeypatch.setattr(
        config,
        "REPLAY_LIVE_PARITY_LIVE_ONLY_INPUTS_FOR_DERIVED",
        ["peg_ratio"],
    )
    monkeypatch.setattr(config, "REPLAY_LIVE_PARITY_EXCLUDED_FIELDS", [])

    live = {"peg_ratio": 1.0, "pe_ratio": 15.0, "ev_ebit_score": 0.5}
    replay = {"pe_ratio": 15.0, "ev_ebit_score": 0.5}

    compared = compare_parity_rows(
        live, replay, ["value_factor_score"], default_tolerance=0.05,
    )

    assert compared["drift_fields"] == []
    assert compared["comparisons"][0]["live"] == pytest.approx(
        compared["comparisons"][0]["replay"]
    )
