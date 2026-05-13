from __future__ import annotations

import pytest

from utils import parity_blocker_breakdown as breakdown


def test_field_breakdown_splits_missing_sides_and_drift():
    live_missing, replay_missing, both_missing, drifted = breakdown._field_breakdown(
        [
            {"field": "a", "status": "missing", "live": None, "replay": 1.0},
            {"field": "b", "status": "missing", "live": 1.0, "replay": None},
            {"field": "c", "status": "missing", "live": None, "replay": None},
            {"field": "d", "status": "drift", "live": 1.0, "replay": 0.0},
            {"field": "e", "status": "ok", "live": 1.0, "replay": 1.0},
        ]
    )

    assert live_missing == ["a"]
    assert replay_missing == ["b"]
    assert both_missing == ["c"]
    assert drifted == ["d"]


def test_fmp_and_non_us_ticker_classification():
    assert breakdown._is_fmp_statement_candidate("AAPL") is True
    assert breakdown._is_fmp_statement_candidate("AAPL.L") is False
    assert breakdown._is_non_us("AAPL.L") is True
    assert breakdown._is_non_us("AAPL") is False


def test_coverage_buckets_distinguish_unknown_low_and_legacy_zero():
    assert "unknown_coverage" in breakdown._coverage_buckets(
        {"f_score": None, "f_score_coverage": None},
        None,
    )
    assert "legacy_zero_coverage" in breakdown._coverage_buckets(
        {"f_score": None, "f_score_coverage": 0.0},
        None,
    )
    assert "computed_zero" in breakdown._coverage_buckets(
        {"f_score": 0, "f_score_coverage": 0.0},
        None,
    )
    buckets = breakdown._coverage_buckets({"f_score": 5, "f_score_coverage": 0.50}, None)
    assert "low_coverage" in buckets
    assert "threshold_disagreement" in buckets


def test_qmj_lite_component_count_matches_available_dimensions():
    assert breakdown._qmj_lite_component_count({}) == 0
    assert breakdown._qmj_lite_component_count({"gpa_score": 0.4}) == 1
    assert breakdown._qmj_lite_component_count({
        "gpa_score": 0.4,
        "f_score": 7,
        "f_score_coverage": 1.0,
        "earnings_stability": 0.2,
    }) == 3


def test_evidence_classification_distinguishes_yfinance_balance_only():
    assert breakdown.classify_evidence(
        ticker="TELIA.ST",
        pit_source="yfinance_info",
        qmj_components=1,
    ) == "yfinance_balance_only"
    assert breakdown.classify_evidence(
        ticker="TELIA.ST",
        pit_source="yfinance_quarterly",
        qmj_components=2,
    ) == "yfinance_partial"
    assert breakdown.classify_evidence(
        ticker="AAPL",
        pit_source="fmp",
        qmj_components=2,
    ) == "fmp_full"
    assert breakdown.classify_evidence(
        ticker="AAPL",
        pit_source="fmp",
        qmj_components=1,
    ) == "fmp_partial"


def test_quality_score_source_flags_opaque_live_fallbacks():
    assert breakdown._quality_score_source({"quality_score_fundamental": 0.85}) == (
        "live_info_fallback_without_components"
    )
    assert breakdown._quality_score_source({
        "quality_score_fundamental": 0.2,
        "gpa": 0.34,
        "pit_source": "fmp",
    }) == "fmp_component_backed_1"
    assert breakdown._quality_score_source({"quality_score_fundamental": None}) == "none"


def test_quality_source_family_ignores_component_count_noise():
    assert breakdown._quality_source_family("fmp_component_backed_6") == "pit_component_backed"
    assert breakdown._quality_source_family("unknown_component_backed_4") == "pit_component_backed"
    assert breakdown._quality_source_family("yfinance_quarterly_component_backed_4") == (
        "yfinance_component_backed"
    )
    assert breakdown._quality_source_family("none") == "none"


def test_actionable_drift_excludes_configured_evidence_classes(monkeypatch):
    class DummyConfig:
        REPLAY_LIVE_PARITY_NON_ACTIONABLE_EVIDENCE_CLASSES = [
            "yfinance_balance_only",
            "no_data",
        ]
        REPLAY_LIVE_PARITY_STRUCTURAL_DRIFT_FIELDS = []

    rows = [
        {
            "field": "quality_factor_score",
            "drifted_by_evidence": {"yfinance_balance_only": 80, "fmp_full": 2},
        },
        {
            "field": "value_factor_score",
            "drifted_by_evidence": {"yfinance_balance_only": 7},
        },
        {
            "field": "institutional_prior_score",
            "drifted_by_evidence": {"fmp_partial": 1, "no_data": 4},
        },
    ]

    assert breakdown._actionable_drift(rows, config_module=DummyConfig) == {
        "quality_factor_score": 2,
        "institutional_prior_score": 1,
    }


def test_non_actionable_evidence_classes_default_when_missing():
    class EmptyConfig:
        pass

    assert breakdown._non_actionable_evidence_classes(EmptyConfig) == {
        "yfinance_balance_only",
        "no_data",
    }


def test_structural_drift_fields_default_empty_when_missing():
    """No structural fields configured -> empty set; actionable behaviour
    unchanged for legacy configs."""
    class EmptyConfig:
        pass

    assert breakdown._structural_drift_fields(EmptyConfig) == set()


def test_actionable_drift_excludes_structural_fields():
    """Fields tagged as structural must NOT contribute to actionable counts
    even when they have actionable-class drift.  Routed instead to
    structural_drift_summary."""
    class DummyConfig:
        REPLAY_LIVE_PARITY_NON_ACTIONABLE_EVIDENCE_CLASSES = ["yfinance_balance_only"]
        REPLAY_LIVE_PARITY_STRUCTURAL_DRIFT_FIELDS = [
            "fundamental_score",
            "institutional_prior_score",
        ]

    rows = [
        {
            "field": "qmj_factor_score",
            "drifted_by_evidence": {"fmp_full": 3, "yfinance_balance_only": 40},
        },
        {
            "field": "fundamental_score",
            "drifted_by_evidence": {"fmp_full": 38},
        },
        {
            "field": "institutional_prior_score",
            "drifted_by_evidence": {"fmp_full": 17},
        },
    ]

    # Only qmj_factor_score's fmp_full count survives in actionable.
    assert breakdown._actionable_drift(rows, config_module=DummyConfig) == {
        "qmj_factor_score": 3,
    }


def test_structural_drift_separates_known_structural_fields():
    """`_structural_drift` returns only the drift on structural-tagged fields,
    using the same evidence-class filter as `_actionable_drift`."""
    class DummyConfig:
        REPLAY_LIVE_PARITY_NON_ACTIONABLE_EVIDENCE_CLASSES = ["yfinance_balance_only"]
        REPLAY_LIVE_PARITY_STRUCTURAL_DRIFT_FIELDS = [
            "fundamental_score",
            "institutional_prior_score",
        ]

    rows = [
        {
            "field": "qmj_factor_score",
            "drifted_by_evidence": {"fmp_full": 3},
        },
        {
            "field": "fundamental_score",
            "drifted_by_evidence": {"fmp_full": 38, "yfinance_balance_only": 25},
        },
        {
            "field": "institutional_prior_score",
            "drifted_by_evidence": {"fmp_full": 17},
        },
        {
            "field": "institutional_prior_percentile",  # not in structural list
            "drifted_by_evidence": {"fmp_full": 10},
        },
    ]

    # fmp_full counts on structural fields only; yfinance_balance_only filtered.
    assert breakdown._structural_drift(rows, config_module=DummyConfig) == {
        "fundamental_score": 38,
        "institutional_prior_score": 17,
    }


def test_value_factor_drilldown_row_compares_value_components():
    class DummyConfig:
        REPLAY_LIVE_PARITY_NON_ACTIONABLE_EVIDENCE_CLASSES = ["yfinance_balance_only"]

    row = breakdown._value_factor_drilldown_row(
        ticker="ALGM",
        live={
            "pit_source": "fmp",
            "value_factor_score": 0.6,
            "pe_ratio": 1.0,
            "fcf_yield": 0.07,
            "ev_ebit_score": 1.0,
        },
        replay={
            "pit_source": "fmp",
            "value_factor_score": 0.1,
            "pe_ratio": None,
            "fcf_yield": 0.07,
            "ev_ebit_score": -1.0,
        },
        fields=["value_factor_score", "pe_ratio", "fcf_yield", "ev_ebit_score"],
        evidence_class="fmp_full",
        source_bucket="fmp_us",
        region="US",
        component_drifted=[],
        live_missing=[],
        replay_missing=["pe_ratio"],
        both_missing=[],
        default_tolerance=0.15,
        config_module=DummyConfig,
    )

    assert row["actionable"] is True
    assert row["comparisons"]["value_factor_score"]["status"] == "drift"
    assert row["comparisons"]["value_factor_score"]["delta"] > 0.35
    assert row["comparisons"]["pe_ratio"]["status"] == "missing"
    assert row["comparisons"]["fcf_yield"]["status"] == "ok"
    assert row["live_pit_source"] == "fmp"
    assert row["replay_pit_source"] == "fmp"


def test_value_factor_drilldown_marks_non_actionable_evidence():
    class DummyConfig:
        REPLAY_LIVE_PARITY_NON_ACTIONABLE_EVIDENCE_CLASSES = ["yfinance_balance_only"]

    row = breakdown._value_factor_drilldown_row(
        ticker="NOKIA.HE",
        live={"value_factor_score": 0.6},
        replay={"value_factor_score": 0.1},
        fields=["value_factor_score"],
        evidence_class="yfinance_balance_only",
        source_bucket="global_non_us",
        region="Nordics",
        component_drifted=[],
        live_missing=[],
        replay_missing=[],
        both_missing=[],
        default_tolerance=0.15,
        config_module=DummyConfig,
    )

    assert row["actionable"] is False


def test_region_bucket_uses_suffix_and_us_default():
    assert breakdown._region_bucket("BHP.AX") == "Australia"
    assert breakdown._region_bucket("AAPL") == "US"
