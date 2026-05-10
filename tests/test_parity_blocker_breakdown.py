from __future__ import annotations

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


def test_region_bucket_uses_suffix_and_us_default():
    assert breakdown._region_bucket("BHP.AX") == "Australia"
    assert breakdown._region_bucket("AAPL") == "US"
