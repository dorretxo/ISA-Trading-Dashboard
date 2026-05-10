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
