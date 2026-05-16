from __future__ import annotations

import daily_orchestrator as orchestrator


def test_live_parity_row_prefers_fresh_candidate_over_stale_db_row():
    row = orchestrator._parity_live_row_from_candidate(
        ticker="abc",
        candidate={
            "ticker": "ABC",
            "r_r_ratio": 0.42,
            "strong_buy_eligible": False,
            "pit_source": "fmp",
        },
        persisted_row={
            "ticker": "ABC",
            "run_date": "2026-05-01T08:00:00",
            "r_r_ratio": 2.0,
            "strong_buy_eligible": 1,
            "pit_source": "yfinance",
            "source": "discovery",
        },
        fallback_run_date="2026-05-15T23:18:25",
    )

    assert row["ticker"] == "ABC"
    assert row["source"] == "cached_discovery"
    assert row["run_date"] == "2026-05-15T23:18:25"
    assert row["r_r_ratio"] == 0.42
    assert row["strong_buy_eligible"] is False
    assert row["pit_source"] == "fmp"


def test_live_parity_row_uses_persisted_values_without_candidate():
    row = orchestrator._parity_live_row_from_candidate(
        ticker="ABC",
        candidate=None,
        persisted_row={
            "ticker": "ABC",
            "run_date": "2026-05-01T08:00:00",
            "r_r_ratio": 2.0,
            "source": "discovery",
        },
        fallback_run_date="2026-05-15T23:18:25",
    )

    assert row["source"] == "discovery"
    assert row["run_date"] == "2026-05-01T08:00:00"
    assert row["r_r_ratio"] == 2.0
