from utils.ticker_diagnostics import _replay_live_parity


def test_replay_live_parity_flags_factor_drift():
    rows = [
        {
            "source": "discovery",
            "run_date": "2026-04-04",
            "value_factor_score": -0.05,
            "quality_factor_score": 0.40,
        },
        {
            "source": "replay_pit_v1",
            "run_date": "2026-03-31",
            "value_factor_score": -0.70,
            "quality_factor_score": -0.32,
        },
    ]

    parity = _replay_live_parity(rows, tolerance=0.15)

    assert parity["available"] is True
    assert "value_factor_score" in parity["drift_fields"]
    assert "quality_factor_score" in parity["drift_fields"]


def test_replay_live_parity_marks_stale_replay_instead_of_drift():
    rows = [
        {
            "source": "discovery",
            "run_date": "2026-05-03",
            "value_factor_score": -0.05,
            "quality_factor_score": 0.40,
        },
        {
            "source": "replay_pit_v1",
            "run_date": "2026-03-31",
            "value_factor_score": -0.70,
            "quality_factor_score": -0.32,
        },
    ]

    parity = _replay_live_parity(rows, tolerance=0.15)

    assert parity["available"] is False
    assert "stale replay row" in parity["reason"]
    assert parity["date_gap_days"] == 33
    assert parity["comparisons"] == []
