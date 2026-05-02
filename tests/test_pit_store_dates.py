from pathlib import Path

from utils.pit_store import latest_as_of, record_snapshot


def test_latest_as_of_uses_accepted_date_when_present(tmp_path: Path):
    path = tmp_path / "pit.json"
    record_snapshot(
        "AAPL",
        "2024-03-31",
        {"net_income": 1.0, "total_assets": 10.0},
        accepted_date="2024-05-20 18:00:00",
        path=path,
    )

    before, before_date = latest_as_of("AAPL", "2024-05-19", lag_days=1, path=path)
    after, after_date = latest_as_of("AAPL", "2024-05-20", lag_days=1, path=path)

    assert before is None
    assert before_date is None
    assert after is not None
    assert after_date == "2024-03-31"


def test_latest_as_of_falls_back_to_report_lag(tmp_path: Path):
    path = tmp_path / "pit.json"
    record_snapshot(
        "MSFT",
        "2024-03-31",
        {"net_income": 1.0, "total_assets": 10.0},
        path=path,
    )

    hidden, _ = latest_as_of("MSFT", "2024-04-10", lag_days=20, path=path)
    visible, report_date = latest_as_of("MSFT", "2024-04-20", lag_days=20, path=path)

    assert hidden is None
    assert visible is not None
    assert report_date == "2024-03-31"


def test_record_snapshot_tags_source(tmp_path: Path):
    path = tmp_path / "pit.json"
    record_snapshot(
        "NESN.SW",
        "2024-06-30",
        {"net_income": 2.0, "total_assets": 20.0},
        source="yfinance_quarterly",
        path=path,
    )

    visible, _ = latest_as_of("NESN.SW", "2024-08-20", path=path)

    assert visible is not None
    assert visible["_source"] == "yfinance_quarterly"
