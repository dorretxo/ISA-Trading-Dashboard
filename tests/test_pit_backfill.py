import pytest

from utils import pit_backfill
from utils.pit_backfill import _snapshot_from_rows


def test_snapshot_from_rows_keeps_period_fundamentals_only():
    snapshot = _snapshot_from_rows(
        {
            "date": "2026-03-31",
            "netIncome": "120",
            "grossProfit": 300,
            "revenue": 800,
            "weightedAverageShsOut": 50,
        },
        {
            "date": "2026-03-31",
            "totalAssets": 1000,
            "longTermDebt": 200,
            "totalCurrentAssets": 400,
            "totalCurrentLiabilities": 250,
        },
    )

    assert snapshot == {
        "net_income": 120.0,
        "gross_profit": 300.0,
        "revenue": 800.0,
        "shares_outstanding": 50.0,
        "total_assets": 1000.0,
        "long_term_debt": 200.0,
        "total_debt": 200.0,
        "current_assets": 400.0,
        "current_liabilities": 250.0,
    }


def test_pit_factor_snapshot_derives_pe_ratio_from_market_cap_and_net_income(monkeypatch):
    """Replay must compute pe_ratio PIT-safely so the canonicalizer recompute
    of value_factor_score has the same pe component as live.  Net income > 0
    and price * shares > 0 are the prerequisites; both come from the PIT
    snapshot and the signal-time price."""
    fake_snapshot = {
        "net_income": 200.0,
        "total_assets": 1000.0,
        "gross_profit": 350.0,
        "ebit": 250.0,
        "shares_outstanding": 100.0,
    }

    monkeypatch.setattr(
        pit_backfill,
        "_latest_as_of_cached",
        lambda ticker, run_date, *, lag_days: (fake_snapshot, "2026-03-31"),
    )
    monkeypatch.setattr(
        pit_backfill,
        "_prior_snapshot_cached",
        lambda ticker, before_report_date: (None, None),
    )

    fields, report_date = pit_backfill._pit_factor_snapshot(
        "AAPL", "2026-05-01", signal_price=50.0,
    )

    assert report_date == "2026-03-31"
    # market_cap = 50 * 100 = 5_000; pe = 5_000 / 200 = 25.0
    assert fields["pe_ratio"] == pytest.approx(25.0)


def test_pit_factor_snapshot_omits_pe_ratio_when_net_income_non_positive(monkeypatch):
    """Loss-making companies (net_income <= 0) must not write a pe_ratio.
    A negative pe is meaningless and would poison the value_factor_score
    component basis."""
    fake_snapshot = {
        "net_income": -50.0,
        "total_assets": 1000.0,
        "shares_outstanding": 100.0,
    }

    monkeypatch.setattr(
        pit_backfill,
        "_latest_as_of_cached",
        lambda ticker, run_date, *, lag_days: (fake_snapshot, "2026-03-31"),
    )
    monkeypatch.setattr(
        pit_backfill,
        "_prior_snapshot_cached",
        lambda ticker, before_report_date: (None, None),
    )

    fields, _ = pit_backfill._pit_factor_snapshot(
        "LOSSCO", "2026-05-01", signal_price=50.0,
    )

    assert "pe_ratio" not in fields
