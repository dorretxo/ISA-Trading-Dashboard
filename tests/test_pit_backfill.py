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


def test_pit_factor_snapshot_does_not_derive_pe_ratio(monkeypatch):
    """Regression pin: replay must NOT derive pe_ratio from a single
    PIT snapshot.  The snapshot's net_income is one quarter; live's
    pe_ratio (yfinance trailing P/E) is TTM.  Naive market_cap /
    net_income creates a systematic ~4x basis drift that pollutes
    value_factor_score parity (observed on PARR, BVS, REPX in the
    2026-05-12 parity report).  A correct PIT pe_ratio needs TTM
    aggregation across 4 quarters — separate change.  Until then
    replay leaves pe_ratio NULL so the parity comparator records
    'missing' rather than 'drift'."""
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

    fields, _ = pit_backfill._pit_factor_snapshot(
        "AAPL", "2026-05-01", signal_price=50.0,
    )

    assert "pe_ratio" not in fields
