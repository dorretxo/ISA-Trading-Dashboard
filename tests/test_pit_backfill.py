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
