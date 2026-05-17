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


def test_sec_adr_symbol_uses_fundamentals_only_overrides():
    assert pit_backfill._sec_adr_symbol_for_ticker("NOKIA.HE") == "NOK"
    assert pit_backfill._sec_adr_symbol_for_ticker("ASML.AS") == "ASML"
    assert pit_backfill._sec_adr_symbol_for_ticker("ERIC-B.ST") == "ERIC"
    assert pit_backfill._sec_adr_symbol_for_ticker("AAPL") is None


def test_adr_mapping_table_loads_maintainable_csv(tmp_path):
    mapping_path = tmp_path / "adr_mappings.csv"
    mapping_path.write_text(
        "local_symbol,adr_symbol,adr_exchange,sec_cik,notes\n"
        "XYZ.L,XYZ,NYSE,12345,test row\n",
        encoding="utf-8",
    )

    rows = pit_backfill._load_adr_mapping_table(mapping_path)

    assert rows["XYZ.L"]["adr_symbol"] == "XYZ"
    assert rows["XYZ.L"]["sec_cik"] == "12345"


def test_sec_companyfacts_snapshot_maps_ifrs_and_derives_gross_profit():
    payload = {
        "entityName": "Example PLC",
        "facts": {
            "ifrs-full": {
                "Revenue": {
                    "units": {
                        "EUR": [
                            {
                                "start": "2025-01-01",
                                "end": "2025-12-31",
                                "val": 1000,
                                "filed": "2026-03-05",
                                "form": "20-F",
                                "fp": "FY",
                            }
                        ]
                    }
                },
                "CostOfSales": {
                    "units": {
                        "EUR": [
                            {
                                "start": "2025-01-01",
                                "end": "2025-12-31",
                                "val": 420,
                                "filed": "2026-03-05",
                                "form": "20-F",
                                "fp": "FY",
                            }
                        ]
                    }
                },
                "Assets": {
                    "units": {
                        "EUR": [
                            {
                                "end": "2025-12-31",
                                "val": 2500,
                                "filed": "2026-03-05",
                                "form": "20-F",
                                "fp": "FY",
                            }
                        ]
                    }
                },
                "ProfitLoss": {
                    "units": {
                        "EUR": [
                            {
                                "start": "2025-01-01",
                                "end": "2025-12-31",
                                "val": 160,
                                "filed": "2026-03-05",
                                "form": "20-F",
                                "fp": "FY",
                            }
                        ]
                    }
                },
                "CashFlowsFromUsedInOperatingActivities": {
                    "units": {
                        "EUR": [
                            {
                                "start": "2025-01-01",
                                "end": "2025-12-31",
                                "val": 220,
                                "filed": "2026-03-06",
                                "form": "20-F",
                                "fp": "FY",
                            }
                        ]
                    }
                },
            }
        },
    }

    rows = pit_backfill._snapshots_from_sec_companyfacts(
        payload,
        adr_symbol="EXM",
        cik=123456,
        entity_name="Example PLC",
        limit=1,
    )

    assert len(rows) == 1
    period_date, snapshot, accepted_date = rows[0]
    assert period_date == "2025-12-31"
    assert accepted_date == "2026-03-06"
    assert snapshot["revenue"] == 1000.0
    assert snapshot["gross_profit"] == 580.0
    assert snapshot["_gross_profit_source"] == "revenue_minus_cost_of_revenue"
    assert snapshot["total_assets"] == 2500.0
    assert snapshot["operating_cashflow"] == 220.0
    assert snapshot["_sec_adr_symbol"] == "EXM"
    assert snapshot["_sec_cik"] == "123456"


def test_yahoo_timeseries_payload_prefers_trailing_quality_fields():
    payload = {
        "timeseries": {
            "result": [
                {
                    "meta": {"type": ["quarterlyGrossProfit"]},
                    "quarterlyGrossProfit": [
                        {"asOfDate": "2025-12-31", "reportedValue": {"raw": 100}}
                    ],
                },
                {
                    "meta": {"type": ["trailingGrossProfit"]},
                    "trailingGrossProfit": [
                        {"asOfDate": "2025-12-31", "reportedValue": {"raw": 450}}
                    ],
                },
                {
                    "meta": {"type": ["quarterlyTotalAssets"]},
                    "quarterlyTotalAssets": [
                        {"asOfDate": "2025-12-31", "reportedValue": {"raw": 2000}}
                    ],
                },
                {
                    "meta": {"type": ["trailingNetIncome"]},
                    "trailingNetIncome": [
                        {"asOfDate": "2025-12-31", "reportedValue": {"raw": 120}}
                    ],
                },
            ]
        }
    }

    rows = pit_backfill._snapshots_from_yahoo_timeseries_payload(payload, limit=1)

    assert rows == [
        (
            "2025-12-31",
            {
                "gross_profit": 450.0,
                "total_assets": 2000.0,
                "net_income": 120.0,
                "_yahoo_timeseries_types": "quarterlyGrossProfit,quarterlyTotalAssets,trailingGrossProfit,trailingNetIncome",
            },
        )
    ]


def test_yahoo_timeseries_carries_prior_instant_assets_forward():
    payload = {
        "timeseries": {
            "result": [
                {
                    "meta": {"type": ["quarterlyTotalAssets"]},
                    "quarterlyTotalAssets": [
                        {"asOfDate": "2025-06-30", "reportedValue": {"raw": 2000}}
                    ],
                },
                {
                    "meta": {"type": ["trailingGrossProfit"]},
                    "trailingGrossProfit": [
                        {"asOfDate": "2025-12-31", "reportedValue": {"raw": 450}}
                    ],
                },
            ]
        }
    }

    rows = pit_backfill._snapshots_from_yahoo_timeseries_payload(payload, limit=1)

    assert rows[0][0] == "2025-12-31"
    assert rows[0][1]["gross_profit"] == 450.0
    assert rows[0][1]["total_assets"] == 2000.0
    assert rows[0][1]["_yahoo_instant_fields_carried_forward_from"] == "total_assets:2025-06-30"


def test_alpha_vantage_payload_maps_adr_statement_fields():
    rows = pit_backfill._snapshots_from_alpha_vantage_payloads(
        {
            "quarterlyReports": [
                {
                    "fiscalDateEnding": "2025-06-30",
                    "grossProfit": "400",
                    "totalRevenue": "1000",
                    "netIncome": "120",
                    "ebit": "150",
                }
            ]
        },
        {
            "quarterlyReports": [
                {
                    "fiscalDateEnding": "2025-06-30",
                    "totalAssets": "2000",
                    "cashAndCashEquivalentsAtCarryingValue": "90",
                    "shortLongTermDebtTotal": "300",
                }
            ]
        },
        {
            "quarterlyReports": [
                {
                    "fiscalDateEnding": "2025-06-30",
                    "operatingCashflow": "180",
                    "capitalExpenditures": "-40",
                }
            ]
        },
        adr_symbol="BHP",
        limit=1,
    )

    assert rows == [
        (
            "2025-06-30",
            {
                "net_income": 120.0,
                "gross_profit": 400.0,
                "revenue": 1000.0,
                "ebit": 150.0,
                "total_assets": 2000.0,
                "cash": 90.0,
                "total_debt": 300.0,
                "operating_cashflow": 180.0,
                "capital_expenditure": -40.0,
                "_gross_profit_source": "direct_alpha_vantage",
                "_alpha_vantage_report_sections": "quarterlyReports",
                "_alpha_vantage_adr_symbol": "BHP",
            },
        )
    ]


def test_alpha_vantage_adr_supplements_existing_snapshot_without_extra_calls(tmp_path, monkeypatch):
    calls = []

    def fake_alpha(function, symbol, **kwargs):
        calls.append((function, symbol))
        assert symbol == "BHP"
        return {
            "quarterlyReports": [
                {
                    "fiscalDateEnding": "2025-06-30",
                    "grossProfit": "400",
                    "totalRevenue": "1000",
                    "netIncome": "120",
                }
            ]
        }

    writes = []
    monkeypatch.setattr(pit_backfill, "_alpha_get_json", fake_alpha)
    monkeypatch.setattr(pit_backfill, "_latest_snapshot_has_qmj_minimum", lambda ticker: False)
    monkeypatch.setattr(
        pit_backfill,
        "_existing_snapshot_for_report_date",
        lambda ticker, report_date: {
            "_source": "yahoo_timeseries",
            "_report_date": report_date,
            "total_assets": 2000.0,
        },
    )
    monkeypatch.setattr(
        pit_backfill,
        "record_snapshot",
        lambda ticker, report_date, snapshot, **kwargs: writes.append((ticker, report_date, snapshot, kwargs)),
    )

    results = pit_backfill.backfill_via_alpha_vantage_adr(
        ["BHP.AX"],
        limit=1,
        sleep_seconds=0,
        daily_call_budget=1,
        attempt_ledger_path=tmp_path / "alpha_ledger.json",
        api_key="test",
        fetch_cashflow=False,
    )

    assert results == {"BHP.AX": 1}
    assert calls == [("INCOME_STATEMENT", "BHP")]
    assert writes[0][0] == "BHP.AX"
    assert writes[0][2]["gross_profit"] == 400.0
    assert writes[0][2]["total_assets"] == 2000.0
    assert writes[0][2]["_alpha_vantage_base_source"] == "yahoo_timeseries"
    assert writes[0][3]["source"] == "alpha_vantage_adr"


def test_alpha_vantage_adr_skips_when_qmj_already_present(tmp_path, monkeypatch):
    def fail_alpha(*args, **kwargs):
        raise AssertionError("Alpha Vantage should not be called for already-populated QMJ")

    monkeypatch.setattr(pit_backfill, "_alpha_get_json", fail_alpha)
    monkeypatch.setattr(pit_backfill, "_latest_snapshot_has_qmj_minimum", lambda ticker: True)

    results = pit_backfill.backfill_via_alpha_vantage_adr(
        ["BHP.AX"],
        sleep_seconds=0,
        daily_call_budget=1,
        attempt_ledger_path=tmp_path / "alpha_ledger.json",
        api_key="test",
    )

    assert results == {"BHP.AX": 0}
