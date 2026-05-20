import io
import zipfile

from utils import non_us_fundamental_coverage as coverage


def test_coverage_audit_routes_missing_non_us_qmj():
    candidates = [
        {
            "ticker": "GRG.L",
            "sector": "Consumer Cyclical",
            "pit_source": "yfinance",
            "qmj_factor_score": None,
            "qmj_component_count": 1,
            "action_gate_reasons": ["Thin yfinance quality evidence (qmj_components=1, source=yfinance)"],
            "sb_score": 1.2,
        },
        {
            "ticker": "BHP.AX",
            "sector": "Materials",
            "pit_source": "yfinance",
            "qmj_factor_score": None,
        },
        {
            "ticker": "3IN.L",
            "name": "3I Infrastructure PLC",
            "sector": "",
            "pit_source": "yfinance",
            "qmj_factor_score": None,
        },
        {
            "ticker": "SAP.DE",
            "sector": "Technology",
            "pit_source": "yfinance",
            "qmj_factor_score": 0.5,
        },
        {"ticker": "AAPL", "qmj_factor_score": 0.9},
    ]

    audit = coverage.build_coverage_audit(
        candidates,
        adr_mapping={"BHP.AX": {"adr_symbol": "BHP", "adr_exchange": "NYSE", "sec_cik": ""}},
    )

    assert audit["summary"]["total_candidates"] == 5
    assert audit["summary"]["non_us_candidates"] == 4
    assert audit["summary"]["non_us_qmj_null"] == 3
    assert audit["route_counts"]["openfigi_to_esef_ixbrl_then_companies_house"] == 1
    assert audit["route_counts"]["sec_edgar_then_alpha_vantage_adr"] == 1
    assert audit["route_counts"]["sector_specific_quality_contract"] == 1
    assert audit["route_counts"]["covered"] == 1


def test_openfigi_mapping_keeps_unsupported_suffix_from_misaligning(monkeypatch):
    class Response:
        status_code = 200

        def json(self):
            return [
                {
                    "data": [
                        {
                            "figi": "BBG000BDKLH1",
                            "name": "GREGGS PLC",
                            "ticker": "GRG",
                            "exchCode": "LN",
                            "securityType": "Common Stock",
                        }
                    ]
                }
            ]

    def fake_request(method, url, **kwargs):
        assert method == "POST"
        assert kwargs["json"] == [{"idType": "TICKER", "idValue": "GRG", "exchCode": "LN"}]
        return Response()

    monkeypatch.setattr(coverage.requests, "request", fake_request)

    rows = coverage.map_openfigi(
        [{"ticker": "GRG.L", "suffix": ".L"}, {"ticker": "BAD.XY", "suffix": ".XY"}],
        batch_size=10,
    )

    by_ticker = {row["ticker"]: row for row in rows}
    assert by_ticker["GRG.L"]["status"] == "mapped"
    assert by_ticker["GRG.L"]["name"] == "GREGGS PLC"
    assert by_ticker["BAD.XY"]["status"] == "unsupported_suffix"


def test_filing_probe_skips_fund_contract_after_openfigi(monkeypatch):
    audit = {
        "items": [
            {
                "ticker": "MYI.L",
                "suffix": ".L",
                "priority": 10,
                "qmj_missing": True,
                "instrument_class": "operating_company",
                "recommended_route": "openfigi_to_esef_ixbrl_then_companies_house",
            }
        ]
    }

    monkeypatch.setattr(
        coverage,
        "map_openfigi",
        lambda items, sleep_seconds=0.0: [
            {
                "ticker": "MYI.L",
                "status": "mapped",
                "name": "MURRAY INTERNATIONAL TR-O",
                "security_type": "Closed-End Fund",
            }
        ],
    )

    def fail_if_called(_name):
        raise AssertionError("fund-like securities should not be probed as standard QMJ")

    monkeypatch.setattr(coverage, "probe_filings_xbrl_entity", fail_if_called)

    probe = coverage.probe_public_filing_routes(audit, limit=1, sleep_seconds=0.0)

    row = probe["items"][0]
    assert row["refined_instrument_class"] == "fund_or_investment_trust"
    assert row["refined_qmj_contract_family"] == "sector_specific_quality"
    assert row["filings_xbrl"]["status"] == "skipped_sector_specific_quality_contract"


def test_extract_esef_ixbrl_snapshot_reads_standard_ifrs_fields():
    xhtml = """<?xml version="1.0" encoding="UTF-8"?>
    <html xmlns="http://www.w3.org/1999/xhtml"
          xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"
          xmlns:xbrli="http://www.xbrl.org/2003/instance">
      <head>
        <ix:resources>
          <xbrli:context id="dur">
            <xbrli:entity><xbrli:identifier scheme="lei">TEST</xbrli:identifier></xbrli:entity>
            <xbrli:period><xbrli:startDate>2025-01-01</xbrli:startDate><xbrli:endDate>2025-12-31</xbrli:endDate></xbrli:period>
          </xbrli:context>
          <xbrli:context id="inst">
            <xbrli:entity><xbrli:identifier scheme="lei">TEST</xbrli:identifier></xbrli:entity>
            <xbrli:period><xbrli:instant>2025-12-31</xbrli:instant></xbrli:period>
          </xbrli:context>
        </ix:resources>
      </head>
      <body>
        <ix:nonFraction name="ifrs-full:Revenue" contextRef="dur" scale="6">1,250.5</ix:nonFraction>
        <ix:nonFraction name="ifrs-full:GrossProfit" contextRef="dur" scale="6">700.0</ix:nonFraction>
        <ix:nonFraction name="ifrs-full:ProfitLoss" contextRef="dur" scale="6">(12.0)</ix:nonFraction>
        <ix:nonFraction name="ifrs-full:Assets" contextRef="inst" scale="6">2,000.0</ix:nonFraction>
      </body>
    </html>
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("TEST/reports/report.xhtml", xhtml)

    snapshot = coverage.extract_esef_ixbrl_snapshot(buf.getvalue(), period_end="2025-12-31")

    assert snapshot["status"] == "extracted"
    assert snapshot["fields"]["revenue"] == 1_250_500_000
    assert snapshot["fields"]["gross_profit"] == 700_000_000
    assert snapshot["fields"]["net_income"] == -12_000_000
    assert snapshot["fields"]["total_assets"] == 2_000_000_000
    assert snapshot["has_qmj_minimum"] is True
