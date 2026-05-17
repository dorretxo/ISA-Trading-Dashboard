from __future__ import annotations

import json

from utils import pit_backfill


def test_refresh_queue_uses_fmp_for_plain_ticker_and_yf_for_suffix(tmp_path, monkeypatch):
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "items": [
                    {"ticker": "AAPL", "priority": 5},
                    {"ticker": "TCAP.L", "priority": 4},
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(pit_backfill, "backfill_ticker", lambda ticker, limit=None: 3)
    monkeypatch.setattr(
        pit_backfill,
        "backfill_via_yfinance_quarterly",
        lambda tickers, limit=None, sleep_seconds=0.0: {ticker: 2 for ticker in tickers},
    )

    stats = pit_backfill.refresh_queue_tickers(
        queue_path=queue,
        max_tickers=10,
        limit=4,
        sec_edgar_fallback=False,
        yahoo_timeseries_fallback=False,
        write_results=False,
    )

    assert stats["selected"] == 2
    assert stats["refreshed"] == 2
    assert stats["fmp_results"] == {"AAPL": 3}
    assert stats["yfinance_results"] == {"TCAP.L": 2}
    assert stats["snapshots_written"] == 5


def test_refresh_queue_uses_sec_before_yahoo_for_mapped_adr(tmp_path, monkeypatch):
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "items": [
                    {"ticker": "NOKIA.HE", "priority": 5},
                    {"ticker": "TCAP.L", "priority": 4},
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        pit_backfill,
        "_sec_adr_symbol_for_ticker",
        lambda ticker: "NOK" if ticker == "NOKIA.HE" else None,
    )
    monkeypatch.setattr(
        pit_backfill,
        "backfill_via_sec_edgar",
        lambda tickers, limit=None, sleep_seconds=0.0: {ticker: 2 for ticker in tickers},
    )
    monkeypatch.setattr(
        pit_backfill,
        "backfill_via_yahoo_timeseries",
        lambda tickers, limit=None, sleep_seconds=0.0: {ticker: 3 for ticker in tickers},
    )

    def fail_yfinance(*args, **kwargs):
        raise AssertionError("yfinance should not run when SEC/Yahoo already wrote snapshots")

    monkeypatch.setattr(pit_backfill, "backfill_via_yfinance_quarterly", fail_yfinance)

    stats = pit_backfill.refresh_queue_tickers(queue_path=queue, max_tickers=10, write_results=False)

    assert stats["sec_edgar_results"] == {"NOKIA.HE": 2}
    assert stats["yahoo_timeseries_results"] == {"TCAP.L": 3}
    assert stats["yfinance_results"] == {}
    assert stats["refreshed_snapshots"] == {"NOKIA.HE": 2, "TCAP.L": 3}
    assert stats["snapshots_written"] == 5


def test_refresh_queue_runs_alpha_adr_only_for_residual_qmj_gap(tmp_path, monkeypatch):
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({"items": [{"ticker": "BHP.AX", "priority": 5}]}), encoding="utf-8")

    monkeypatch.setattr(pit_backfill, "_adr_mapping_row_for_ticker", lambda ticker: {"adr_symbol": "BHP"})
    monkeypatch.setattr(pit_backfill, "backfill_via_sec_edgar", lambda tickers, limit=None, sleep_seconds=0.0: {ticker: 0 for ticker in tickers})
    monkeypatch.setattr(pit_backfill, "backfill_via_yahoo_timeseries", lambda tickers, limit=None, sleep_seconds=0.0: {ticker: 0 for ticker in tickers})
    monkeypatch.setattr(pit_backfill, "backfill_via_yfinance_quarterly", lambda tickers, limit=None, sleep_seconds=0.0: {ticker: 1 for ticker in tickers})
    monkeypatch.setattr(pit_backfill, "_latest_snapshot_has_qmj_minimum", lambda ticker: False)
    monkeypatch.setattr(
        pit_backfill,
        "backfill_via_alpha_vantage_adr",
        lambda tickers, limit=None, daily_call_budget=None: {ticker: 2 for ticker in tickers},
    )

    stats = pit_backfill.refresh_queue_tickers(
        queue_path=queue,
        max_tickers=10,
        alpha_vantage_adr_fallback=True,
        write_results=False,
    )

    assert stats["yfinance_results"] == {"BHP.AX": 1}
    assert stats["alpha_vantage_adr_results"] == {"BHP.AX": 2}
    assert stats["refreshed_snapshots"] == {"BHP.AX": 3}
    assert stats["snapshots_written"] == 3


def test_refresh_queue_falls_back_to_yfinance_when_fmp_writes_zero(tmp_path, monkeypatch):
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({"items": [{"ticker": "AAPL", "priority": 5}]}), encoding="utf-8")

    monkeypatch.setattr(pit_backfill, "backfill_ticker", lambda ticker, limit=None: 0)
    monkeypatch.setattr(
        pit_backfill,
        "backfill_via_yfinance_quarterly",
        lambda tickers, limit=None, sleep_seconds=0.0: {ticker: 1 for ticker in tickers},
    )

    stats = pit_backfill.refresh_queue_tickers(queue_path=queue, write_results=False)

    assert stats["fmp_results"] == {"AAPL": 0}
    assert stats["yfinance_results"] == {"AAPL": 1}
    assert stats["refreshed_snapshots"] == {"AAPL": 1}


def test_refresh_queue_fmp_only_skips_suffix_tickers(tmp_path, monkeypatch):
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "items": [
                    {"ticker": "TCAP.L", "priority": 6},
                    {"ticker": "AAPL", "priority": 5},
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(pit_backfill, "backfill_ticker", lambda ticker, limit=None: 4)

    def fail_yfinance(*args, **kwargs):
        raise AssertionError("yfinance fallback should stay disabled")

    monkeypatch.setattr(pit_backfill, "backfill_via_yfinance_quarterly", fail_yfinance)

    stats = pit_backfill.refresh_queue_tickers(
        queue_path=queue,
        max_tickers=10,
        yfinance_fallback=False,
        fmp_only=True,
        write_results=False,
    )

    assert stats["selected"] == 1
    assert stats["fmp_only"] is True
    assert stats["yfinance_fallback"] is False
    assert stats["fmp_results"] == {"AAPL": 4}
    assert stats["yfinance_results"] == {}
    assert stats["refreshed_snapshots"] == {"AAPL": 4}
