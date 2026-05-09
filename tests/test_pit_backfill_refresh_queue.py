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
        write_results=False,
    )

    assert stats["selected"] == 2
    assert stats["refreshed"] == 2
    assert stats["fmp_results"] == {"AAPL": 3}
    assert stats["yfinance_results"] == {"TCAP.L": 2}
    assert stats["snapshots_written"] == 5


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
