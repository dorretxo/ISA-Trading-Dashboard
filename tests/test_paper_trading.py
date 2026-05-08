from datetime import datetime as RealDateTime

import pandas as pd

from engine import paper_trading


def test_get_next_open_skips_future_session(monkeypatch):
    class FixedDateTime(RealDateTime):
        @classmethod
        def now(cls):
            return cls(2026, 5, 3, 10, 0, 0)

    def fail_download(*args, **kwargs):
        raise AssertionError("download should not be called for future fills")

    monkeypatch.setattr(paper_trading, "datetime", FixedDateTime)
    monkeypatch.setattr(paper_trading.yf, "download", fail_download)

    assert paper_trading._get_next_open("CHG.L", "2026-05-02T16:00:00") is None


def test_get_next_open_clips_download_end_to_available_date(monkeypatch):
    class FixedDateTime(RealDateTime):
        @classmethod
        def now(cls):
            return cls(2026, 5, 5, 10, 0, 0)

    calls = []

    def fake_download(ticker, start, end, progress, auto_adjust):
        calls.append({"ticker": ticker, "start": start, "end": end})
        return pd.DataFrame({"Open": [123.45]}, index=pd.to_datetime(["2026-05-04"]))

    monkeypatch.setattr(paper_trading, "datetime", FixedDateTime)
    monkeypatch.setattr(paper_trading.yf, "download", fake_download)

    assert paper_trading._get_next_open("CHG.L", "2026-05-01T16:00:00") == 123.45
    assert calls == [{"ticker": "CHG.L", "start": "2026-05-02", "end": "2026-05-06"}]
