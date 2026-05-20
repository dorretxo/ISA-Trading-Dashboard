from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def _ohlc_frame(rows: int = 130) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=rows, freq="B")
    close = pd.Series(np.linspace(100.0, 125.0, rows), index=idx)
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(rows, 100_000),
        },
        index=idx,
    )


def test_portfolio_signal_prior_momentum_uses_shared_price_history(monkeypatch):
    from engine import discovery_backtest
    from utils import data_fetch

    monkeypatch.setattr(data_fetch, "get_price_history", lambda ticker: _ohlc_frame())
    monkeypatch.setattr(
        discovery_backtest.yf,
        "download",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("direct Yahoo download should not run")),
    )

    ret_10d, ret_30d, ret_90d, vol_20d = discovery_backtest._compute_prior_momentum("GFRD")

    assert ret_10d is not None
    assert ret_30d is not None
    assert ret_90d is not None
    assert vol_20d is not None


def test_paper_trading_next_open_resolves_alias_before_yahoo(monkeypatch):
    from engine import paper_trading

    seen = []

    def fake_download(ticker, *args, **kwargs):
        seen.append(ticker)
        idx = pd.DatetimeIndex([pd.Timestamp.today()])
        return pd.concat({"Open": pd.DataFrame({"GFRD.L": [520.0]}, index=idx)}, axis=1)

    monkeypatch.setattr(paper_trading.yf, "download", fake_download)

    price = paper_trading._get_next_open("GFRD", date(2026, 5, 19).isoformat())

    assert price == 520.0
    assert seen == ["GFRD.L"]


def test_exit_engine_uses_shared_price_history(monkeypatch):
    from engine import exit_engine

    seen = []

    def fake_history(ticker):
        seen.append(ticker)
        return _ohlc_frame(260)

    monkeypatch.setattr(exit_engine, "get_price_history", fake_history)
    monkeypatch.setattr(
        exit_engine.yf,
        "download",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("direct Yahoo download should not run")),
    )

    exit_engine.assess_exits(
        [
            {
                "ticker": "GFRD",
                "name": "Galliford Try",
                "current_price": 520.0,
                "aggregate_score": 0.1,
                "action": "KEEP",
                "atr": 10.0,
            }
        ],
        [{"ticker": "GFRD", "quantity": 10, "currency": "GBX"}],
    )

    assert seen == ["GFRD"]
