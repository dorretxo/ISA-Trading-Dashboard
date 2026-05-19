import pandas as pd


def test_get_ticker_info_cache_only_never_calls_yahoo(monkeypatch):
    from utils import data_fetch

    data_fetch._info_cache.clear()
    data_fetch.reset_ticker_info_stats()

    def boom(_ticker):
        raise AssertionError("Yahoo network should not be called in cache-only mode")

    monkeypatch.setattr(data_fetch.yf, "Ticker", boom)

    assert data_fetch.get_ticker_info("ABC", allow_network=False) == {}
    assert data_fetch.get_ticker_info_stats()["skipped_cache_only"] == 1

    data_fetch.set_cached_ticker_info("ABC", {"sector": "Industrials"})
    assert data_fetch.get_ticker_info("abc", allow_network=False) == {"sector": "Industrials"}
    assert data_fetch.get_ticker_info_stats()["cache"] == 1


def test_fundamental_uses_supplied_info_when_yahoo_network_disabled(monkeypatch):
    from engine import fundamental

    def boom(*_args, **_kwargs):
        raise AssertionError("fundamental.analyse should use supplied info")

    def insider_boom(*_args, **_kwargs):
        raise AssertionError("fundamental.analyse should not fetch Yahoo insider metadata")

    monkeypatch.setattr(fundamental, "get_ticker_info", boom)
    monkeypatch.setattr(fundamental, "_get_fmp_fundamentals", lambda _ticker: None)
    monkeypatch.setattr(fundamental, "get_insider_transactions", insider_boom)

    result = fundamental.analyse(
        "ABC",
        info={
            "trailingPE": 12.0,
            "marketCap": 5_000_000_000,
            "sector": "Industrials",
            "returnOnEquity": 0.18,
            "totalDebt": 0,
            "totalCash": 100_000_000,
        },
        allow_yahoo_network=False,
    )

    assert result["pe_ratio"] == 12.0
    assert result["market_cap"] == 5_000_000_000
    assert result["sector"] == "Industrials"


def test_risk_overlay_uses_fmp_earnings_when_yahoo_disabled(monkeypatch):
    from engine import risk_overlay

    risk_overlay._post_earnings_cache.clear()
    prices = pd.DataFrame({"Close": [100.0] * 80})
    monkeypatch.setattr(risk_overlay, "get_price_history", lambda _ticker: prices)

    overlay = risk_overlay.apply_risk_overlay(
        {
            "_earnings_surprises": [
                {
                    "date": pd.Timestamp.today().date().isoformat(),
                    "actualEarningResult": 0.8,
                    "estimatedEarning": 1.0,
                }
            ],
            "market_cap": 5_000_000_000,
        },
        "ABC",
        allow_yahoo_earnings=False,
    )

    assert overlay.post_earnings_recent is True
    assert overlay.earnings_miss is True
    assert overlay.earnings_miss_pct == -20.0


def test_risk_overlay_resolves_yahoo_alias_for_earnings(monkeypatch):
    from engine import risk_overlay

    risk_overlay._post_earnings_cache.clear()
    prices = pd.DataFrame({"Close": [100.0] * 80})
    seen = []
    today = pd.Timestamp.today().normalize()
    earnings = pd.DataFrame(
        {"Reported EPS": [1.1], "EPS Estimate": [1.0]},
        index=pd.DatetimeIndex([today]),
    )

    class FakeTicker:
        def __init__(self, ticker):
            seen.append(ticker)

        @property
        def calendar(self):
            return None

        @property
        def earnings_dates(self):
            return earnings

    monkeypatch.setattr(risk_overlay, "get_price_history", lambda _ticker: prices)
    monkeypatch.setattr("yfinance.Ticker", FakeTicker)

    overlay = risk_overlay.apply_risk_overlay(
        {"market_cap": 1_000_000_000},
        "GFRD",
        allow_yahoo_earnings=True,
    )

    assert seen == ["GFRD.L"]
    assert overlay.post_earnings_recent is True
