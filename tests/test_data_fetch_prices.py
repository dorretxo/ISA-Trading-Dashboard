import pandas as pd

from utils import data_fetch


def test_get_current_price_handles_duplicate_close_columns(monkeypatch):
    df = pd.DataFrame(
        [[10.0, None], [11.0, 99.0]],
        columns=["Close", "Close"],
    )
    monkeypatch.setitem(data_fetch._price_cache, "DUP", df)

    assert data_fetch.get_current_price("DUP") == 11.0


def test_get_daily_change_handles_duplicate_close_columns(monkeypatch):
    df = pd.DataFrame(
        [[10.0, None], [11.0, 99.0]],
        columns=["Close", "Close"],
    )
    monkeypatch.setitem(data_fetch._price_cache, "DUPCHG", df)

    assert data_fetch.get_daily_change("DUPCHG") == 10.0


def test_normalise_price_frame_selects_requested_ticker_from_multiindex():
    columns = pd.MultiIndex.from_product(
        [["Close", "Open"],
         ["AAA", "BBB"]],
        names=["Price", "Ticker"],
    )
    df = pd.DataFrame(
        [[10.0, 20.0, 9.0, 19.0], [11.0, 21.0, 10.0, 20.0]],
        columns=columns,
    )

    normalised = data_fetch._normalise_price_frame(df, "BBB")

    assert list(normalised.columns) == ["Close", "Open"]
    assert normalised["Close"].iloc[-1] == 21.0
