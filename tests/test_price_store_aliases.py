import pandas as pd

from utils import price_store


def test_price_store_resolves_alias_before_yahoo_download(monkeypatch, tmp_path):
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    close = pd.DataFrame({"GFRD.L": [100.0, 101.0, 102.0]}, index=dates)
    downloaded = pd.concat({"Close": close}, axis=1)
    calls = []

    def fake_download(tickers, *args, **kwargs):
        calls.append(tickers)
        return downloaded

    monkeypatch.setattr(price_store.yf, "download", fake_download)

    frames = price_store.download_price_history(
        ["GFRD"],
        start="2024-01-01",
        end="2024-01-03",
        cache_dir=tmp_path,
        force=True,
    )

    assert calls == [["GFRD.L"]]
    assert "GFRD" in frames
    assert frames["GFRD"]["Close"].iloc[-1] == 102.0
    assert (tmp_path / "GFRD.L.pkl").exists()
    assert not (tmp_path / "GFRD.pkl").exists()
