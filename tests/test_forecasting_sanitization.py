import numpy as np
import pandas as pd
import pytest

from engine import forecasting


def test_forecast_drops_nan_close_rows_before_training(monkeypatch):
    dates = pd.date_range("2025-01-01", periods=260, freq="B")
    closes = np.linspace(100.0, 130.0, len(dates))
    closes[245] = np.nan
    df = pd.DataFrame({"Close": closes}, index=dates)

    monkeypatch.setattr(forecasting, "get_price_history", lambda ticker: df)
    monkeypatch.setattr(forecasting, "get_macro_data", lambda: {})
    monkeypatch.setattr(
        forecasting,
        "_load_store",
        lambda: {"predictions": [], "rolling_maes": {}},
    )
    monkeypatch.setattr(forecasting, "_save_store", lambda store: None)

    result = forecasting.forecast("NANUNIT", horizon_days=5)

    assert result.ticker == "NANUNIT"
    assert np.isfinite(result.predicted_price)
    assert result.predicted_price > 0


def test_clean_price_history_requires_finite_positive_close():
    dates = pd.date_range("2025-01-01", periods=5, freq="B")
    df = pd.DataFrame(
        {"Close": [100.0, np.nan, float("inf"), -1.0, 101.0]},
        index=dates,
    )

    cleaned = forecasting._clean_price_history(df, "BADROWS")

    assert cleaned["Close"].tolist() == pytest.approx([100.0, 101.0])
