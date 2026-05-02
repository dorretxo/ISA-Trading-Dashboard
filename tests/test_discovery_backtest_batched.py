import pandas as pd

from engine.discovery_backtest import (
    _check_stop_target_hits_from_frame,
    _extract_ticker_frame,
    _first_bar_value,
)


def test_batched_price_helpers_slice_multi_ticker_frames():
    dates = pd.to_datetime(["2026-04-06", "2026-04-07", "2026-04-08"])
    raw = pd.DataFrame(
        {
            ("AAA", "Open"): [10.0, 10.5, 11.0],
            ("AAA", "Close"): [10.4, 10.9, 11.4],
            ("AAA", "Low"): [9.8, 10.2, 10.8],
            ("AAA", "High"): [10.6, 11.2, 11.6],
            ("BBB", "Open"): [20.0, 20.5, 21.0],
            ("BBB", "Close"): [20.4, 20.9, 21.4],
            ("BBB", "Low"): [19.8, 20.2, 20.8],
            ("BBB", "High"): [20.6, 21.2, 21.6],
        },
        index=dates,
    )

    frame = _extract_ticker_frame(raw, "AAA")

    assert frame is not None
    assert _first_bar_value(frame, "2026-04-04T12:00:00", 2) == 10.4
    assert _first_bar_value(frame, "2026-04-04T12:00:00", 2, column="Open") == 10.0


def test_stop_target_scan_uses_downloaded_ohlc_window():
    frame = pd.DataFrame(
        {
            "Open": [10.0, 10.2, 10.3],
            "Close": [10.1, 10.4, 10.5],
            "Low": [9.8, 9.7, 10.2],
            "High": [10.3, 10.8, 11.2],
        },
        index=pd.to_datetime(["2026-04-07", "2026-04-08", "2026-04-09"]),
    )

    stop_hit, stop_day, target_hit, target_day = _check_stop_target_hits_from_frame(
        frame,
        "2026-04-06T09:30:00",
        stop_loss=9.75,
        take_profit=11.0,
    )

    assert stop_hit is True
    assert stop_day == 2
    assert target_hit is True
    assert target_day == 3
