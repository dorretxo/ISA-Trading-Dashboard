import pandas as pd

from engine.labeling import triple_barrier_from_frame


def test_triple_barrier_labels_target_first():
    frame = pd.DataFrame(
        {
            "Open": [100, 101, 104],
            "High": [101, 105, 110],
            "Low": [99, 100, 103],
            "Close": [100, 104, 108],
            "Volume": [1000, 1000, 1000],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )

    result = triple_barrier_from_frame(
        frame,
        "2024-01-01",
        100.0,
        stop_loss=95.0,
        take_profit=104.0,
        horizon_days=5,
    )

    assert result is not None
    assert result.label == 1
    assert result.hit == "target"
    assert result.days == 1


def test_triple_barrier_uses_stop_first_when_both_hit_same_bar():
    frame = pd.DataFrame(
        {
            "Open": [100, 100],
            "High": [101, 106],
            "Low": [99, 94],
            "Close": [100, 100],
            "Volume": [1000, 1000],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    result = triple_barrier_from_frame(
        frame,
        "2024-01-01",
        100.0,
        stop_loss=95.0,
        take_profit=105.0,
        horizon_days=5,
    )

    assert result is not None
    assert result.label == -1
    assert result.hit == "stop"

