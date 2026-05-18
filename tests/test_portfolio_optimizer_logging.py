import logging

import numpy as np

from engine import portfolio_optimizer


def test_nan_expected_return_warning_is_ascii(monkeypatch, caplog):
    monkeypatch.setattr(portfolio_optimizer, "_load_score_to_return_scale", lambda: 0.1)
    caplog.set_level(logging.WARNING, logger="engine.portfolio_optimizer")

    out = portfolio_optimizer._estimate_expected_returns_v2(
        [{"ticker": "AAA", "expected_return_90d": float("nan"), "aggregate_score": 0.0}],
        ["AAA"],
        confidence=False,
    )

    assert np.isfinite(out).all()
    assert "v2 mu: NaN" in caplog.text
    assert "\u03bc" not in caplog.text
