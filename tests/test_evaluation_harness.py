import warnings

import numpy as np
import pytest

from engine.evaluation_harness import _max_drawdown_from_returns_pct


def test_max_drawdown_uses_stable_log_equity_for_large_signal_sets():
    returns = np.full(100_000, 2.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        max_dd = _max_drawdown_from_returns_pct(returns)

    assert max_dd == pytest.approx(0.0)


def test_max_drawdown_handles_total_loss_without_invalid_warning():
    returns = np.array([10.0, -150.0, 25.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        max_dd = _max_drawdown_from_returns_pct(returns)

    assert -1.0 <= max_dd < -0.99
