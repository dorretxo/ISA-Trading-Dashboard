"""Tests for engine/momentum_gates.py — Tier-4 momentum sanity gates."""

from __future__ import annotations

import pytest

from engine.momentum_gates import (
    evaluate_momentum_gates,
    suggest_limit_price,
)


def test_momentum_gates_clean_pass():
    res = evaluate_momentum_gates(
        rsi=55, stretch_200dma=0.10, entry_stance="Ready", realized_vol_pctile=0.50,
    )
    assert res.action_ceiling == "STRONG BUY"
    assert not res.needs_pullback


def test_momentum_gates_trustpilot_profile_blocks_strong_buy():
    """TRST.L: RSI 70 (just under cap), stretch +35%, entry='Pullback Preferred'."""
    res = evaluate_momentum_gates(
        rsi=70, stretch_200dma=0.35, entry_stance="Pullback Preferred",
    )
    # Entry-stance block is the binding constraint; all paths trigger pullback.
    assert res.action_ceiling == "BUY"
    assert res.needs_pullback
    assert any("Entry stance" in r for r in res.reasons)


def test_momentum_gates_rsi_70_passes_strong_buy_cap_just():
    """RSI 70 sits below the configured 75 cap — does not block."""
    res = evaluate_momentum_gates(rsi=70, stretch_200dma=0.10, entry_stance="Ready")
    assert res.action_ceiling == "STRONG BUY"
    assert res.flags["rsi"] == "pass"


def test_momentum_gates_rsi_above_75_caps_at_buy():
    res = evaluate_momentum_gates(rsi=78, stretch_200dma=0.10, entry_stance="Ready")
    assert res.action_ceiling == "BUY"
    assert res.needs_pullback
    assert any("RSI" in r for r in res.reasons)


def test_momentum_gates_rsi_above_80_caps_at_neutral():
    res = evaluate_momentum_gates(rsi=82, stretch_200dma=0.10, entry_stance="Ready")
    assert res.action_ceiling == "NEUTRAL"


def test_momentum_gates_stretch_above_35pct_caps_at_buy():
    res = evaluate_momentum_gates(rsi=60, stretch_200dma=0.40, entry_stance="Ready")
    assert res.action_ceiling == "BUY"
    assert res.needs_pullback


def test_momentum_gates_high_vol_caps_at_buy():
    res = evaluate_momentum_gates(
        rsi=60, stretch_200dma=0.10, entry_stance="Ready", realized_vol_pctile=0.95,
    )
    assert res.action_ceiling == "BUY"


def test_momentum_gates_skip_on_missing_data():
    res = evaluate_momentum_gates(
        rsi=None, stretch_200dma=None, entry_stance="Ready", realized_vol_pctile=None,
    )
    assert res.action_ceiling == "STRONG BUY"


# ---------------------------------------------------------------------------
# Limit-price advisor
# ---------------------------------------------------------------------------

def test_suggest_limit_picks_highest_below_current():
    """Of all reference points below current, pick the closest pullback target."""
    s = suggest_limit_price(
        current_price=251.0, sma_200=185.0, support_level=214.0, atr=8.0,
        placement_price=214.0,
    )
    assert s is not None
    # 200-DMA*1.05 = 194.25; support*1.02 = 218.28; placement = 214; cp-1.5atr=239
    # Highest below 251 is 239 (current - 1.5×ATR).
    assert s.limit_price == pytest.approx(239.0, abs=1e-4)
    assert "ATR" in s.method


def test_suggest_limit_falls_back_to_sma_when_no_other_refs():
    s = suggest_limit_price(current_price=100.0, sma_200=80.0)
    assert s is not None
    assert s.limit_price == pytest.approx(80.0 * 1.05, abs=1e-4)


def test_suggest_limit_returns_none_when_nothing_below_current():
    """All references above current price => no pullback target."""
    s = suggest_limit_price(current_price=10.0, sma_200=15.0, support_level=20.0)
    assert s is None


def test_suggest_limit_handles_missing_inputs():
    assert suggest_limit_price(current_price=None, sma_200=80.0) is None
    s = suggest_limit_price(current_price=100.0, sma_200=None, support_level=None,
                            atr=None, placement_price=None)
    assert s is None
