"""Golden-case tests for engine.action_smoother.

The smoother gates the EXIT side of the action map (KEEP / SELL / STRONG SELL)
through a Constantinides-Davis-Norman no-trade region plus a Wald-style
N-of-M persistence count.  CUSUM-confirmed urgent signals override.
"""

from __future__ import annotations

import config
import pytest

from engine.action_smoother import (
    DOWNSIDE_ACTIONS,
    EXIT_ACTIONS,
    REASON_BAND_HELD_KEEP,
    REASON_BAND_HELD_SELL,
    REASON_COLD_START,
    REASON_CUSUM_OVERRIDE,
    REASON_DISABLED,
    REASON_PASSTHROUGH_BUY,
    REASON_PERSISTENCE_SATISFIED,
    REASON_PERSISTENCE_SHORT,
    REASON_RECOVERED,
    smooth_exit_action,
)


# ---------------------------------------------------------------------------
# Defaults — match what the smoother reads from config when invoked live.
# ---------------------------------------------------------------------------

def _default_kwargs(**overrides):
    base = dict(
        score_vol=0.05,           # typical equity score sigma
        vix_pct=50.0,
        persistence_m_sell=0,
        persistence_m_strong=0,
        persistence_n=config.EXIT_SMOOTHER_PERSISTENCE_N,
        persistence_since=None,
        cusum_urgent=False,
        cusum_score=0.0,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1.  Master toggle off → pass through with reason=disabled.
# ---------------------------------------------------------------------------

def test_smoother_disabled_passes_through(monkeypatch):
    monkeypatch.setattr(config, "EXIT_SMOOTHER_ENABLED", False)
    out = smooth_exit_action(
        raw_score=-0.30,
        base_action="SELL",
        prev_smoothed_action="KEEP",
        **_default_kwargs(),
    )
    assert out.action == "SELL"
    assert out.reason == REASON_DISABLED


# ---------------------------------------------------------------------------
# 2.  BUY / STRONG BUY pass through unchanged (EXIT-side only design).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("action", ["STRONG BUY", "BUY"])
def test_buy_side_passthrough(action):
    out = smooth_exit_action(
        raw_score=0.45,
        base_action=action,
        prev_smoothed_action="KEEP",
        **_default_kwargs(),
    )
    assert out.action == action
    assert out.reason == REASON_PASSTHROUGH_BUY


# ---------------------------------------------------------------------------
# 3.  Cold-start ticker (no prior smoothed action) → falls back to base_action.
# ---------------------------------------------------------------------------

def test_cold_start_uses_legacy_action():
    out = smooth_exit_action(
        raw_score=-0.30,
        base_action="SELL",
        prev_smoothed_action=None,
        **_default_kwargs(),
    )
    assert out.action == "SELL"
    assert out.reason == REASON_COLD_START


# ---------------------------------------------------------------------------
# 4.  Score noise around the KEEP threshold → stays KEEP.
#     Score = -0.27, prev = KEEP, σ=0.05 → band ≈ 0.075, sell_entry ≈ -0.325.
# ---------------------------------------------------------------------------

def test_band_holds_keep_on_small_noise():
    out = smooth_exit_action(
        raw_score=-0.27,
        base_action="SELL",  # legacy threshold says SELL (score < -0.25)
        prev_smoothed_action="KEEP",
        **_default_kwargs(score_vol=0.05),
    )
    # Inside dead-zone — must hold KEEP.
    assert out.action == "KEEP"
    assert out.reason == REASON_BAND_HELD_KEEP
    assert out.band_low is not None and out.band_high is not None


# ---------------------------------------------------------------------------
# 5.  Score crosses sell_entry but persistence is short → stays KEEP.
# ---------------------------------------------------------------------------

def test_persistence_short_holds_keep():
    out = smooth_exit_action(
        raw_score=-0.40,                # well past the band
        base_action="SELL",
        prev_smoothed_action="KEEP",
        **_default_kwargs(
            persistence_m_sell=1,       # only 1 of 5 — too short
            persistence_m_strong=0,
        ),
    )
    assert out.action == "KEEP"
    assert out.reason == REASON_PERSISTENCE_SHORT


# ---------------------------------------------------------------------------
# 6.  Score crosses sell_entry AND persistence satisfied → emits SELL.
# ---------------------------------------------------------------------------

def test_persistence_satisfied_emits_sell():
    out = smooth_exit_action(
        raw_score=-0.40,
        base_action="SELL",
        prev_smoothed_action="KEEP",
        **_default_kwargs(
            persistence_m_sell=3,
            persistence_m_strong=0,
            persistence_since="2026-04-29",
        ),
    )
    assert out.action == "SELL"
    assert out.reason == REASON_PERSISTENCE_SATISFIED
    assert out.since_date == "2026-04-29"
    assert out.persistence_m == 3


# ---------------------------------------------------------------------------
# 7.  Strong persistence + score in STRONG SELL band → emits STRONG SELL.
# ---------------------------------------------------------------------------

def test_strong_sell_persistence_escalation():
    out = smooth_exit_action(
        raw_score=-0.70,
        base_action="STRONG SELL",
        prev_smoothed_action="KEEP",
        **_default_kwargs(
            persistence_m_sell=5,
            persistence_m_strong=4,
            persistence_since="2026-04-25",
        ),
    )
    assert out.action == "STRONG SELL"
    assert out.reason == REASON_PERSISTENCE_SATISFIED
    assert out.since_date == "2026-04-25"


# ---------------------------------------------------------------------------
# 8.  Single-day spike to STRONG SELL territory but no persistence → stays KEEP.
# ---------------------------------------------------------------------------

def test_single_day_strong_sell_spike_stays_keep():
    out = smooth_exit_action(
        raw_score=-0.65,
        base_action="STRONG SELL",
        prev_smoothed_action="KEEP",
        **_default_kwargs(
            persistence_m_sell=1,
            persistence_m_strong=1,
        ),
    )
    assert out.action == "KEEP"
    assert out.reason == REASON_PERSISTENCE_SHORT


# ---------------------------------------------------------------------------
# 9.  CUSUM urgent override forces STRONG SELL regardless of band/persistence.
# ---------------------------------------------------------------------------

def test_cusum_override_forces_strong_sell():
    out = smooth_exit_action(
        raw_score=-0.10,                 # *not* in SELL territory
        base_action="KEEP",
        prev_smoothed_action="KEEP",
        **_default_kwargs(
            cusum_urgent=True,
            cusum_score=0.85,            # well above override floor
        ),
    )
    assert out.action == "STRONG SELL"
    assert out.reason == REASON_CUSUM_OVERRIDE


def test_cusum_override_below_threshold_does_not_fire():
    out = smooth_exit_action(
        raw_score=-0.10,
        base_action="KEEP",
        prev_smoothed_action="KEEP",
        **_default_kwargs(
            cusum_urgent=True,
            cusum_score=0.40,            # below EXIT_SMOOTHER_CUSUM_OVERRIDE_MIN=0.6
        ),
    )
    assert out.action == "KEEP"
    assert out.reason != REASON_CUSUM_OVERRIDE


# ---------------------------------------------------------------------------
# 10.  Recovery from SELL back to KEEP — easier path (tighter δ_up).
# ---------------------------------------------------------------------------

def test_recovery_from_sell_to_keep():
    out = smooth_exit_action(
        raw_score=-0.20,                 # comfortably above KEEP threshold
        base_action="KEEP",
        prev_smoothed_action="SELL",
        **_default_kwargs(),
    )
    assert out.action == "KEEP"
    assert out.reason == REASON_RECOVERED


def test_sell_holds_when_score_inside_recovery_band():
    out = smooth_exit_action(
        raw_score=-0.26,                 # legacy says SELL by 0.01
        base_action="SELL",
        prev_smoothed_action="SELL",
        **_default_kwargs(),
    )
    assert out.action == "SELL"
    assert out.reason == REASON_BAND_HELD_SELL


# ---------------------------------------------------------------------------
# 11.  High VIX widens the band — 90th percentile → +45% wider.
# ---------------------------------------------------------------------------

def test_high_vix_widens_band():
    low_vix = smooth_exit_action(
        raw_score=-0.30,
        base_action="SELL",
        prev_smoothed_action="KEEP",
        **_default_kwargs(score_vol=0.05, vix_pct=10.0),
    )
    high_vix = smooth_exit_action(
        raw_score=-0.30,
        base_action="SELL",
        prev_smoothed_action="KEEP",
        **_default_kwargs(score_vol=0.05, vix_pct=90.0),
    )
    # high_vix band must be wider (band_low more negative, band_high more positive)
    assert high_vix.band_low is not None and low_vix.band_low is not None
    assert high_vix.band_low < low_vix.band_low      # entry threshold pushed deeper
    assert high_vix.band_high > low_vix.band_high    # recovery threshold pushed higher


# ---------------------------------------------------------------------------
# 12.  σ_score=None → falls back to EXIT_SMOOTHER_VOL_FALLBACK without crashing.
# ---------------------------------------------------------------------------

def test_none_score_vol_uses_fallback():
    out = smooth_exit_action(
        raw_score=-0.27,
        base_action="SELL",
        prev_smoothed_action="KEEP",
        **_default_kwargs(score_vol=None),
    )
    assert out.action == "KEEP"  # fallback band still wide enough to hold
    assert out.reason == REASON_BAND_HELD_KEEP


# ---------------------------------------------------------------------------
# 13.  Recovery from STRONG SELL back to SELL (intermediate step).
# ---------------------------------------------------------------------------

def test_recovery_from_strong_sell_to_sell():
    out = smooth_exit_action(
        raw_score=-0.42,                 # above sell_recovery, below keep_recovery
        base_action="SELL",
        prev_smoothed_action="STRONG SELL",
        **_default_kwargs(),
    )
    assert out.action == "SELL"
    assert out.reason == REASON_RECOVERED
