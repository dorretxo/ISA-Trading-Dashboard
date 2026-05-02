"""Tests for engine/threshold_learner.py — Thompson sampling, EG sleeves, persistence."""

from __future__ import annotations

import json
import random

import pytest

from engine.threshold_learner import (
    BetaPosterior,
    DEFAULT_SLEEVE_WEIGHTS,
    OutcomeBatch,
    PROFILES,
    ThresholdLearnerState,
    apply_overlay,
    eg_update_sleeve_weights,
    load_state,
    save_state,
    select_profile,
    update_posteriors,
    update_sleeve_weights_state,
)


# ---------------------------------------------------------------------------
# BetaPosterior
# ---------------------------------------------------------------------------

def test_beta_posterior_initial_mean_is_uniform():
    p = BetaPosterior()
    assert p.mean() == pytest.approx(0.5)


def test_beta_posterior_update_shifts_mean_toward_observed_rate():
    p = BetaPosterior()
    p.update(successes=20, failures=5)
    assert 0.7 < p.mean() < 0.9


def test_beta_posterior_sample_in_unit_interval():
    rng = random.Random(42)
    p = BetaPosterior(alpha=10, beta=5)
    samples = [p.sample(rng) for _ in range(100)]
    assert all(0.0 <= s <= 1.0 for s in samples)
    # Mean of samples should be near alpha/(alpha+beta) = 10/15 ≈ 0.667
    avg = sum(samples) / len(samples)
    assert 0.55 < avg < 0.78


# ---------------------------------------------------------------------------
# Profile overlay
# ---------------------------------------------------------------------------

def test_profile_overlay_replaces_only_specified_keys():
    class _Cfg:
        PERCENTILE_STRONG_BUY_PCT = 0.05
        ALTMAN_Z_STRONG_BUY_MIN = 2.6
        UNTOUCHED_KEY = "stay"

    o = apply_overlay(_Cfg, "conservative")
    assert o.PERCENTILE_STRONG_BUY_PCT == 0.03    # conservative override
    assert o.UNTOUCHED_KEY == "stay"               # falls through to base


def test_profile_moderate_uses_base_config_unchanged():
    class _Cfg:
        PERCENTILE_STRONG_BUY_PCT = 0.05

    o = apply_overlay(_Cfg, "moderate")
    assert o.PERCENTILE_STRONG_BUY_PCT == 0.05


# ---------------------------------------------------------------------------
# Profile selection — Thompson + drift override
# ---------------------------------------------------------------------------

def test_select_profile_drift_forces_conservative():
    class _Cfg:
        DRIFT_FORCE_CONSERVATIVE = True

    state = ThresholdLearnerState()
    # Aggressive has the highest mean — Thompson would pick it
    state.posteriors["aggressive"].update(successes=100, failures=10)
    profile = select_profile(state, drift_alert=True, config_module=_Cfg)
    assert profile == "conservative"


def test_select_profile_picks_higher_posterior_in_expectation():
    """Thompson is stochastic but the better profile wins in expectation."""
    class _Cfg:
        DRIFT_FORCE_CONSERVATIVE = True

    state = ThresholdLearnerState()
    state.posteriors["aggressive"].update(successes=80, failures=20)    # 80% rate
    state.posteriors["conservative"].update(successes=10, failures=90)    # 10% rate

    rng = random.Random(0)
    counts = {"aggressive": 0, "moderate": 0, "conservative": 0}
    for _ in range(200):
        counts[select_profile(state, drift_alert=False, rng=rng, config_module=_Cfg)] += 1
    # Aggressive should dominate
    assert counts["aggressive"] > counts["conservative"]
    assert counts["aggressive"] > counts["moderate"]


# ---------------------------------------------------------------------------
# EG sleeve-weight updates
# ---------------------------------------------------------------------------

def test_eg_update_increases_weight_of_positive_ic_sleeve():
    weights = dict(DEFAULT_SLEEVE_WEIGHTS)
    ic = {"quality": 0.10, "momentum": -0.05, "value": 0.0,
          "low_risk": 0.0, "ready": 0.0, "pead": 0.0}
    new_weights = eg_update_sleeve_weights(weights, ic, learning_rate=2.0)

    assert new_weights["quality"] > weights["quality"]
    assert new_weights["momentum"] < weights["momentum"]
    assert sum(new_weights.values()) == pytest.approx(1.0, abs=0.001)


def test_eg_update_handles_missing_ic_keys():
    weights = dict(DEFAULT_SLEEVE_WEIGHTS)
    new_weights = eg_update_sleeve_weights(weights, {}, learning_rate=0.05)
    # No IC info => weights basically unchanged after renormalisation
    for k in weights:
        assert abs(new_weights[k] - weights[k]) < 0.005


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_state_round_trip(tmp_path, monkeypatch):
    class _Cfg:
        THRESHOLD_LEARNER_STATE_FILE = str(tmp_path / "state.json")

    state = ThresholdLearnerState()
    state.posteriors["aggressive"].update(successes=12, failures=3)
    state.last_active_profile = "aggressive"
    state.n_updates = 4
    state.sleeve_weights["quality"] = 0.30

    save_state(state, config_module=_Cfg)
    loaded = load_state(config_module=_Cfg)

    assert loaded.last_active_profile == "aggressive"
    assert loaded.posteriors["aggressive"].alpha == pytest.approx(13.0)
    assert loaded.posteriors["aggressive"].beta == pytest.approx(4.0)
    assert loaded.sleeve_weights["quality"] == pytest.approx(0.30)
    assert loaded.n_updates == 4


def test_load_state_returns_default_on_missing_file(tmp_path):
    class _Cfg:
        THRESHOLD_LEARNER_STATE_FILE = str(tmp_path / "missing.json")
    s = load_state(config_module=_Cfg)
    assert s.last_active_profile == "moderate"
    assert s.posteriors["moderate"].alpha == 1.0


# ---------------------------------------------------------------------------
# Outcome aggregation
# ---------------------------------------------------------------------------

def test_update_posteriors_accumulates_across_outcomes():
    state = ThresholdLearnerState()
    outcomes = [
        OutcomeBatch(profile="moderate", successes=10, failures=2),
        OutcomeBatch(profile="moderate", successes=5, failures=3),
        OutcomeBatch(profile="aggressive", successes=2, failures=8),
    ]
    update_posteriors(state, outcomes)
    assert state.posteriors["moderate"].alpha == pytest.approx(16.0)    # 1 + 10 + 5
    assert state.posteriors["moderate"].beta == pytest.approx(6.0)       # 1 + 2 + 3
    assert state.n_updates == 1


def test_update_sleeve_weights_state_uses_default_lr():
    class _Cfg:
        SLEEVE_WEIGHTS_LEARNING_RATE = 1.0

    state = ThresholdLearnerState()
    update_sleeve_weights_state(state, {"quality": 0.10}, config_module=_Cfg)
    # Quality weight should rise relative to baseline
    assert state.sleeve_weights["quality"] > DEFAULT_SLEEVE_WEIGHTS["quality"]
