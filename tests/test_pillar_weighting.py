from __future__ import annotations

import json
from datetime import datetime

import pytest

import config
from engine import bayesian_learning
from engine import discovery_backtest
from engine.backtest import _ics_to_weights
from engine.pillar_weighting import (
    apply_weight_guardrails,
    blend_with_default_prior,
    pillar_parity_gate,
    positive_ic_allocation,
)


def test_positive_ic_allocation_ignores_negative_ic():
    weights = positive_ic_allocation(
        {
            "technical": -0.10,
            "fundamental": 0.04,
            "sentiment": 0.02,
            "forecast": -0.03,
        }
    )

    assert weights is not None
    assert weights["technical"] == 0.0
    assert weights["forecast"] == 0.0
    assert weights["fundamental"] > weights["sentiment"]


def test_guardrails_cap_long_horizon_sentiment_and_negative_forecast(monkeypatch):
    monkeypatch.setattr(config, "PILLAR_WEIGHT_MIN_FLOOR", 0.03)
    monkeypatch.setattr(config, "PILLAR_WEIGHT_MAX_SINGLE", 0.55)
    monkeypatch.setattr(config, "PILLAR_WEIGHT_SENTIMENT_LONG_MAX", 0.08)
    monkeypatch.setattr(config, "PILLAR_WEIGHT_FORECAST_MAX_WITH_NEGATIVE_IC", 0.03)

    weights = apply_weight_guardrails(
        {"technical": 0.05, "fundamental": 0.10, "sentiment": 0.55, "forecast": 0.30},
        horizon="90d",
        ic_by_pillar={"technical": 0.02, "fundamental": 0.04, "sentiment": 0.03, "forecast": -0.05},
    )

    assert sum(weights.values()) == pytest.approx(1.0, abs=0.0002)
    assert max(weights.values()) <= 0.55
    assert weights["sentiment"] <= 0.08
    assert weights["forecast"] <= 0.03


def test_adaptive_blend_tilts_toward_default_prior(monkeypatch):
    monkeypatch.setattr(
        config,
        "WEIGHTS",
        {"technical": 0.30, "fundamental": 0.40, "sentiment": 0.08, "forecast": 0.22},
    )
    monkeypatch.setattr(config, "ADAPTIVE_WEIGHTS_LIVE_BLEND", 0.60)

    weights = blend_with_default_prior(
        {"technical": 0.0, "fundamental": 1.0, "sentiment": 0.0, "forecast": 0.0}
    )

    assert weights["fundamental"] == pytest.approx(0.76)
    assert weights["technical"] == pytest.approx(0.12)
    assert weights["forecast"] == pytest.approx(0.088)


def test_backtest_ics_to_weights_does_not_reward_negative_ic(monkeypatch):
    monkeypatch.setattr(config, "WEIGHT_SHRINKAGE", 0.0)
    monkeypatch.setattr(config, "WEIGHT_MIN_FLOOR", 0.03)
    monkeypatch.setattr(config, "PILLAR_WEIGHT_MIN_FLOOR", 0.03)
    monkeypatch.setattr(config, "PILLAR_WEIGHT_MAX_SINGLE", 0.55)

    weights = _ics_to_weights(
        {
            "technical": -0.20,
            "fundamental": 0.08,
            "sentiment": 0.04,
            "forecast": -0.10,
        },
        shrinkage=0.0,
        min_floor=0.03,
    )

    assert weights["fundamental"] > weights["technical"]
    assert weights["sentiment"] > weights["forecast"]
    assert max(weights.values()) <= 0.55
    assert weights["forecast"] <= 0.03


def test_adaptive_weights_block_when_parity_gate_fails(monkeypatch):
    monkeypatch.setattr(discovery_backtest, "pillar_parity_gate", lambda: (False, ["drifted"], {}))

    assert discovery_backtest.get_adaptive_weights(source="all", horizon="90d") is None


def test_pillar_parity_gate_blocks_critical_replay_missingness(monkeypatch, tmp_path):
    report_path = tmp_path / "parity.json"
    report_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "available": True,
                "sample": 10,
                "available_pairs": 10,
                "missing_replay": 0,
                "stale_replay": 0,
                "drifted_tickers": 0,
                "replay_missing_field_counts": {"forecast_score": 3},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "REPLAY_LIVE_PARITY_REPORT_PATH", str(report_path))
    monkeypatch.setattr(config, "ADAPTIVE_WEIGHTS_GATE_ENABLED", True)
    monkeypatch.setattr(config, "ADAPTIVE_WEIGHTS_GATE_REQUIRE_PARITY", True)
    monkeypatch.setattr(config, "ADAPTIVE_WEIGHTS_PARITY_CRITICAL_FIELDS", ["forecast_score"])
    monkeypatch.setattr(config, "ML_RANKER_PARITY_MAX_CRITICAL_MISSING_RATIO", 0.15)

    ok, reasons, summary = pillar_parity_gate()

    assert not ok
    assert any("critical replay missingness" in reason for reason in reasons)
    assert summary["critical_missing"]["forecast_score"] == pytest.approx(0.3)


def test_pillar_parity_gate_prefers_adaptive_weight_subreport(monkeypatch, tmp_path):
    report_path = tmp_path / "parity.json"
    report_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "available": True,
                "sample": 10,
                "available_pairs": 10,
                "missing_replay": 0,
                "stale_replay": 0,
                "drifted_tickers": 10,
                "replay_missing_field_counts": {"technical_score": 10},
                "adaptive_weight_parity": {
                    "available": True,
                    "sample": 10,
                    "available_pairs": 10,
                    "missing_replay": 0,
                    "stale_replay": 0,
                    "drifted_tickers": 1,
                    "replay_missing_field_counts": {},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "REPLAY_LIVE_PARITY_REPORT_PATH", str(report_path))
    monkeypatch.setattr(config, "ADAPTIVE_WEIGHTS_GATE_ENABLED", True)
    monkeypatch.setattr(config, "ADAPTIVE_WEIGHTS_GATE_REQUIRE_PARITY", True)
    monkeypatch.setattr(config, "ADAPTIVE_WEIGHTS_PARITY_CRITICAL_FIELDS", ["technical_score"])
    monkeypatch.setattr(config, "ML_RANKER_PARITY_MAX_AGE_HOURS", 999999)

    ok, reasons, summary = pillar_parity_gate()

    assert ok
    assert reasons == []
    assert summary["drifted_pair_ratio"] == pytest.approx(0.1)


def test_adaptive_weights_use_signed_ic_and_caps(monkeypatch):
    rows = [
        {"pillar": "technical", "information_coefficient": -0.20},
        {"pillar": "fundamental", "information_coefficient": 0.08},
        {"pillar": "sentiment", "information_coefficient": 0.06},
        {"pillar": "forecast", "information_coefficient": -0.10},
    ]
    monkeypatch.setattr(discovery_backtest, "pillar_parity_gate", lambda: (True, [], {}))
    monkeypatch.setattr(discovery_backtest, "_estimate_signal_halflife", lambda source: {p: 63.0 for p in ("technical", "fundamental", "sentiment", "forecast")})
    monkeypatch.setattr(config, "WEIGHT_SHRINKAGE", 0.0)
    monkeypatch.setattr(config, "PILLAR_WEIGHT_MIN_FLOOR", 0.03)
    monkeypatch.setattr(config, "PILLAR_WEIGHT_MAX_SINGLE", 0.55)
    monkeypatch.setattr(config, "ADAPTIVE_WEIGHTS_LIVE_BLEND", 0.60)

    weights = discovery_backtest._ic_rows_to_weights(rows, "all", "90d", min_samples=1000)

    assert weights is not None
    assert weights["fundamental"] > weights["technical"]
    assert max(weights.values()) <= 0.55
    assert weights["sentiment"] <= 0.08
    assert weights["forecast"] <= 0.03


def test_bayesian_pillar_effectiveness_uses_signed_ic(monkeypatch):
    monkeypatch.setattr(bayesian_learning, "pillar_parity_gate", lambda: (True, [], {}))
    monkeypatch.setattr(bayesian_learning, "_persist_weight_deltas", lambda **kwargs: None)
    monkeypatch.setattr(
        bayesian_learning,
        "_load_pillar_ic",
        lambda min_n=200: {
            ("discovery", "technical", "30d", None): (-0.20, 1000),
            ("discovery", "fundamental", "30d", None): (0.08, 1000),
            ("discovery", "sentiment", "30d", None): (0.06, 1000),
            ("discovery", "forecast", "30d", None): (-0.10, 1000),
        },
    )
    monkeypatch.setattr(config, "BAYESIAN_EFFECTIVENESS_MIN_SAMPLES", 100)
    monkeypatch.setattr(config, "BAYESIAN_HORIZON_CHAIN", ["30d"])
    monkeypatch.setattr(config, "BAYESIAN_MAX_LIVE_BLEND", 1.0)
    monkeypatch.setattr(config, "BAYESIAN_EFFECTIVENESS_PRIOR_STRENGTH", 0)
    monkeypatch.setattr(config, "BAYESIAN_DAILY_MAX_REL_DELTA", 10.0)
    monkeypatch.setattr(config, "PILLAR_WEIGHT_MAX_SINGLE", 0.55)
    monkeypatch.setattr(config, "ADAPTIVE_WEIGHTS_LIVE_BLEND", 0.60)

    weights = bayesian_learning._weights_from_pillar_effectiveness(
        {"technical": 0.25, "fundamental": 0.25, "sentiment": 0.25, "forecast": 0.25},
        source="discovery",
        regime=None,
    )

    assert weights is not None
    assert weights["fundamental"] > weights["technical"]
    assert max(weights.values()) <= 0.55
    assert weights["sentiment"] <= 0.08
    assert weights["forecast"] <= 0.03
