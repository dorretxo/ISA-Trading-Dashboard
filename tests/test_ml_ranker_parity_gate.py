from __future__ import annotations

from datetime import datetime

from engine import ml_ranker


def _configure_parity(monkeypatch, *, use_actionable: bool = False):
    settings = {
        "ML_RANKER_PARITY_GATE_ENABLED": True,
        "ML_RANKER_PARITY_GATE_REQUIRE_REPORT": True,
        "ML_RANKER_PARITY_MAX_AGE_HOURS": 999999,
        "ML_RANKER_PARITY_MIN_AVAILABLE_RATIO": 0.70,
        "ML_RANKER_PARITY_MAX_MISSING_REPLAY_RATIO": 0.20,
        "ML_RANKER_PARITY_MAX_STALE_RATIO": 0.10,
        "ML_RANKER_PARITY_MAX_DRIFTED_PAIR_RATIO": 0.25,
        "ML_RANKER_PARITY_MAX_CRITICAL_MISSING_RATIO": 0.15,
        "ML_RANKER_PARITY_CRITICAL_FIELDS": ["strong_buy_eligible", "r_r_ratio"],
        "ML_RANKER_PARITY_USE_ACTIONABLE_DRIFT": use_actionable,
        "ML_DRIFT_MONITOR_ENABLED": False,
    }
    for name, value in settings.items():
        monkeypatch.setattr(ml_ranker.config, name, value, raising=False)


def _report(**overrides):
    base = {
        "available": True,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sample": 250,
        "available_pairs": 225,
        "missing_replay": 10,
        "stale_replay": 5,
        "drifted_tickers": 20,
        "replay_missing_field_counts": {},
    }
    base.update(overrides)
    return base


def test_replay_live_parity_gate_passes_healthy_report(monkeypatch):
    _configure_parity(monkeypatch)

    ok, reasons, summary = ml_ranker._replay_live_parity_gate(_report())

    assert ok is True
    assert reasons == []
    assert summary["available_ratio"] == 0.9


def test_replay_live_parity_gate_blocks_sparse_drifted_report(monkeypatch):
    _configure_parity(monkeypatch)

    ok, reasons, summary = ml_ranker._replay_live_parity_gate(
        _report(
            available_pairs=110,
            missing_replay=80,
            stale_replay=54,
            drifted_tickers=102,
            replay_missing_field_counts={"strong_buy_eligible": 100},
        )
    )

    assert ok is False
    assert any("available_pairs_ratio" in reason for reason in reasons)
    assert any("missing_replay_ratio" in reason for reason in reasons)
    assert any("stale_replay_ratio" in reason for reason in reasons)
    assert any("drifted_pair_ratio" in reason for reason in reasons)
    assert any("critical replay missingness" in reason for reason in reasons)
    assert summary["critical_missing"]["strong_buy_eligible"] > 0.15


def test_replay_live_parity_gate_actionable_mode_passes_when_raw_would_fail(monkeypatch):
    """Today's 2026-05-15 case in miniature: raw drift is 64% (would block)
    but actionable drift is 18.6% (passes).  Gate must read actionable when
    the report carries those fields and the config flag is True."""
    _configure_parity(monkeypatch, use_actionable=True)

    ok, reasons, summary = ml_ranker._replay_live_parity_gate(
        _report(
            available_pairs=247,
            missing_replay=3,
            stale_replay=0,
            drifted_tickers=160,  # raw — would fail (160/247 = 65% > 25%)
            actionable_available_pairs=145,
            actionable_drifted_tickers=27,  # actionable — passes (27/145 = 18.6%)
            actionable_replay_missing_field_counts={},
            replay_missing_field_counts={"strong_buy_eligible": 100},  # raw missingness
        )
    )

    assert ok is True
    assert reasons == []
    assert summary["drift_mode"] == "actionable"
    assert summary["actionable_drifted_tickers"] == 27
    assert summary["actionable_available_pairs"] == 145
    assert summary["drifted_pair_ratio"] < 0.25


def test_replay_live_parity_gate_actionable_mode_falls_back_when_fields_absent(monkeypatch):
    """If the report has no actionable_* fields (older orchestrator write or
    a stale cache), the gate must fall back to raw drifted_tickers behaviour
    rather than crash or silently pass."""
    _configure_parity(monkeypatch, use_actionable=True)

    ok, reasons, summary = ml_ranker._replay_live_parity_gate(_report())  # no actionable_* keys

    # Should behave exactly like the healthy-report test — same raw values.
    assert ok is True
    assert reasons == []
    assert summary["drift_mode"] == "raw"
    assert summary["actionable_drifted_tickers"] is None


def test_replay_live_parity_gate_flag_off_uses_raw_even_with_actionable_fields(monkeypatch):
    """Rollback path: flipping ML_RANKER_PARITY_USE_ACTIONABLE_DRIFT to False
    must restore raw-mode behaviour even when actionable counts are present
    in the report."""
    _configure_parity(monkeypatch, use_actionable=False)

    ok, reasons, summary = ml_ranker._replay_live_parity_gate(
        _report(
            available_pairs=247,
            drifted_tickers=160,
            actionable_available_pairs=145,
            actionable_drifted_tickers=27,
        )
    )

    # With flag off, raw 160/247=65% drift trips the gate.
    assert ok is False
    assert any("drifted_pair_ratio" in r for r in reasons)
    assert summary["drift_mode"] == "raw"


def test_promotion_eligibility_requires_parity_gate(monkeypatch):
    _configure_parity(monkeypatch)
    monkeypatch.setattr(ml_ranker, "_load_persisted_model", lambda: False)
    monkeypatch.setattr(
        ml_ranker,
        "_replay_live_parity_gate",
        lambda report=None: (False, ["parity failed"], {"enabled": True, "blockers": ["parity failed"]}),
    )
    ml_ranker._model_cache.update(
        {
            "promotion_eligible": True,
            "n_samples": 1000,
            "oos_rank_ic": 0.10,
            "oos_r_squared": 0.01,
        }
    )

    assert ml_ranker.is_promotion_eligible() is False
