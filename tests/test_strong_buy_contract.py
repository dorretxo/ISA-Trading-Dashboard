from __future__ import annotations

from dataclasses import dataclass, field

import engine.discovery as discovery


@dataclass
class _Cand:
    ticker: str
    action: str = "NEUTRAL"
    aggregate_score: float = 0.5
    f_score: int | None = 8
    f_score_coverage: float = 1.0
    trap_safeguard_triggered: bool = False
    gate_v2_status: str = "PASS"
    ticker_identity_warning: str | None = None
    sb_score: float = 1.2
    strong_buy_eligible: bool = True
    institutional_prior_percentile: float = 0.9
    ready_contract_core_status: str = "PASS"
    ready_contract_status: str = "PASS"
    ready_contract_reasons: list = field(default_factory=list)
    strong_buy_blockers: list = field(default_factory=list)
    meta_success_prob: float | None = 0.8
    action_gate_ceiling: str = "STRONG BUY"
    action_gate_reasons: list = field(default_factory=list)
    entry_stance: str = "Ready"
    entry_price: float | None = 100.0
    stop_loss: float | None = 94.0
    take_profit: float | None = 115.0
    r_r_ratio: float | None = 2.5
    position_weight: float = 0.02
    effective_data_confidence: float | None = 0.9
    scorecard_override: bool = False
    scorecard_override_blockers: list = field(default_factory=list)


def _configure_percentile_contract(monkeypatch):
    settings = {
        "USE_PERCENTILE_ACTIONS": True,
        "STRONG_BUY_SCORECARD_ENABLED": True,
        "STRONG_BUY_SCORECARD_MIN_Z": 0.5,
        "STRONG_BUY_SCORECARD_TOP_PCT": 1.0,
        "STRONG_BUY_SCORECARD_OVERRIDE_ENABLED": True,
        "STRONG_BUY_SCORECARD_OVERRIDE_Z": 1.0,
        "STRONG_BUY_SCORECARD_OVERRIDE_TOP_N": 5,
        "PERCENTILE_STRONG_BUY_PCT": 1.0,
        "PERCENTILE_BUY_PCT": 1.0,
        "PERCENTILE_NEUTRAL_PCT": 1.0,
        "PERCENTILE_STRONG_BUY_MIN_AGG": 0.0,
        "PERCENTILE_BUY_MIN_AGG": 0.0,
        "PERCENTILE_STRONG_BUY_MIN_PRIOR_PCT": 0.85,
        "SCORE_KEEP_THRESHOLD": -999.0,
        "F_SCORE_GATE_ENABLED": False,
        "ACTION_GATES_ENABLED": False,
        "ACTION_GATES_SHADOW": False,
        "THRESHOLD_LEARNER_ENABLED": False,
        "META_LABEL_STRONG_BUY_MIN_PROB": 0.6,
        "READY_STRONG_BUY_MIN_RR": 1.5,
        "READY_STRONG_BUY_MIN_POSITION_WEIGHT": 0.005,
        "READY_STRONG_BUY_MIN_CONFIDENCE": 0.70,
    }
    for name, value in settings.items():
        monkeypatch.setattr(discovery.config, name, value, raising=False)

    monkeypatch.setattr(
        discovery,
        "compute_strong_buy_scorecard",
        lambda candidates: {c.ticker: c.sb_score for c in candidates},
    )


def test_scorecard_override_cannot_restore_failed_ready_contract(monkeypatch):
    _configure_percentile_contract(monkeypatch)
    candidate = _Cand(
        ticker="HOTFAIL",
        ready_contract_status="FAIL",
        strong_buy_eligible=False,
        action_gate_ceiling="BUY",
        action_gate_reasons=["Entry stance blocks STRONG BUY"],
        entry_stance="Watch Only",
    )

    discovery._assign_percentile_actions([candidate])

    assert candidate.scorecard_override is True
    assert candidate.action == "BUY"
    assert candidate.ready_contract_status == "FAIL"
    assert candidate.strong_buy_eligible is False
    assert any("ready contract FAIL" in reason for reason in candidate.strong_buy_blockers)
    assert any("action gate ceiling BUY" in reason for reason in candidate.strong_buy_blockers)


def test_scorecard_override_allows_contract_clean_strong_buy(monkeypatch):
    _configure_percentile_contract(monkeypatch)
    candidate = _Cand(ticker="CLEAN")

    discovery._assign_percentile_actions([candidate])

    assert candidate.action == "STRONG BUY"
    assert candidate.ready_contract_status == "PASS"
    assert candidate.strong_buy_eligible is True
    assert candidate.strong_buy_blockers == []


def test_meta_floor_remains_load_bearing_for_strong_buy(monkeypatch):
    _configure_percentile_contract(monkeypatch)
    candidate = _Cand(ticker="LOWMETA", meta_success_prob=0.4)

    discovery._assign_percentile_actions([candidate])

    assert candidate.action == "BUY"
    assert candidate.ready_contract_status == "FAIL"
    assert any("Meta-label confidence" in reason for reason in candidate.strong_buy_blockers)
