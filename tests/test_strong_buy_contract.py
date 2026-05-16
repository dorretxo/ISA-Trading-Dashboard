from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

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


def test_drift_active_conservative_profile_still_allows_clean_strong_buy(monkeypatch, tmp_path):
    _configure_percentile_contract(monkeypatch)
    monkeypatch.setattr(discovery.config, "THRESHOLD_LEARNER_ENABLED", True, raising=False)
    monkeypatch.setattr(discovery.config, "ACTION_GATES_ENABLED", True, raising=False)
    monkeypatch.setattr(
        discovery.config,
        "THRESHOLD_LEARNER_STATE_FILE",
        str(tmp_path / "threshold_state.json"),
        raising=False,
    )

    from engine.threshold_learner import ThresholdLearnerState, save_state

    state = ThresholdLearnerState()
    state.drift_active = True
    state.posteriors["aggressive"].update(successes=100, failures=10)
    save_state(state, config_module=discovery.config)

    candidate = SimpleNamespace(
        ticker="CLEANCON",
        action="NEUTRAL",
        aggregate_score=0.9,
        f_score=9,
        f_score_coverage=1.0,
        trap_safeguard_triggered=False,
        gate_v2_status="PASS",
        sb_score=1.5,
        strong_buy_eligible=True,
        institutional_prior_percentile=0.95,
        institutional_prior_confidence=0.90,
        institutional_prior_coverage=0.90,
        ready_contract_core_status="PASS",
        ready_contract_status="PASS",
        ready_contract_reasons=[],
        strong_buy_blockers=[],
        meta_success_prob=0.90,
        action_gate_ceiling="STRONG BUY",
        action_gate_reasons=[],
        entry_stance="Ready",
        entry_price=100.0,
        stop_loss=92.0,
        take_profit=120.0,
        r_r_ratio=2.5,
        position_weight=0.02,
        effective_data_confidence=0.95,
        scorecard_override=False,
        scorecard_override_blockers=[],
        sector="Technology",
        altman_z=5.0,
        beneish_m=-3.0,
        accruals_factor_score=0.50,
        investment_factor_score=0.30,
        net_debt_ebitda=0.2,
        ev_ebit=16.0,
        ev_sales=3.5,
        pe_ratio=20.0,
        pe_forward=20.0,
        eps_growth_3y_cagr=0.18,
        qmj_factor_score=0.95,
        qmj_component_count=3,
        gpa=0.85,
        gpa_score=0.90,
        roic=0.28,
        wacc=0.10,
        op_margin_yoy_delta=0.04,
        rsi=58,
        price_vs_sma200_stretch=0.12,
        realized_vol_pctile=0.30,
        ready_contract_score=1.0,
        pit_source="fmp_full",
        current_price=100.0,
        sma_200=90.0,
        support_levels={},
        atr=2.0,
    )

    discovery._assign_percentile_actions([candidate])

    assert candidate.threshold_profile == "conservative"
    assert candidate.action_gate_ceiling == "STRONG BUY"
    assert candidate.action == "STRONG BUY"
    assert candidate.strong_buy_eligible is True
    assert candidate.strong_buy_blockers == []
