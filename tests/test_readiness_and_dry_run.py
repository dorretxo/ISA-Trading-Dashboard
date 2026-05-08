from utils.discovery_digest import candidate_entry_ready, candidate_entry_trigger, candidate_readiness_summary
from daily_orchestrator import _evaluate_swaps, _paper_trading_enabled_for_run


def test_buy_candidate_readiness_summary_includes_trigger():
    candidate = {
        "ticker": "CM.TO",
        "action": "BUY",
        "entry_stance": "Pullback Preferred",
        "ready_contract_status": "FAIL",
        "ready_contract_reasons": ["gate status REVIEW", "entry stance is Pullback Preferred"],
        "entry_price": 148.08,
        "currency": "CAD",
        "position_weight": 0.02,
        "r_r_ratio": 1.8,
    }

    summary = candidate_readiness_summary(candidate)

    assert "Buy candidate, not ready" in summary
    assert "CAD 148.08" in summary
    assert candidate_entry_trigger(candidate) == "Wait for a pullback toward CAD 148.08."


def test_dry_run_does_not_write_paper_trading_by_default():
    assert _paper_trading_enabled_for_run(True) is False
    assert _paper_trading_enabled_for_run(False) is True


def test_entry_ready_requires_ready_contract_pass():
    candidate = {
        "ticker": "BUYFAIL",
        "action": "BUY",
        "entry_stance": "Ready",
        "ready_contract_status": "FAIL",
        "entry_price": 100.0,
        "stop_loss": 94.0,
        "take_profit": 115.0,
        "r_r_ratio": 2.5,
        "position_weight": 0.03,
    }

    assert candidate_entry_ready(candidate) is False


def test_swaps_require_entry_ready_candidate():
    results = [{"ticker": "WEAK", "aggregate_score": -0.4, "action": "SELL"}]
    candidates = [
        {
            "ticker": "NOTREADY",
            "action": "BUY",
            "final_rank": 0.5,
            "aggregate_score": 0.4,
            "portfolio_fit_score": 1.0,
            "entry_stance": "Ready",
            "ready_contract_status": "FAIL",
            "entry_price": 100.0,
            "stop_loss": 94.0,
            "take_profit": 115.0,
            "r_r_ratio": 2.5,
            "position_weight": 0.03,
        }
    ]

    assert _evaluate_swaps(results, candidates, {"cooldowns": {}}) == []
