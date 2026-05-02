import config
from engine.discovery import (
    _candidate_exclusion_reason,
    _evaluate_ready_strong_buy_contract,
    _filter_excluded_candidates,
    _promote_challenge_reserve,
)
from utils.global_universe import (
    get_full_universe,
    get_global_universe,
    get_universe_stats,
    is_excluded_ticker,
)


class PassingPrior:
    passes_strong_buy_bar = True
    reasons = []


def test_korea_is_excluded_from_static_universe_and_stats():
    universe = get_full_universe()
    tickers = {entry.ticker.upper() for entry in universe}

    assert "005930.KS" not in tickers
    assert all(not ticker.endswith((".KS", ".KQ")) for ticker in tickers)

    global_tickers = {ticker.upper() for ticker in get_global_universe(include_dynamic=False)}
    assert "005930.KS" not in global_tickers
    assert all(not ticker.endswith((".KS", ".KQ")) for ticker in global_tickers)

    stats = get_universe_stats()
    assert "KR" not in stats["by_country"]
    assert "South Korea" not in stats["by_region"]


def test_candidate_exclusion_blocks_korea_and_common_adrs():
    assert is_excluded_ticker("005930.KS")
    assert is_excluded_ticker("SSNLF")
    assert _candidate_exclusion_reason({"symbol": "005930.KS", "country": "KR"})
    assert _candidate_exclusion_reason({"symbol": "SSNLF", "country": "US"})
    assert _candidate_exclusion_reason({"symbol": "ABC", "country": "South Korea"})
    assert _candidate_exclusion_reason({"symbol": "BA.L", "country": "GB"}) is None


def test_filter_excluded_candidates_records_rejections():
    rejections = []
    kept = _filter_excluded_candidates(
        [
            {"symbol": "005930.KS", "companyName": "Samsung Electronics", "country": "KR"},
            {"symbol": "SLB", "companyName": "SLB", "country": "US"},
        ],
        rejections,
    )

    assert [candidate["symbol"] for candidate in kept] == ["SLB"]
    assert len(rejections) == 1
    assert rejections[0].ticker == "005930.KS"


def test_missing_data_review_does_not_fail_ready_contract():
    status, reasons = _evaluate_ready_strong_buy_contract(
        gate_status="REVIEW",
        trap_triggered=False,
        prior=PassingPrior(),
        entry_stance="Ready",
        entry_price=10.0,
        stop_loss=9.0,
        take_profit=12.0,
        rr_ratio=2.0,
        position_weight=0.02,
        data_confidence=0.80,
        gate_reasons=["F-score missing/low coverage", "GPA missing"],
    )

    assert status == "PASS"
    assert reasons == []


def test_non_missing_review_still_fails_ready_contract():
    status, reasons = _evaluate_ready_strong_buy_contract(
        gate_status="REVIEW",
        trap_triggered=False,
        prior=PassingPrior(),
        entry_stance="Ready",
        entry_price=10.0,
        stop_loss=9.0,
        take_profit=12.0,
        rr_ratio=2.0,
        position_weight=0.02,
        data_confidence=0.80,
        gate_reasons=["No clear value/FCF/growth support"],
    )

    assert status == "FAIL"
    assert "gate status REVIEW" in reasons


def test_challenge_reserve_promotes_configured_name(monkeypatch):
    monkeypatch.setattr(config, "DISCOVERY_CHALLENGE_RESERVE_ENABLED", True)
    monkeypatch.setattr(config, "DISCOVERY_CHALLENGE_AUTO_ENABLED", False)
    monkeypatch.setattr(config, "DISCOVERY_CHALLENGE_TICKERS", ["BA.L"])

    pool = [
        {"symbol": "A", "_quick_score": 0.90},
        {"symbol": "B", "_quick_score": 0.80},
        {"symbol": "BA.L", "_quick_score": 0.10, "country": "GB"},
    ]
    selected = [pool[0], pool[1]]

    promoted = _promote_challenge_reserve(
        pool,
        selected,
        target_n=2,
        reserve_n=1,
        score_key="_quick_score",
    )

    assert "BA.L" in {_candidate["symbol"] for _candidate in promoted}
    assert len(promoted) == 2
