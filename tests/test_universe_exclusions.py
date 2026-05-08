from datetime import datetime, timedelta

import pandas as pd

import config
from engine.discovery import (
    _candidate_exclusion_reason,
    _derive_entry_stance,
    _evaluate_ready_strong_buy_contract,
    _filter_excluded_candidates,
    _promote_challenge_reserve,
)
from utils.global_universe import (
    get_full_universe,
    get_global_universe,
    get_region_for_ticker,
    get_universe_stats,
    is_excluded_ticker,
    is_quarantined_ticker,
    resolve_yahoo_ticker,
)


class PassingPrior:
    passes_strong_buy_bar = True
    reasons = []


class SoftPassingPrior:
    passes_strong_buy_bar = False
    percentile = 0.82
    confidence = 0.60
    coverage = 0.42
    score = 0.10
    reasons = ["institutional prior below top-decile bar (82%)", "prior coverage thin (42%)"]


class FailingPrior:
    passes_strong_buy_bar = False
    percentile = 0.60
    confidence = 0.60
    coverage = 0.42
    score = 0.10
    reasons = ["institutional prior below top-decile bar (60%)", "prior coverage thin (42%)"]


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


def test_static_universe_resolves_yahoo_aliases_and_deduplicates():
    assert resolve_yahoo_ticker("BAE.L") == "BA.L"
    assert resolve_yahoo_ticker("CMC.L") == "CMCX.L"
    assert resolve_yahoo_ticker("DSM.AS") == "DSFIR.AS"

    tickers = [entry.ticker.upper() for entry in get_full_universe()]
    assert len(tickers) == len(set(tickers))
    assert "BAE.L" not in tickers
    assert "BA.L" in tickers
    assert "CMC.L" not in tickers
    assert "CMCX.L" in tickers
    assert "DSM.AS" not in tickers
    assert "DSFIR.AS" in tickers

    global_tickers = {ticker.upper() for ticker in get_global_universe(include_dynamic=False)}
    assert {"BA.L", "CMCX.L", "DSFIR.AS"} <= global_tickers
    assert not {"BAE.L", "CMC.L", "DSM.AS"} & global_tickers
    assert get_region_for_ticker("CMCX.L") == "UK"


def test_static_universe_quarantines_dead_yahoo_symbols():
    assert is_quarantined_ticker("ARMN")
    assert is_excluded_ticker("ARMN")
    assert is_quarantined_ticker("CINE.L")
    assert is_excluded_ticker("CINE.L")
    assert is_excluded_ticker("CSGN.SW")

    tickers = {entry.ticker.upper() for entry in get_full_universe()}
    assert "CINE.L" not in tickers
    assert "CSGN.SW" not in tickers
    assert "NCM.AX" not in tickers


def test_feature_store_filters_quarantined_symbols_before_download(monkeypatch):
    from utils import feature_store

    calls = []

    def fake_download(tickers, *args, **kwargs):
        rendered = repr(tickers)
        assert "ARMN" not in rendered
        assert "PHNX.L" not in rendered
        calls.append(tickers)
        if tickers == "SPY":
            return pd.DataFrame({"Close": [100.0] * 80})
        return pd.DataFrame()

    monkeypatch.setattr("yfinance.download", fake_download)

    feature_store.compute_batch_factors(["ARMN", "PHNX.L", "SPY"], batch_size=10)

    assert calls
    assert all("ARMN" not in repr(call) and "PHNX.L" not in repr(call) for call in calls)


def test_batched_evaluator_filters_quarantined_symbols_before_download(monkeypatch):
    from engine import discovery_backtest
    from utils import price_store

    def fake_cache_download(tickers, *args, **kwargs):
        assert "ARMN" not in tickers
        assert "PHNX.L" not in tickers
        return {}

    def fake_yahoo_download(tickers, *args, **kwargs):
        rendered = repr(tickers)
        assert "ARMN" not in rendered
        assert "PHNX.L" not in rendered
        raise AssertionError("Excluded-only request should not fall back to yfinance")

    monkeypatch.setattr(price_store, "download_price_history", fake_cache_download)
    monkeypatch.setattr(discovery_backtest.yf, "download", fake_yahoo_download)

    frames = discovery_backtest._download_price_frames(
        ["ARMN", "PHNX.L"],
        datetime.now() - timedelta(days=10),
        datetime.now(),
    )

    assert frames == {}


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


def test_ready_contract_soft_passes_near_miss_prior():
    status, reasons, details = _evaluate_ready_strong_buy_contract(
        gate_status="PASS",
        trap_triggered=False,
        prior=SoftPassingPrior(),
        entry_stance="Ready",
        entry_price=10.0,
        stop_loss=9.0,
        take_profit=12.0,
        rr_ratio=2.0,
        position_weight=0.02,
        data_confidence=0.80,
        return_details=True,
    )

    assert status == "PASS"
    assert reasons == []
    assert details["score"] >= 0.95
    assert details["soft_passes"]


def test_ready_contract_rejects_prior_below_soft_band():
    status, reasons = _evaluate_ready_strong_buy_contract(
        gate_status="PASS",
        trap_triggered=False,
        prior=FailingPrior(),
        entry_stance="Ready",
        entry_price=10.0,
        stop_loss=9.0,
        take_profit=12.0,
        rr_ratio=2.0,
        position_weight=0.02,
        data_confidence=0.80,
    )

    assert status == "FAIL"
    assert "institutional prior below top-decile bar (60%)" in reasons


def test_near_high_is_ready_when_not_stretched_and_upside_ok():
    assert _derive_entry_stance(
        governance_flag=False,
        asymmetric_risk_flag=False,
        earnings_imminent=False,
        is_parabolic=False,
        analyst_upside=12.0,
        near_52w_high=True,
        return_30d=0.06,
        insider_sells=0,
        insider_buys=0,
        earnings_near=False,
    ) == "Ready"


def test_near_high_stretched_still_prefers_pullback():
    assert _derive_entry_stance(
        governance_flag=False,
        asymmetric_risk_flag=False,
        earnings_imminent=False,
        is_parabolic=False,
        analyst_upside=12.0,
        near_52w_high=True,
        return_30d=0.16,
        insider_sells=0,
        insider_buys=0,
        earnings_near=False,
    ) == "Pullback Preferred"


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
