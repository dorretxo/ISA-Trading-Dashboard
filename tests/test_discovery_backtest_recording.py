import json

from engine.discovery_backtest import record_discovery_picks
from engine import paper_trading


def _candidate(action="BUY", **overrides):
    base = {
        "ticker": "KLR.L",
        "name": "Keller Group",
        "current_price": 12.0,
        "action": action,
        "aggregate_score": 0.41,
        "final_rank": 0.35,
        "technical_score": 0.2,
        "fundamental_score": 0.3,
        "sentiment_score": 0.1,
        "forecast_score": 0.2,
        "momentum_score": 0.7,
        "sector": "Industrials",
        "exchange": "LSE",
        "entry_stance": "Ready",
        "entry_price": 12.0,
        "stop_loss": 10.8,
        "take_profit": 15.0,
        "r_r_ratio": 2.5,
        "position_weight": 0.04,
        "ready_contract_status": "PASS",
        "strong_buy_eligible": action == "STRONG BUY",
        "meta_success_prob": 0.51,
        "qmj_component_count": 2,
        "pit_source": "yfinance",
    }
    base.update(overrides)
    return base


def test_record_discovery_picks_upserts_same_day_action_upgrade(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_trading, "DB_PATH", tmp_path / "paper_trading.db")

    assert record_discovery_picks([_candidate("BUY")]) == 1
    assert record_discovery_picks([
        _candidate(
            "STRONG BUY",
            final_rank=0.62,
            aggregate_score=0.48,
            strong_buy_blockers=[],
        )
    ]) == 1

    with paper_trading._connect() as conn:
        rows = conn.execute(
            "SELECT ticker, action, final_rank, aggregate_score, meta_prob, "
            "ready_contract_status, strong_buy_eligible, qmj_component_count, pit_source "
            "FROM signal_backtest WHERE source='discovery'"
        ).fetchall()

    assert len(rows) == 1
    row = dict(rows[0])
    assert row["ticker"] == "KLR.L"
    assert row["action"] == "STRONG BUY"
    assert row["final_rank"] == 0.62
    assert row["aggregate_score"] == 0.48
    assert row["meta_prob"] == 0.51
    assert row["ready_contract_status"] == "PASS"
    assert row["strong_buy_eligible"] == 1
    assert row["qmj_component_count"] == 2
    assert row["pit_source"] == "yfinance"


def test_live_sanity_report_confirms_cache_email_and_backtest(tmp_path, monkeypatch):
    import daily_orchestrator

    monkeypatch.setattr(paper_trading, "DB_PATH", tmp_path / "paper_trading.db")
    monkeypatch.setattr(daily_orchestrator, "_LIVE_RUN_SANITY_REPORT", tmp_path / "sanity.json")
    monkeypatch.setattr(daily_orchestrator, "_FINAL_STRONG_BUY_VETO_REPORT", tmp_path / "veto.json")

    candidate = _candidate("STRONG BUY", final_rank=0.62, aggregate_score=0.48)
    assert record_discovery_picks([candidate]) == 1
    (tmp_path / "veto.json").write_text(
        json.dumps({
            "core_ready_candidates": 1,
            "core_meta_threshold": 0.47,
            "veto_layer_counts": {},
            "candidates": [{"ticker": "KLR.L", "action": "STRONG BUY"}],
        }),
        encoding="utf-8",
    )

    report = daily_orchestrator._write_live_run_sanity_report(
        [candidate],
        dry_run=True,
        discovery_ran=True,
        email_sent=True,
        email_subject="[ISA Alert] Discovery: KLR.L STRONG BUY ready",
        email_html="<html><body>KLR.L</body></html>",
        n_recorded=1,
    )

    assert report["ok"] is True
    assert report["cached"]["strong_buy_tickers"] == ["KLR.L"]
    assert report["email"]["strong_buy_tickers_visible"] == ["KLR.L"]
    assert report["signal_backtest"]["strong_buy_tickers"] == ["KLR.L"]
    assert report["final_veto_report"]["strong_buy_count"] == 1
    assert (tmp_path / "sanity.json").exists()
