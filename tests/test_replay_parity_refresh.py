from __future__ import annotations

import json
from datetime import date

from utils import replay_parity_refresh as refresh


def test_replay_window_latest_uses_single_asof_date():
    assert refresh.replay_window(date(2026, 5, 3), frequency="latest", days=7) == (
        "2026-05-03",
        "2026-05-03",
    )


def test_replay_window_daily_uses_lookback_days():
    assert refresh.replay_window(date(2026, 5, 3), frequency="daily", days=7) == (
        "2026-04-27",
        "2026-05-03",
    )


def test_normalise_candidates_resolves_aliases_and_filters_quarantine():
    candidates = refresh._normalise_candidates(
        [
            {"ticker": "BAE.L"},
            {"ticker": "ARMN"},
            {"symbol": "CMC.L"},
            {"ticker": "BA.L"},
        ],
        limit=None,
    )

    assert [candidate["ticker"] for candidate in candidates] == ["BA.L", "CMCX.L"]


def test_build_fundamental_refresh_queue_prioritizes_replay_missing_fields():
    candidates = [
        {"ticker": "TCAP.L", "action": "BUY", "final_rank": 0.9},
        {"ticker": "AAPL", "action": "BUY", "final_rank": 0.8, "ready_contract_status": "PASS"},
        {"ticker": "MSFT", "action": "NEUTRAL", "final_rank": 0.2},
    ]
    missing = {
        "TCAP.L": {"available": True, "missing_fields": ["f_score", "gpa"], "replay_run_date": "2026-05-09"},
        "AAPL": {"available": True, "missing_fields": ["quality_factor_score", "f_score", "gpa"], "replay_run_date": "2026-05-09"},
        "MSFT": {"available": True, "missing_fields": [], "replay_run_date": "2026-05-09"},
    }

    payload = refresh.build_fundamental_refresh_queue(candidates, missing_by_ticker=missing, max_items=10)

    assert payload["count"] == 2
    assert payload["items"][0]["ticker"] == "AAPL"
    assert payload["items"][0]["fmp_statement_candidate"] is True
    assert payload["items"][1]["ticker"] == "TCAP.L"
    assert payload["items"][1]["fmp_statement_candidate"] is False


def test_build_fundamental_refresh_queue_skips_recent_unresolved_fmp_attempts():
    candidates = [
        {"ticker": "AAPL", "action": "BUY", "final_rank": 0.9},
        {"ticker": "MSFT", "action": "BUY", "final_rank": 0.8},
        {"ticker": "TCAP.L", "action": "BUY", "final_rank": 0.7},
    ]
    missing = {
        "AAPL": {"available": True, "missing_fields": ["f_score"], "replay_run_date": "2026-05-09"},
        "MSFT": {"available": True, "missing_fields": ["f_score"], "replay_run_date": "2026-05-09"},
        "TCAP.L": {"available": True, "missing_fields": ["f_score"], "replay_run_date": "2026-05-09"},
    }
    ledger = {
        "tickers": {
            "AAPL": {
                "last_attempted_at": "2026-05-09T08:00:00",
                "last_resolved": False,
                "last_missing_fields": ["f_score"],
            }
        }
    }

    payload = refresh.build_fundamental_refresh_queue(
        candidates,
        missing_by_ticker=missing,
        attempt_ledger=ledger,
        skip_recent_attempts=True,
        attempt_cooldown_hours=24,
        generated_at="2026-05-09T10:00:00",
        max_items=10,
    )

    assert [item["ticker"] for item in payload["items"]] == ["MSFT", "TCAP.L"]
    assert payload["skipped_recent_count"] == 1
    assert payload["skipped_recent_attempts"][0]["ticker"] == "AAPL"


def test_attempt_ledger_records_and_finalizes_attempts(tmp_path, monkeypatch):
    ledger_path = tmp_path / "ledger.json"
    queue_payload = {
        "items": [
            {"ticker": "AAPL", "priority": 3.2, "missing_fields": ["f_score"], "fmp_statement_candidate": True}
        ]
    }
    refresh_payload = {"fmp_results": {"AAPL": 16}, "yfinance_results": {}}

    attempted = refresh._record_fundamental_attempts(
        queue_payload=queue_payload,
        refresh_payload=refresh_payload,
        ledger_path=ledger_path,
    )

    assert attempted == ["AAPL"]
    first = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert first["tickers"]["AAPL"]["last_snapshots_written"] == 16
    assert first["tickers"]["AAPL"]["attempt_count"] == 1

    monkeypatch.setattr(
        refresh,
        "_latest_replay_fundamental_missing",
        lambda tickers: {
            "AAPL": {
                "available": True,
                "missing_fields": ["gpa"],
                "replay_run_date": "2026-05-09",
            }
        },
    )

    refresh._finalize_fundamental_attempts(["AAPL"], ledger_path=ledger_path)

    final = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert final["tickers"]["AAPL"]["last_resolved"] is False
    assert final["tickers"]["AAPL"]["last_missing_fields"] == ["gpa"]
    assert final["tickers"]["AAPL"]["unresolved_attempt_count"] == 1


def test_refresh_fundamentals_for_parity_writes_fmp_only_queue(tmp_path, monkeypatch):
    queue_path = tmp_path / "replay_queue.json"
    captured = {}

    monkeypatch.setattr(
        refresh,
        "_latest_replay_fundamental_missing",
        lambda tickers: {
            "AAPL": {"available": True, "missing_fields": ["f_score"], "replay_run_date": "2026-05-09"},
            "TCAP.L": {"available": True, "missing_fields": ["gpa"], "replay_run_date": "2026-05-09"},
        },
    )

    def fake_refresh_queue_tickers(**kwargs):
        captured.update(kwargs)
        return {"selected": 1, "refreshed": 1, "snapshots_written": 4}

    monkeypatch.setattr("utils.pit_backfill.refresh_queue_tickers", fake_refresh_queue_tickers)
    monkeypatch.setattr(refresh, "reset_pit_cache", lambda: captured.setdefault("cache_reset", True))

    result = refresh._refresh_fundamentals_for_parity(
        [{"ticker": "AAPL"}, {"ticker": "TCAP.L"}],
        queue_path=queue_path,
        attempt_ledger_path=tmp_path / "ledger.json",
        max_tickers=5,
        limit=12,
        allow_yfinance_fallback=False,
    )

    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    assert queue["count"] == 2
    assert captured["queue_path"] == queue_path
    assert captured["max_tickers"] == 5
    assert captured["limit"] == 12
    assert captured["yfinance_fallback"] is False
    assert captured["fmp_only"] is True
    assert captured["cache_reset"] is True
    assert result["queue_count"] == 2
    assert result["queue_fmp_candidates"] == 1
