from __future__ import annotations

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
