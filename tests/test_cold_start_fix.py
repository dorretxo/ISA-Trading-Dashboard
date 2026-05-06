"""Unit tests for the Phase 1 + 2 + 3 cold-start fix helpers.

Covers:
  - ``_zscore_array``                   — cross-sectional z-score primitive
  - ``apply_action_label``              — pure label-assignment helper
  - ``compute_strong_buy_scorecard``    — additive scorecard composite
  - ``backfill_panel_actions``          — replay re-labelling (upgrade-only)
  - ``record_borderline_signals``       — active-label boundary recording
  - ``_blend_multi_horizon_ic``         — multi-horizon IC weighting

These tests lock in the current correct behaviour so future changes to the
discovery scoring pipeline do not silently regress STRONG BUY emission
quality.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from engine import paper_trading
from engine.discovery import (
    _zscore_array,
    apply_action_label,
    compute_strong_buy_scorecard,
)
from engine.discovery_backtest import (
    _blend_multi_horizon_ic,
    backfill_panel_actions,
    init_backtest_db,
    record_borderline_signals,
)


# ---------------------------------------------------------------------------
# _zscore_array — primitive
# ---------------------------------------------------------------------------

def test_zscore_uniform_input():
    """All-equal input -> all zeros (no z dispersion)."""
    out = _zscore_array([3.0, 3.0, 3.0, 3.0])
    assert all(abs(v) < 1e-9 for v in out)


def test_zscore_handles_nan_and_none():
    """None and NaN are passed through as 0.0 in output."""
    out = _zscore_array([1.0, None, 5.0, float("nan"), 3.0])
    assert out[1] == 0.0  # None -> 0
    assert out[3] == 0.0  # NaN -> 0
    # The three real values get standard z-score (sd-population formula)
    assert out[0] < 0
    assert out[2] > 0
    assert abs(out[4]) < 1e-9


def test_zscore_single_finite_value_returns_zero():
    """Less than 2 finite values -> all zeros."""
    out = _zscore_array([None, 5.0, None, None])
    assert all(v == 0.0 for v in out)


# ---------------------------------------------------------------------------
# apply_action_label — pure function
# ---------------------------------------------------------------------------

def _result(**overrides):
    base = {
        "aggregate_score": 0.50,
        "f_score": 7,
        "f_score_coverage": 0.9,
        "gpa": 0.20,
        "ev_ebit": 12.0,
        "fcf_yield": 0.05,
        "revenue_growth": 0.08,
        "price_vs_sma200_stretch": 0.10,
        "name": "Test",
    }
    base.update(overrides)
    return base


def test_apply_action_label_strong_buy():
    action, gate, trap = apply_action_label(
        _result(aggregate_score=0.55),
        pillars_all_zero=False,
        adjusted_aggregate=0.55,
        sector="Technology",
    )
    assert action == "STRONG BUY"
    assert trap is False


def test_apply_action_label_buy():
    action, _g, _t = apply_action_label(
        _result(aggregate_score=0.30),
        pillars_all_zero=False,
        adjusted_aggregate=0.30,
        sector="Industrials",
    )
    assert action == "BUY"


def test_apply_action_label_neutral():
    action, _g, _t = apply_action_label(
        _result(aggregate_score=0.05),
        pillars_all_zero=False,
        adjusted_aggregate=0.05,
        sector="Healthcare",
    )
    assert action == "NEUTRAL"


def test_apply_action_label_avoid():
    action, _g, _t = apply_action_label(
        _result(aggregate_score=-0.50),
        pillars_all_zero=False,
        adjusted_aggregate=-0.50,
        sector="Healthcare",
    )
    assert action == "AVOID"


def test_apply_action_label_insufficient_data():
    action, _g, _t = apply_action_label(
        _result(aggregate_score=0.55),
        pillars_all_zero=True,  # all pillars zero
        adjusted_aggregate=0.55,
        sector="Tech",
    )
    assert action == "INSUFFICIENT DATA"


def test_fscore_gate_caps_buy_to_neutral(monkeypatch):
    """F-score <= 3 with sufficient coverage downgrades BUY/SB to NEUTRAL."""
    import config
    monkeypatch.setattr(config, "F_SCORE_GATE_ENABLED", True, raising=False)
    action, _g, _t = apply_action_label(
        _result(aggregate_score=0.55, f_score=2, f_score_coverage=0.9),
        pillars_all_zero=False,
        adjusted_aggregate=0.55,
    )
    assert action == "NEUTRAL"


def test_fscore_gate_skipped_when_low_coverage(monkeypatch):
    """F-score gate should not fire when coverage < 6/9."""
    import config
    monkeypatch.setattr(config, "F_SCORE_GATE_ENABLED", True, raising=False)
    action, _g, _t = apply_action_label(
        _result(aggregate_score=0.55, f_score=2, f_score_coverage=0.3),
        pillars_all_zero=False,
        adjusted_aggregate=0.55,
    )
    assert action == "STRONG BUY"  # gate did not fire — coverage too low


# ---------------------------------------------------------------------------
# compute_strong_buy_scorecard
# ---------------------------------------------------------------------------

def _scored_candidate(ticker, **kwargs):
    base = {
        "aggregate_score": 0.0,
        "f_score": None,
        "f_score_coverage": None,
        "gpa": None,
        "meta_success_prob": None,
        "institutional_prior_percentile": None,
        "momentum_factor_score": None,
        "quality_factor_score": None,
        "value_factor_score": None,
        "price_vs_sma200_stretch": None,
    }
    base.update(kwargs)
    base["ticker"] = ticker
    return SimpleNamespace(**base)


def test_scorecard_high_input_produces_high_score():
    """A candidate with all positive inputs scores well above the cohort mean."""
    cohort = [
        _scored_candidate("LOW", aggregate_score=-0.5, f_score=3),
        _scored_candidate("MID", aggregate_score=0.0, f_score=5),
        _scored_candidate(
            "HIGH",
            aggregate_score=0.6,
            f_score=8,
            f_score_coverage=0.9,
            gpa=0.30,
            meta_success_prob=0.80,
            institutional_prior_percentile=0.95,
            momentum_factor_score=0.6,
            quality_factor_score=0.7,
            value_factor_score=0.4,
        ),
    ]
    out = compute_strong_buy_scorecard(cohort)
    assert out["HIGH"] > out["MID"] > out["LOW"]
    assert out["HIGH"] > 0.5  # well above cohort mean


def test_scorecard_stretch_penalty():
    """Overextended candidates lose score via the stretch penalty."""
    base_args = dict(
        aggregate_score=0.5,
        f_score=8,
        f_score_coverage=0.9,
        gpa=0.30,
    )
    cohort = [
        _scored_candidate("CALM", price_vs_sma200_stretch=0.10, **base_args),
        _scored_candidate("STRETCHED", price_vs_sma200_stretch=0.80, **base_args),
    ]
    out = compute_strong_buy_scorecard(cohort)
    assert out["CALM"] > out["STRETCHED"]


def test_scorecard_persists_sb_score_attr():
    """compute_strong_buy_scorecard sets c.sb_score on each candidate."""
    cohort = [
        _scored_candidate("A", aggregate_score=0.1),
        _scored_candidate("B", aggregate_score=0.5),
    ]
    compute_strong_buy_scorecard(cohort)
    assert hasattr(cohort[0], "sb_score")
    assert hasattr(cohort[1], "sb_score")
    assert cohort[1].sb_score > cohort[0].sb_score


def test_scorecard_handles_empty_cohort():
    out = compute_strong_buy_scorecard([])
    assert out == {}


# ---------------------------------------------------------------------------
# backfill_panel_actions — DB roundtrip
# ---------------------------------------------------------------------------

def _seed_replay_row(conn, **overrides):
    """Insert a minimal replay_pit_v1 row for backfill tests."""
    payload = {
        "run_date": "2025-01-15T10:00:00",
        "ticker": "TST",
        "name": "Test Co",
        "source": "replay_pit_v1",
        "signal_price": 100.0,
        "action": "BUY",
        "aggregate_score": 0.55,
        "technical_score": 0.3,
        "fundamental_score": 0.4,
        "sentiment_score": 0.1,
        "forecast_score": 0.2,
        "f_score": 7,
        "f_score_coverage": 0.9,
        "gpa": 0.30,
        "ev_ebit": 12.0,
        "fcf_yield": 0.05,
        "revenue_growth": 0.10,
        "price_vs_sma200_stretch": 0.10,
        "sector": "Technology",
        "evaluated_90d": 1,
        "return_90d": 8.0,
    }
    payload.update(overrides)
    cols = ", ".join(payload.keys())
    placeholders = ", ".join("?" for _ in payload)
    conn.execute(
        f"INSERT INTO signal_backtest ({cols}) VALUES ({placeholders})",
        tuple(payload.values()),
    )
    conn.commit()


def test_backfill_promotes_strong_buy_zone(tmp_path, monkeypatch):
    """Replay row with agg >= 0.40 + clean gates is upgraded BUY -> STRONG BUY."""
    monkeypatch.setattr(paper_trading, "DB_PATH", tmp_path / "paper_trading.db")
    init_backtest_db()

    with paper_trading._connect() as conn:
        _seed_replay_row(conn, ticker="UPGRADE_ME", aggregate_score=0.55, action="BUY")
        _seed_replay_row(conn, ticker="STAY_BUY", aggregate_score=0.30, action="BUY")
        _seed_replay_row(conn, ticker="STAY_NEUT", aggregate_score=0.10, action="NEUTRAL")

    summary = backfill_panel_actions(dry_run=False)
    assert summary["sources"] == ["replay_pit_v1"]
    assert summary["updated"] >= 1
    assert summary["by_action"].get("STRONG BUY", 0) >= 1

    with paper_trading._connect() as conn:
        rows = {
            row["ticker"]: row["action"]
            for row in conn.execute(
                "SELECT ticker, action FROM signal_backtest"
            ).fetchall()
        }
    assert rows["UPGRADE_ME"] == "STRONG BUY"
    # Upgrade-only: BUY stays BUY when score doesn't qualify
    assert rows["STAY_BUY"] == "BUY"
    assert rows["STAY_NEUT"] == "NEUTRAL"


def test_backfill_does_not_demote_existing_labels(tmp_path, monkeypatch):
    """Upgrade-only mode should never replace an existing label with a lower one."""
    monkeypatch.setattr(paper_trading, "DB_PATH", tmp_path / "paper_trading.db")
    init_backtest_db()

    with paper_trading._connect() as conn:
        # A row labelled STRONG BUY but with score that no longer qualifies under
        # F-score gate (F=2).  Upgrade-only must not demote it.
        _seed_replay_row(
            conn, ticker="KEEP_SB",
            aggregate_score=0.55, action="STRONG BUY",
            f_score=2, f_score_coverage=0.9,
        )

    backfill_panel_actions(dry_run=False)
    with paper_trading._connect() as conn:
        action = conn.execute(
            "SELECT action FROM signal_backtest WHERE ticker='KEEP_SB'"
        ).fetchone()[0]
    assert action == "STRONG BUY"  # not demoted


# ---------------------------------------------------------------------------
# record_borderline_signals — active labelling
# ---------------------------------------------------------------------------

def _candidate_with_sb(ticker, sb_score, sector="Technology", action="BUY"):
    return SimpleNamespace(
        ticker=ticker,
        name=ticker,
        current_price=100.0,
        sector=sector,
        exchange="NYSE",
        action=action,
        sb_score=sb_score,
        aggregate_score=0.20,
        technical_score=0.1,
        fundamental_score=0.2,
        sentiment_score=0.1,
        forecast_score=0.1,
        momentum_score=0.5,
        f_score=6,
        f_score_coverage=0.8,
        gpa=0.15,
        gpa_score=0.5,
        ev_ebit=14.0,
        ev_ebit_score=0.4,
        fcf_yield=0.04,
        revenue_growth=0.06,
        price_vs_sma200_stretch=0.05,
    )


def test_record_borderline_only_writes_in_band(tmp_path, monkeypatch):
    """Only candidates with sb_score in [low, high) are recorded."""
    monkeypatch.setattr(paper_trading, "DB_PATH", tmp_path / "paper_trading.db")
    init_backtest_db()

    cohort = [
        _candidate_with_sb("LOW", 0.30),         # below band -> skip
        _candidate_with_sb("MID_A", 0.75),       # in band -> record
        _candidate_with_sb("MID_B", 0.85),       # in band -> record
        _candidate_with_sb("HIGH", 1.20),        # above band -> skip
        _candidate_with_sb("STRONG", 0.85, action="STRONG BUY"),  # excluded by action
    ]
    n = record_borderline_signals(cohort, max_records=25)
    assert n == 2

    with paper_trading._connect() as conn:
        tickers = sorted(
            row[0] for row in conn.execute(
                "SELECT ticker FROM signal_backtest WHERE source='active_label'"
            ).fetchall()
        )
    assert tickers == ["MID_A", "MID_B"]


def test_record_borderline_dedupes_same_day(tmp_path, monkeypatch):
    """Calling twice on the same day must not double-insert."""
    monkeypatch.setattr(paper_trading, "DB_PATH", tmp_path / "paper_trading.db")
    init_backtest_db()
    cohort = [_candidate_with_sb("MID", 0.80)]
    assert record_borderline_signals(cohort, max_records=25) == 1
    assert record_borderline_signals(cohort, max_records=25) == 0  # deduped


def test_record_borderline_respects_max_records(tmp_path, monkeypatch):
    """Cap honoured even when many candidates qualify."""
    monkeypatch.setattr(paper_trading, "DB_PATH", tmp_path / "paper_trading.db")
    init_backtest_db()

    # 50 borderline candidates spread across 5 sectors -> cap at 10
    cohort = []
    for i in range(50):
        sector = f"Sector{i % 5}"
        cohort.append(_candidate_with_sb(f"T{i:03d}", 0.75 + (i / 200), sector=sector))

    n = record_borderline_signals(cohort, max_records=10)
    assert n == 10

    # And distributed across sectors (no single sector dominates)
    with paper_trading._connect() as conn:
        sectors = [
            row[0] for row in conn.execute(
                "SELECT sector FROM signal_backtest WHERE source='active_label'"
            ).fetchall()
        ]
    assert len(set(sectors)) >= 3  # at least 3 sectors represented


# ---------------------------------------------------------------------------
# _blend_multi_horizon_ic
# ---------------------------------------------------------------------------

def _seed_pillar_effectiveness(conn, source, horizon, pillar, ic, n, regime=None):
    conn.execute(
        """INSERT INTO pillar_effectiveness
           (updated_at, source, pillar, horizon, information_coefficient,
            hit_rate, avg_return_high, avg_return_low, sample_size, regime)
           VALUES (?, ?, ?, ?, ?, 0.55, 5.0, -3.0, ?, ?)""",
        ("2026-05-06T10:00:00", source, pillar, horizon, ic, n, regime),
    )


def test_multi_horizon_blend_weights_short_horizons_more(tmp_path, monkeypatch):
    """5d/10d/30d should dominate the blended IC over 60d/90d (per default weights)."""
    monkeypatch.setattr(paper_trading, "DB_PATH", tmp_path / "paper_trading.db")
    init_backtest_db()

    with paper_trading._connect() as conn:
        # technical: high IC on short, zero on long -> blend should be positive
        _seed_pillar_effectiveness(conn, "all", "5d", "technical", 0.10, 1000)
        _seed_pillar_effectiveness(conn, "all", "10d", "technical", 0.10, 1000)
        _seed_pillar_effectiveness(conn, "all", "30d", "technical", 0.10, 1000)
        _seed_pillar_effectiveness(conn, "all", "60d", "technical", 0.0, 1000)
        _seed_pillar_effectiveness(conn, "all", "90d", "technical", 0.0, 1000)
        # fundamental: zero on short, high on long -> blend should be lower
        _seed_pillar_effectiveness(conn, "all", "5d", "fundamental", 0.0, 1000)
        _seed_pillar_effectiveness(conn, "all", "10d", "fundamental", 0.0, 1000)
        _seed_pillar_effectiveness(conn, "all", "30d", "fundamental", 0.0, 1000)
        _seed_pillar_effectiveness(conn, "all", "60d", "fundamental", 0.10, 1000)
        _seed_pillar_effectiveness(conn, "all", "90d", "fundamental", 0.10, 1000)
        conn.commit()

    blend = _blend_multi_horizon_ic("all")
    assert blend is not None
    rows, min_n = blend
    by_pillar = {r["pillar"]: r["information_coefficient"] for r in rows}
    # Default weights: 5d/10d/30d sum to 0.80, 60d/90d sum to 0.20
    # technical IC = 0.80*0.10 = 0.08;  fundamental = 0.20*0.10 = 0.02
    assert by_pillar["technical"] == pytest.approx(0.08, abs=0.001)
    assert by_pillar["fundamental"] == pytest.approx(0.02, abs=0.001)


def test_multi_horizon_blend_filters_regime_null(tmp_path, monkeypatch):
    """Regime-conditional rows must be ignored — only pooled (regime IS NULL) used."""
    monkeypatch.setattr(paper_trading, "DB_PATH", tmp_path / "paper_trading.db")
    init_backtest_db()

    with paper_trading._connect() as conn:
        # Pooled: 0.10
        _seed_pillar_effectiveness(conn, "all", "30d", "technical", 0.10, 1000)
        # Regime BEAR with very different IC
        _seed_pillar_effectiveness(conn, "all", "30d", "technical", -0.50, 100, regime="BEAR")
        # Other horizons (pooled) for full coverage
        for hz in ("5d", "10d", "60d", "90d"):
            _seed_pillar_effectiveness(conn, "all", hz, "technical", 0.10, 1000)
        conn.commit()

    blend = _blend_multi_horizon_ic("all")
    assert blend is not None
    rows, _ = blend
    by_pillar = {r["pillar"]: r["information_coefficient"] for r in rows}
    # Should reflect pooled IC (0.10), NOT mixed with BEAR (-0.50)
    assert by_pillar["technical"] == pytest.approx(0.10, abs=0.001)
