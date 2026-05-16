from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from engine import historical_replay as replay
from engine.factors import compute_factor_scores_from_result


def test_pit_fundamentals_enrich_replay_factor_inputs(monkeypatch):
    monkeypatch.setattr(
        replay,
        "_PIT_STORE_CACHE",
        {
            "tickers": {
                "ABC": {
                    "2023-03-31": {
                        "_accepted_date": "2023-04-20",
                        "revenue": 900.0,
                        "net_income": 80.0,
                        "operating_cashflow": 120.0,
                        "capital_expenditure": -30.0,
                        "gross_profit": 260.0,
                        "total_assets": 950.0,
                        "total_debt": 80.0,
                        "cash": 20.0,
                        "ebit": 90.0,
                        "ebitda": 110.0,
                        "eps": 1.80,
                        "shares_outstanding": 100.0,
                    },
                    "2024-03-31": {
                        "_source": "fmp",
                        "_accepted_date": "2024-04-20",
                        "revenue": 1000.0,
                        "net_income": 100.0,
                        "operating_cashflow": 140.0,
                        "capital_expenditure": -40.0,
                        "gross_profit": 300.0,
                        "total_assets": 1000.0,
                        "total_debt": 100.0,
                        "cash": 25.0,
                        "ebit": 125.0,
                        "ebitda": 150.0,
                        "eps": 2.0,
                        "shares_outstanding": 100.0,
                    },
                }
            }
        },
    )

    features = replay._fundamental_features(
        "ABC",
        pd.Timestamp("2024-06-30"),
        signal_price=10.0,
    )

    assert features["market_cap"] == 1000.0
    assert features["pit_source"] == "fmp"
    assert features["pe_ratio"] == 5.0
    assert features["fcf_yield"] == 0.10
    assert round(features["ev_ebit"], 2) == 8.60
    assert round(features["revenue_growth"], 4) == 0.1111

    factors = compute_factor_scores_from_result({
        **features,
        "momentum_score": 0.75,
        "vol_20d": 20.0,
    })
    assert factors["value_factor_score"] > 0
    assert factors["momentum_factor_score"] == 0.5
    assert factors["volatility_factor_score"] is not None

    payload = replay._replay_factor_payload(
        "ABC",
        pd.Timestamp("2024-06-30"),
        {"signal_price": 10.0, "momentum_score": 0.75, "vol_20d": 20.0},
    )
    assert payload["pit_source"] == "fmp"
    assert payload["qmj_component_count"] >= 1
    assert payload["value_factor_score"] > 0
    assert payload["momentum_factor_score"] == 0.5


def test_fundamental_features_use_pit_ttm_quarterly_gpa(monkeypatch):
    entries = {}
    quarters = [
        ("2024-06-30", "2024-07-30", 10.0, 100.0, 5.0),
        ("2024-09-30", "2024-10-30", 20.0, 110.0, 6.0),
        ("2024-12-31", "2025-01-30", 30.0, 120.0, 7.0),
        ("2025-03-31", "2025-04-30", 40.0, 130.0, 8.0),
        ("2025-06-30", "2025-07-30", 50.0, 140.0, 9.0),
        ("2025-09-30", "2025-10-30", 60.0, 150.0, 10.0),
        ("2025-12-31", "2026-01-30", 70.0, 160.0, 11.0),
        ("2026-03-31", "2026-04-30", 80.0, 180.0, 12.0),
    ]
    for report_date, accepted, gross_profit, assets, net_income in quarters:
        entries[report_date] = {
            "_accepted_date": accepted,
            "gross_profit": gross_profit,
            "total_assets": assets,
            "net_income": net_income,
            "operating_cashflow": net_income + 2.0,
            "capital_expenditure": -1.0,
            "revenue": gross_profit * 2.0,
            "eps": 1.0,
            "shares_outstanding": 10.0,
            "ebit": net_income + 1.0,
            "ebitda": net_income + 2.0,
            "total_debt": 5.0,
            "cash": 1.0,
            "current_assets": assets * 0.5,
            "current_liabilities": assets * 0.25,
        }
    monkeypatch.setattr(replay, "_PIT_STORE_CACHE", {"tickers": {"TTM": entries}})

    features = replay._fundamental_features("TTM", pd.Timestamp("2026-05-03"), signal_price=20.0)

    expected_ttm_gpa = (80.0 + 70.0 + 60.0 + 50.0) / 180.0
    assert round(features["gpa"], 4) == round(expected_ttm_gpa, 4)
    assert features["gpa"] > 80.0 / 180.0
    assert features["ttm_source"] == "pit_quarterly"
    assert features["quality_score_fundamental"] > 0


def test_fundamental_features_reject_sparse_ttm_snapshots(monkeypatch):
    entries = {}
    for report_date, accepted, gross_profit in [
        ("2024-09-30", "2024-10-30", 20.0),
        ("2025-03-31", "2025-04-30", 30.0),
        ("2025-09-30", "2025-10-30", 40.0),
        ("2026-03-31", "2026-04-30", 50.0),
    ]:
        entries[report_date] = {
            "_accepted_date": accepted,
            "gross_profit": gross_profit,
            "total_assets": 200.0,
            "net_income": 10.0,
            "operating_cashflow": 12.0,
            "capital_expenditure": -1.0,
            "revenue": 100.0,
            "eps": 1.0,
            "shares_outstanding": 10.0,
            "ebit": 12.0,
            "ebitda": 14.0,
        }
    monkeypatch.setattr(replay, "_PIT_STORE_CACHE", {"tickers": {"SPARSE": entries}})

    features = replay._fundamental_features("SPARSE", pd.Timestamp("2026-05-03"), signal_price=20.0)

    assert round(features["gpa"], 4) == 0.25
    assert features["ttm_source"] is None


def test_fundamental_features_reject_implausible_ttm_gpa(monkeypatch):
    entries = {}
    for report_date, accepted, gross_profit in [
        ("2025-06-30", "2025-07-30", 55.0),
        ("2025-09-30", "2025-10-30", 60.0),
        ("2025-12-31", "2026-01-30", 65.0),
        ("2026-03-31", "2026-04-30", 70.0),
    ]:
        entries[report_date] = {
            "_accepted_date": accepted,
            "gross_profit": gross_profit,
            "total_assets": 100.0,
            "net_income": 10.0,
            "operating_cashflow": 12.0,
            "capital_expenditure": -1.0,
            "revenue": 100.0,
            "eps": 1.0,
            "shares_outstanding": 10.0,
            "ebit": 12.0,
            "ebitda": 14.0,
        }
    monkeypatch.setattr(replay, "_PIT_STORE_CACHE", {"tickers": {"ANNUALIZED": entries}})

    features = replay._fundamental_features("ANNUALIZED", pd.Timestamp("2026-05-03"), signal_price=20.0)

    assert round(features["gpa"], 4) == 0.7
    assert features["ttm_source"] is None


def test_replay_dates_support_fresh_cadences():
    assert [d.date().isoformat() for d in replay.replay_dates("2026-05-01", "2026-05-03", "latest")] == [
        "2026-05-03"
    ]
    assert [d.date().isoformat() for d in replay.replay_dates("2026-05-01", "2026-05-08", "weekly")] == [
        "2026-05-01",
        "2026-05-08",
    ]
    assert [d.date().isoformat() for d in replay.replay_dates("2026-05-01", "2026-05-05", "daily")] == [
        "2026-05-01",
        "2026-05-04",
        "2026-05-05",
    ]


def test_price_features_use_live_decimal_units():
    idx = pd.bdate_range("2025-01-01", periods=230)
    close = pd.Series([100.0 + i * 0.1 for i in range(len(idx))], index=idx)
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=idx,
    )

    features = replay._price_features(frame, idx[-1])

    assert features is not None
    assert 0 < abs(features["return_10d_prior"]) < 1
    assert 0 < abs(features["return_30d_prior"]) < 1
    assert 0 <= features["vol_20d"] < 2
    assert "rsi" in features
    assert "bb_pct" in features


def test_cross_sectional_momentum_scores_rank_replay_cohort():
    features = {
        "AAA": {
            "ret_12m1m": 0.30,
            "return_90d_prior": 0.20,
            "return_30d_prior": 0.10,
            "return_10d_prior": 0.04,
            "volume_ratio": 1.5,
            "pct_from_high_252d": 0.98,
            "momentum_score": 0.5,
        },
        "BBB": {
            "ret_12m1m": -0.10,
            "return_90d_prior": -0.05,
            "return_30d_prior": -0.02,
            "return_10d_prior": -0.01,
            "volume_ratio": 0.8,
            "pct_from_high_252d": 0.70,
            "momentum_score": 0.5,
        },
    }

    replay._apply_cross_sectional_momentum_scores(features)

    assert features["AAA"]["momentum_score"] > features["BBB"]["momentum_score"]
    assert 0 <= features["BBB"]["momentum_score"] <= 1
    assert 0 <= features["AAA"]["momentum_score"] <= 1


def _ready_replay_row(**overrides):
    row = {
        "signal_price": 100.0,
        "technical_score": 0.8,
        "momentum_score": 0.75,
        "aggregate_score": 0.55,
        "action": "STRONG BUY",
        "sector": "Technology",
        "industry": "Software",
        "atr": 2.0,
        "rsi": 55.0,
        "sma_200": 90.0,
        "price_vs_sma200_stretch": 0.11,
        "return_10d_prior": 0.02,
        "return_30d_prior": 0.06,
        "return_90d_prior": 0.18,
        "vol_20d": 0.20,
        "quality_factor_score": 0.9,
        "quality_score_fundamental": 0.85,
        "value_factor_score": 0.7,
        "momentum_factor_score": 0.8,
        "volatility_factor_score": 0.6,
        "qmj_factor_score": 0.9,
        "gpa": 0.8,
        "gpa_score": 0.9,
        "gross_profitability": 0.8,
        "fcf_to_assets": 0.12,
        "fcf_yield": 0.10,
        "revenue_growth": 0.20,
        "f_score": 9,
        "f_score_coverage": 1.0,
        "ev_ebit": 16.0,
        "ev_ebit_score": 0.7,
        "pe_ratio": 20.0,
        "roe": 0.25,
        "net_debt_ebitda": 0.2,
        "current_ratio": 2.0,
        "cash_to_debt": 2.0,
        "market_cap": 10_000_000_000.0,
        "avg_dollar_volume": 50_000_000.0,
        "stop_loss": 95.0,
        "take_profit": 112.0,
    }
    row.update(overrides)
    return row


def test_replay_readiness_backfill_populates_pit_safe_contract_fields():
    rows = {
        "GOOD": _ready_replay_row(),
        "MID": _ready_replay_row(
            quality_factor_score=0.1, qmj_factor_score=0.1, gpa=0.2,
            gpa_score=0.1, f_score=5, ev_ebit=28.0, ev_ebit_score=0.0,
            fcf_yield=0.02, revenue_growth=0.01, aggregate_score=0.25, action="BUY",
        ),
        "WEAK": _ready_replay_row(
            quality_factor_score=-0.5, qmj_factor_score=-0.5, gpa=0.05,
            gpa_score=-0.5, f_score=3, ev_ebit=60.0, ev_ebit_score=-1.0,
            fcf_yield=-0.05, revenue_growth=-0.20, aggregate_score=-0.1, action="NEUTRAL",
        ),
    }

    replay._apply_replay_readiness_fields(rows)

    good = rows["GOOD"]
    assert good["entry_stance"] == "Ready"
    assert good["r_r_ratio"] > 2.0
    assert good["action_gate_ceiling"] == "STRONG BUY"
    assert good["ready_contract_status"] == "PASS"
    assert good["strong_buy_eligible"] == 1
    assert good["institutional_prior_percentile"] >= 0.9
    assert good["institutional_prior_components"] is not None


def test_replay_readiness_backfill_blocks_overbought_rows():
    rows = {
        "HOT": _ready_replay_row(rsi=82.0, price_vs_sma200_stretch=0.70),
        "GOOD": _ready_replay_row(quality_factor_score=0.7, qmj_factor_score=0.7),
        "WEAK": _ready_replay_row(quality_factor_score=-0.5, qmj_factor_score=-0.5, f_score=3),
    }

    replay._apply_replay_readiness_fields(rows)

    hot = rows["HOT"]
    assert hot["entry_stance"] == "Watch Only"
    assert hot["ready_contract_status"] == "FAIL"
    assert hot["strong_buy_eligible"] == 0
    assert "action gate ceiling" in (hot["strong_buy_blockers"] or "")


def test_replay_readiness_preserves_precomputed_execution_fields():
    rows = {
        "LOWRR": _ready_replay_row(
            entry_price=99.0,
            fill_probability=0.8,
            stop_loss=90.0,
            take_profit=103.0,
        ),
        "GOOD": _ready_replay_row(quality_factor_score=0.7, qmj_factor_score=0.7),
        "WEAK": _ready_replay_row(quality_factor_score=-0.5, qmj_factor_score=-0.5, f_score=3),
    }

    replay._apply_replay_readiness_fields(rows)

    lowrr = rows["LOWRR"]
    assert lowrr["entry_price"] == 99.0
    assert lowrr["fill_probability"] == 0.8
    assert lowrr["r_r_ratio"] == pytest.approx(4.0 / 9.0)
    assert lowrr["ready_contract_status"] == "FAIL"
    assert "R/R below" in (lowrr["strong_buy_blockers"] or "")


def test_insert_replay_row_can_refresh_existing_row():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE signal_backtest (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            source TEXT,
            run_date TEXT,
            signal_price REAL,
            aggregate_score REAL,
            replay_version TEXT,
            replay_asof TEXT,
            action TEXT,
            name TEXT
        )
        """
    )
    replay._SIGNAL_COLUMNS_CACHE = None

    first = replay._insert_replay_row(
        "ABC",
        pd.Timestamp("2026-05-03"),
        {"signal_price": 10.0, "aggregate_score": 0.1},
        conn=conn,
    )
    second = replay._insert_replay_row(
        "ABC",
        pd.Timestamp("2026-05-03"),
        {"signal_price": 10.0, "aggregate_score": 0.4},
        conn=conn,
        refresh_existing=True,
    )
    row = conn.execute("SELECT COUNT(*) AS n, aggregate_score FROM signal_backtest").fetchone()

    assert first == "inserted"
    assert second == "updated"
    assert row[0] == 1
    assert row[1] == 0.4
