from __future__ import annotations

from types import SimpleNamespace

import config
from engine import paper_trading
from engine.value_cap_shadow import (
    annotate_value_cap_shadow,
    build_value_cap_shadow_report,
    record_value_cap_shadow_events,
)


def _candidate(ticker: str, **overrides):
    base = {
        "ticker": ticker,
        "action": "BUY",
        "sector": "Technology",
        "threshold_profile": "conservative",
        "final_rank": 0.5,
        "aggregate_score": 0.25,
        "sb_score": 1.0,
        "ready_contract_core_status": "PASS",
        "ready_contract_status": "PASS",
        "entry_stance": "Ready",
        "ev_ebit": 10.0,
        "ev_sales": 3.0,
        "pe_forward": 18.0,
        "pe_ratio": 18.0,
        "eps_growth_3y_cagr": 0.10,
        "revenue_growth": 0.10,
        "f_score": 8,
        "f_score_coverage": 1.0,
        "altman_z": 4.0,
        "beneish_m": -3.0,
        "accruals_factor_score": 0.2,
        "investment_factor_score": 0.2,
        "net_debt_ebitda": 0.5,
        "gpa_score": 0.70,
        "qmj_factor_score": 0.70,
        "roic": 0.18,
        "wacc": 0.08,
        "op_margin_yoy_delta": 0.01,
        "rsi": 55.0,
        "price_vs_sma200_stretch": 0.10,
        "realized_vol_pctile": 0.50,
        "qmj_component_count": 3,
        "pit_source": "fmp_full",
        "institutional_prior_percentile": 0.95,
        "institutional_prior_confidence": 0.80,
        "institutional_prior_coverage": 0.60,
        "institutional_prior_coverage_source": "live_prior_pipeline",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_value_cap_shadow_attributes_valuation_and_prior_knobs(monkeypatch):
    monkeypatch.setattr(config, "STRONG_BUY_VALUATION_QUALITY_OVERRIDE_ENABLED", False)
    monkeypatch.setattr(config, "STRONG_BUY_VALUATION_OVERRIDE_QMJ_FLOOR", 0.80)
    monkeypatch.setattr(config, "STRONG_BUY_VALUATION_OVERRIDE_MIN_F_SCORE", 7)
    monkeypatch.setattr(config, "STRONG_BUY_VALUATION_OVERRIDE_MAX_EV_EBIT", 35.0)
    monkeypatch.setattr(config, "STRONG_BUY_VALUATION_OVERRIDE_MAX_PE_FORWARD", 40.0)
    monkeypatch.setattr(config, "INSTITUTIONAL_PRIOR_MIN_COVERAGE", 0.45)
    monkeypatch.setattr(config, "INSTITUTIONAL_PRIOR_MIN_COVERAGE_SHADOW", 0.42)

    candidates = [
        _candidate(
            "VAL",
            ev_ebit=29.0,
            pe_forward=33.0,
            pe_ratio=33.0,
            revenue_growth=0.20,
            qmj_factor_score=0.95,
            sb_score=1.2,
        ),
        _candidate(
            "PRIOR",
            ev_ebit=10.0,
            pe_forward=18.0,
            qmj_factor_score=0.80,
            institutional_prior_coverage=0.42,
            sb_score=0.9,
        ),
        _candidate(
            "LOW",
            ev_ebit=10.0,
            pe_forward=18.0,
            qmj_factor_score=0.10,
            institutional_prior_percentile=0.40,
            institutional_prior_coverage=0.30,
            sb_score=0.1,
        ),
    ]

    rows = {row["ticker"]: row for row in annotate_value_cap_shadow(candidates, config_module=config)}

    assert rows["VAL"]["current"]["ceiling"] == "BUY"
    assert rows["VAL"]["valuation_a"]["ceiling"] == "STRONG BUY"
    assert rows["VAL"]["valuation_would_clear_gate"] is True
    assert rows["VAL"]["primary_bucket"] == "valuation_gate_gain"

    assert rows["PRIOR"]["current_prior_pass"] is False
    assert rows["PRIOR"]["coverage_b_prior_pass"] is True
    assert rows["PRIOR"]["prior_coverage_gain"] is True
    assert rows["PRIOR"]["primary_bucket"] == "prior_coverage_gain"


def test_value_cap_shadow_report_is_shadow_only(monkeypatch):
    monkeypatch.setattr(config, "STRONG_BUY_VALUATION_QUALITY_OVERRIDE_ENABLED", False)
    candidates = [
        _candidate("VAL", ev_ebit=29.0, pe_forward=33.0, pe_ratio=33.0, revenue_growth=0.20, qmj_factor_score=0.95),
        _candidate("MID", qmj_factor_score=0.50),
        _candidate("LOW", qmj_factor_score=0.10),
    ]

    report = build_value_cap_shadow_report(candidates, config_module=config, include_historical=False)

    assert report["mode"] == "shadow_only"
    assert report["live_impact"] is False
    assert report["knobs"]["valuation_a"]["live_enabled"] is False
    assert report["summary"]["valuation_gate_gains"] == 1
    assert report["changed_candidates"][0]["ticker"] == "VAL"


def test_value_cap_shadow_report_schema_is_stable(monkeypatch):
    monkeypatch.setattr(config, "STRONG_BUY_VALUATION_QUALITY_OVERRIDE_ENABLED", False)
    report = build_value_cap_shadow_report([_candidate("VAL")], config_module=config, include_historical=False)

    assert set(report["knobs"]) == {"valuation_a", "prior_b"}
    assert "summary" in report
    assert "changed_candidates" in report
    assert "valuation_a_candidates" in report
    assert "valuation_blocked_candidates" in report
    assert "prior_b_candidates" in report
    assert "fresh_qmj_health" in report
    for key in (
        "valuation_rank_gains",
        "valuation_gate_gains",
        "prior_coverage_gains",
        "combined_gains",
    ):
        assert isinstance(report["summary"][key], int)
        assert report["summary"][key] >= 0


def test_value_cap_shadow_prior_rows_include_coverage_attribution(monkeypatch):
    monkeypatch.setattr(config, "INSTITUTIONAL_PRIOR_MIN_COVERAGE", 0.45)
    monkeypatch.setattr(config, "INSTITUTIONAL_PRIOR_MIN_COVERAGE_SHADOW", 0.42)
    candidate = _candidate(
        "PRIOR",
        institutional_prior_coverage=0.42,
        institutional_prior_coverage_source="backfilled_from_features",
    )

    row = annotate_value_cap_shadow([candidate], config_module=config)[0]

    assert row["prior_coverage_gain"] is True
    assert row["institutional_prior_coverage"] == 0.42
    assert row["institutional_prior_coverage_source"] == "backfilled_from_features"


def test_value_cap_shadow_prior_rows_fallback_legacy_coverage_source(monkeypatch):
    monkeypatch.setattr(config, "INSTITUTIONAL_PRIOR_MIN_COVERAGE", 0.45)
    monkeypatch.setattr(config, "INSTITUTIONAL_PRIOR_MIN_COVERAGE_SHADOW", 0.42)
    candidate = _candidate(
        "PRIOR",
        institutional_prior_coverage=0.42,
        institutional_prior_coverage_source=None,
    )

    row = build_value_cap_shadow_report([candidate], config_module=config, include_historical=False)["prior_b_candidates"][0]

    assert row["prior_coverage_gain"] is True
    assert row["institutional_prior_coverage_source"] == "legacy_cached_prior_pipeline"


def test_value_cap_shadow_recomputes_missing_serialized_qmj(monkeypatch):
    monkeypatch.setattr(config, "STRONG_BUY_VALUATION_QUALITY_OVERRIDE_ENABLED", False)
    monkeypatch.setattr(config, "STRONG_BUY_VALUATION_OVERRIDE_MAX_EV_EBIT", 35.0)
    monkeypatch.setattr(config, "STRONG_BUY_VALUATION_OVERRIDE_MAX_PE_FORWARD", 40.0)
    row = _candidate(
        "SER",
        ev_ebit=29.0,
        pe_forward=33.0,
        pe_ratio=33.0,
        revenue_growth=0.20,
        qmj_factor_score=None,
        gpa_score=1.0,
        f_score_score=1.0,
        f_score=9,
    ).__dict__

    ann = annotate_value_cap_shadow([row], config_module=config)[0]

    assert ann["qmj_shadow_recomputed"] is True
    assert ann["qmj_factor_score"] is not None


def test_value_cap_shadow_report_tracks_fresh_qmj_health(monkeypatch):
    monkeypatch.setattr(config, "STRONG_BUY_VALUATION_QUALITY_OVERRIDE_ENABLED", False)
    candidates = [
        _candidate("US", qmj_factor_score=0.8, exchange="NASDAQ"),
        _candidate("NONUS.L", qmj_factor_score=None, qmj_component_count=0, exchange="LSE"),
    ]

    report = build_value_cap_shadow_report(candidates, config_module=config, include_historical=False)

    assert report["fresh_qmj_health"]["total"] == 2
    assert report["fresh_qmj_health"]["qmj_null"] == 1
    assert report["fresh_qmj_health"]["by_region"]["non_us"]["qmj_null"] == 1


def test_value_cap_shadow_event_ledger_records_firings(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_trading, "DB_PATH", tmp_path / "paper_trading.db")
    monkeypatch.setattr(config, "STRONG_BUY_VALUATION_QUALITY_OVERRIDE_ENABLED", False)
    report = build_value_cap_shadow_report(
        [
            _candidate(
                "VAL",
                ev_ebit=29.0,
                pe_forward=33.0,
                pe_ratio=33.0,
                revenue_growth=0.20,
                qmj_factor_score=0.95,
                entry_price=20.0,
                stop_loss=18.0,
                take_profit=25.0,
                r_r_ratio=2.5,
            ),
            _candidate("MID", qmj_factor_score=0.50),
            _candidate("LOW", qmj_factor_score=0.10),
        ],
        config_module=config,
        include_historical=False,
    )

    summary = record_value_cap_shadow_events(report, config_module=config)

    assert summary["available"] is True
    assert summary["events_recorded_or_updated"] == 1
    with paper_trading._connect() as conn:
        row = conn.execute(
            "SELECT ticker, primary_bucket, entry_price, stop_loss, take_profit "
            "FROM value_cap_shadow_events"
        ).fetchone()
    assert dict(row) == {
        "ticker": "VAL",
        "primary_bucket": "valuation_gate_gain",
        "entry_price": 20.0,
        "stop_loss": 18.0,
        "take_profit": 25.0,
    }
