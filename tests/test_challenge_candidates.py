import json
from pathlib import Path

import config
from engine.challenge_candidates import build_challenge_candidates


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_factor_validated_generator_excludes_korea_and_requires_multiple_groups(tmp_path, monkeypatch):
    feature_dir = tmp_path / "feature_cache"
    pit_path = feature_dir / "pit_fundamentals.json"
    _write_json(
        feature_dir / "features_2026-04-30.json",
        {
            "version": 1,
            "date": "2026-04-30",
            "features": {
                "GOOD": {
                    "ticker": "GOOD",
                    "last_price": 50,
                    "avg_dollar_volume": 50_000_000,
                    "ret_90d": 0.20,
                    "ret_30d": 0.08,
                    "ret_10d": 0.02,
                    "relative_strength": 0.05,
                    "pct_from_high_252d": 0.92,
                    "above_sma50": True,
                    "above_sma200": True,
                    "vol_20d": 0.25,
                },
                "005930.KS": {
                    "ticker": "005930.KS",
                    "last_price": 150000,
                    "avg_dollar_volume": 500_000_000,
                    "ret_90d": 0.50,
                    "ret_30d": 0.20,
                    "ret_10d": 0.08,
                    "relative_strength": 0.10,
                    "pct_from_high_252d": 0.98,
                    "above_sma50": True,
                    "above_sma200": True,
                    "vol_20d": 0.20,
                },
                "ONEGROUP": {
                    "ticker": "ONEGROUP",
                    "last_price": 40,
                    "avg_dollar_volume": 50_000_000,
                    "ret_90d": -0.20,
                    "ret_30d": -0.10,
                    "ret_10d": -0.04,
                    "relative_strength": -0.10,
                    "pct_from_high_252d": 0.50,
                    "above_sma50": False,
                    "above_sma200": False,
                    "vol_20d": 0.60,
                },
            },
        },
    )
    _write_json(
        pit_path,
        {
            "version": 1,
            "tickers": {
                "GOOD": {
                    "2026-03-31": {
                        "gross_profit": 350,
                        "total_assets": 1000,
                        "operating_cashflow": 150,
                        "capital_expenditure": -30,
                        "net_income": 100,
                        "total_debt": 150,
                        "market_cap": 1200,
                        "cash": 50,
                        "ebit": 140,
                        "revenue_growth": 0.12,
                        "earnings_growth": 0.20,
                        "trailing_pe": 12,
                    }
                },
                "005930.KS": {
                    "2026-03-31": {
                        "gross_profit": 800,
                        "total_assets": 1000,
                        "net_income": 300,
                        "market_cap": 2000,
                        "revenue_growth": 0.30,
                    }
                },
                "ONEGROUP": {
                    "2026-03-31": {
                        "gross_profit": 40,
                        "total_assets": 1000,
                        "net_income": 10,
                        "market_cap": 2000,
                    }
                },
            },
        },
    )

    monkeypatch.setattr(config, "DISCOVERY_CHALLENGE_MIN_DOLLAR_VOLUME", 1_000_000)
    monkeypatch.setattr(config, "DISCOVERY_CHALLENGE_MIN_FACTOR_GROUPS", 2)

    candidates = build_challenge_candidates(
        target_n=10,
        as_of="2026-05-01",
        feature_cache_dir=feature_dir,
        pit_path=pit_path,
        manual_overrides=[],
        include_near_miss=False,
        write_cache=False,
    )

    symbols = {candidate["symbol"] for candidate in candidates}
    assert "GOOD" in symbols
    assert "005930.KS" not in symbols
    assert "ONEGROUP" not in symbols
    good = next(candidate for candidate in candidates if candidate["symbol"] == "GOOD")
    assert good["challenge_source"] == "factor_validated"
    assert len(good["factor_groups"]) >= 2


def test_manual_overrides_expire_and_are_capped(tmp_path, monkeypatch):
    feature_dir = tmp_path / "feature_cache"
    _write_json(feature_dir / "features_2026-04-30.json", {"features": {}})
    _write_json(feature_dir / "pit_fundamentals.json", {"tickers": {}})

    monkeypatch.setattr(config, "DISCOVERY_CHALLENGE_MANUAL_PCT", 0.5)
    monkeypatch.setattr(config, "DISCOVERY_CHALLENGE_NEAR_MISS_PCT", 0.0)
    monkeypatch.setattr(config, "DISCOVERY_CHALLENGE_FACTOR_PCT", 0.0)
    monkeypatch.setattr(config, "DISCOVERY_CHALLENGE_MAX_PER_COUNTRY", 2)

    candidates = build_challenge_candidates(
        target_n=4,
        as_of="2026-05-01",
        feature_cache_dir=feature_dir,
        pit_path=feature_dir / "pit_fundamentals.json",
        include_near_miss=False,
        write_cache=False,
        manual_overrides=[
            {"symbol": "A", "country": "US", "reason": "fresh", "expires_at": "2026-05-30"},
            {"symbol": "B", "country": "US", "reason": "fresh", "expires_at": "2026-05-30"},
            {"symbol": "C", "country": "US", "reason": "over country cap", "expires_at": "2026-05-30"},
            {"symbol": "OLD", "country": "US", "reason": "expired", "expires_at": "2026-04-01"},
            {"symbol": "005930.KS", "country": "KR", "reason": "excluded", "expires_at": "2026-05-30"},
        ],
    )

    symbols = [candidate["symbol"] for candidate in candidates]
    assert symbols == ["A", "B"]
    assert all(candidate["challenge_source"] == "manual_override" for candidate in candidates)
