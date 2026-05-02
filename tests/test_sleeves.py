from engine.sleeves import compute_sleeve_scores


def test_sleeve_scores_are_bounded_and_coverage_aware():
    candidates = [
        {
            "symbol": "AAA",
            "sector": "Healthcare",
            "_momentum_score": 0.9,
            "_pit_quality_score": 0.8,
            "_pit_value_score": 0.4,
            "_pit_low_risk_score": 0.7,
            "_ret_90d": 0.18,
            "_ret_30d": 0.08,
            "_ret_10d": 0.02,
            "_relative_strength": 0.05,
            "_above_sma50": True,
            "_above_sma200": True,
            "_vol_20d": 0.14,
        },
        {
            "symbol": "BBB",
            "sector": "Healthcare",
            "_momentum_score": 0.2,
            "_pit_quality_score": -0.4,
            "_pit_value_score": -0.2,
            "_pit_low_risk_score": 0.2,
            "_ret_90d": -0.12,
            "_ret_30d": -0.04,
            "_ret_10d": -0.01,
            "_relative_strength": -0.03,
            "_above_sma50": False,
            "_above_sma200": False,
            "_vol_20d": 0.45,
        },
    ]

    scores = compute_sleeve_scores(candidates, feature_cache={})

    assert set(scores) == {"AAA", "BBB"}
    assert 0.0 <= scores["AAA"]["composite"] <= 1.0
    assert 0.0 <= scores["BBB"]["composite"] <= 1.0
    assert scores["AAA"]["composite"] > scores["BBB"]["composite"]
    assert "pead" in scores["AAA"]["low_coverage_sleeves"]
