from engine.institutional_prior import score_universe


def test_institutional_prior_rewards_quality_value_momentum_intersection():
    rows = []
    for i in range(14):
        rows.append({
            "ticker": f"BAD{i}",
            "sector": "Technology",
            "quality_factor_score": -0.4,
            "value_factor_score": -0.2,
            "momentum_factor_score": -0.3,
            "volatility_factor_score": -0.2,
            "f_score": 3,
            "f_score_coverage": 1.0,
            "gpa": 0.05,
            "gpa_score": -0.6,
            "avg_dollar_volume": 5_000_000,
        })
    rows.append({
        "ticker": "GOOD",
        "sector": "Technology",
        "quality_factor_score": 0.9,
        "quality_score_fundamental": 0.8,
        "value_factor_score": 0.7,
        "ev_ebit_score": 0.8,
        "momentum_factor_score": 0.6,
        "volatility_factor_score": 0.5,
        "f_score": 8,
        "f_score_coverage": 1.0,
        "gpa": 0.42,
        "gpa_score": 0.7,
        "avg_dollar_volume": 50_000_000,
        "return_90d_prior": 0.25,
    })

    priors = score_universe(rows)

    assert priors["GOOD"].percentile >= 0.90
    assert priors["GOOD"].confidence >= 0.55
    assert priors["GOOD"].passes_strong_buy_bar
    assert priors["GOOD"].components["qmj"] is not None
    assert priors["GOOD"].components["bab"] is not None
    assert priors["GOOD"].components["turnover"] is not None


def test_institutional_prior_blocks_thin_low_quality_candidate():
    priors = score_universe([
        {
            "ticker": "THIN",
            "sector": "Basic Materials",
            "momentum_factor_score": 0.9,
            "return_90d_prior": 0.80,
            "price_vs_sma200_stretch": 0.70,
        }
    ])

    prior = priors["THIN"]

    assert not prior.passes_strong_buy_bar
    assert prior.coverage < 0.45
