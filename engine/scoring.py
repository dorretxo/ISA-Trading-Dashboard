"""Multi-factor scoring engine and recommendation logic."""

import config
from engine import technical, fundamental, sentiment, stops
from engine.forecasting import forecast_dual_horizon
from engine.regime import get_regime_adjusted_weights
from utils.data_fetch import get_current_price, get_daily_change


def analyse_holding(holding: dict) -> dict:
    """Run full analysis on a single holding. Returns all data needed for the UI."""
    ticker = holding["ticker"]
    name = holding.get("name", ticker)

    # Run all three analysis modules
    tech = technical.analyse(ticker)
    fund = fundamental.analyse(ticker)
    sent = sentiment.analyse(ticker, company_name=name)

    # Get price data
    current_price = tech.get("current_price") or get_current_price(ticker)
    daily_change = get_daily_change(ticker)

    # Calculate stop-loss and take-profit (pass SMA-200 for support level)
    stop = stops.calculate_stop_loss(ticker, tech.get("atr"), current_price,
                                     sma_200=tech.get("sma_200"))
    target = stops.calculate_take_profit(ticker, current_price, stop["stop_loss"])

    # Run dual-horizon MoE price forecast and convert to -1..+1 scores
    forecast_data = {}
    forecast_score = 0.0
    forecast_reasons = []
    try:
        dual = forecast_dual_horizon(ticker)
        fc_short = dual["short"]
        fc_long = dual["long"]

        # Short-horizon forecast data (primary display)
        forecast_data = {
            "forecast_price": fc_short.predicted_price,
            "forecast_low": fc_short.confidence_low,
            "forecast_high": fc_short.confidence_high,
            "forecast_direction": fc_short.direction,
            "forecast_pct_change": fc_short.pct_change,
            "forecast_horizon": fc_short.horizon_days,
            "forecast_expert_weights": fc_short.expert_weights,
            "forecast_ensemble_mae": fc_short.ensemble_mae,
            "forecast_expert_maes": fc_short.expert_maes,
            "forecast_experts": [
                {"name": e.name, "price": round(e.predicted_price, 4),
                 "low": round(e.confidence_low, 4), "high": round(e.confidence_high, 4)}
                for e in fc_short.expert_forecasts
            ],
        }

        # Long-horizon forecast data
        if fc_long is not None:
            forecast_data.update({
                "forecast_price_long": fc_long.predicted_price,
                "forecast_low_long": fc_long.confidence_low,
                "forecast_high_long": fc_long.confidence_high,
                "forecast_direction_long": fc_long.direction,
                "forecast_pct_change_long": fc_long.pct_change,
                "forecast_horizon_long": fc_long.horizon_days,
                "forecast_expert_weights_long": fc_long.expert_weights,
                "forecast_ensemble_mae_long": fc_long.ensemble_mae,
                "forecast_experts_long": [
                    {"name": e.name, "price": round(e.predicted_price, 4),
                     "low": round(e.confidence_low, 4), "high": round(e.confidence_high, 4)}
                    for e in fc_long.expert_forecasts
                ],
            })

        # Blended forecast score: 30% short-horizon + 70% long-horizon
        # The long horizon (63 trading days ≈ 90 calendar days) aligns with
        # the 90-day holding cycle — it should dominate the signal.
        score_short = max(-1.0, min(1.0, fc_short.pct_change / config.FORECAST_SCORE_SCALE))
        score_long_scale = getattr(config, "FORECAST_SCORE_SCALE_LONG", 15.0)
        score_long = (
            max(-1.0, min(1.0, fc_long.pct_change / score_long_scale))
            if fc_long is not None else score_short
        )
        forecast_score = 0.30 * score_short + 0.70 * score_long

        # Unified 90-day expected return (used by optimizer and discovery)
        # Primary: long-horizon MoE forecast (already ≈90 calendar days)
        # Secondary: short-horizon extrapolated to 90d as sanity anchor
        if fc_long is not None:
            forecast_return_90d = fc_long.pct_change / 100.0
        else:
            # Fallback: extrapolate 5-day to 63 trading days
            forecast_return_90d = (fc_short.pct_change / 100.0) * (63 / max(fc_short.horizon_days, 1))
        forecast_data["expected_return_90d"] = forecast_return_90d

        # Reason text
        if fc_short.pct_change > 3.0:
            forecast_reasons.append(f"MoE predicts +{fc_short.pct_change:.1f}% ({fc_short.horizon_days}d)")
        elif fc_short.pct_change < -3.0:
            forecast_reasons.append(f"MoE predicts {fc_short.pct_change:.1f}% ({fc_short.horizon_days}d)")
        else:
            forecast_reasons.append(f"MoE predicts {fc_short.pct_change:+.1f}% ({fc_short.horizon_days}d)")

        if fc_long is not None:
            if fc_long.pct_change > 5.0:
                forecast_reasons.append(f"90d outlook +{fc_long.pct_change:.1f}%")
            elif fc_long.pct_change < -5.0:
                forecast_reasons.append(f"90d outlook {fc_long.pct_change:.1f}%")
            else:
                forecast_reasons.append(f"90d outlook {fc_long.pct_change:+.1f}%")
    except Exception:
        forecast_data = {"forecast_price": None, "expected_return_90d": 0.0}

    # Get weights: prefer adaptive (backtest-driven) → regime-adjusted → config defaults
    adjusted_weights = None
    try:
        from engine.discovery_backtest import get_adaptive_weights
        adjusted_weights = get_adaptive_weights(source="portfolio", horizon="90d")
    except Exception:
        pass
    if adjusted_weights is None:
        try:
            adjusted_weights = get_regime_adjusted_weights(config.WEIGHTS)
        except Exception:
            adjusted_weights = dict(config.WEIGHTS)

    # Weighted aggregate score (4 pillars with regime tilt)
    aggregate_score = (
        tech["score"] * adjusted_weights["technical"]
        + fund["score"] * adjusted_weights["fundamental"]
        + sent["score"] * adjusted_weights["sentiment"]
        + forecast_score * adjusted_weights["forecast"]
    )

    # Risk overlay — parabolic penalty + metadata flags
    risk_overlay = None
    try:
        from engine.risk_overlay import apply_risk_overlay
        risk_overlay = apply_risk_overlay(fund, ticker)
        aggregate_score -= risk_overlay.parabolic_penalty
    except Exception:
        pass

    # Determine action
    if aggregate_score >= config.SCORE_STRONG_BUY_THRESHOLD:
        action = "STRONG BUY"
    elif aggregate_score >= config.SCORE_BUY_THRESHOLD:
        action = "BUY"
    elif aggregate_score >= config.SCORE_KEEP_THRESHOLD:
        action = "KEEP"
    elif aggregate_score >= config.SCORE_SELL_THRESHOLD:
        action = "SELL"
    else:
        action = "STRONG SELL"

    # Build the "Why?" summary
    all_reasons = tech["reasons"] + fund["reasons"] + sent["reasons"] + forecast_reasons
    # Pick the most impactful reasons (up to 4)
    why = " + ".join(all_reasons[:4]) if all_reasons else "No significant signals"

    result = {
        "ticker": ticker,
        "name": name,
        "current_price": current_price,
        "daily_change_pct": daily_change,
        "avg_buy_price": holding["avg_buy_price"],
        "quantity": holding["quantity"],
        "currency": holding.get("currency", "GBP"),
        "action": action,
        "aggregate_score": round(aggregate_score, 3),
        "stop_loss": stop["stop_loss"],
        "stop_method": stop["method"],
        "take_profit": target["take_profit"],
        "target_method": target["method"],
        "why": why,
        # Detailed sub-scores for drill-down
        "technical_score": round(tech["score"], 3),
        "fundamental_score": round(fund["score"], 3),
        "sentiment_score": round(sent["score"], 3),
        "forecast_score": round(forecast_score, 3),
        # Technical details
        "rsi": tech.get("rsi"),
        "bb_pct": tech.get("bb_pct"),
        "bb_upper": tech.get("bb_upper"),
        "bb_lower": tech.get("bb_lower"),
        "stoch_k": tech.get("stoch_k"),
        "stoch_d": tech.get("stoch_d"),
        "obv_trend": tech.get("obv_trend"),
        "obv_divergence": tech.get("obv_divergence"),
        "adx": tech.get("adx"),
        "williams_r": tech.get("williams_r"),
        "atr": tech.get("atr"),
        "sma_50": tech.get("sma_50"),
        "sma_200": tech.get("sma_200"),
        "macd_signal": tech.get("macd_signal"),
        # Fundamental details
        "pe_ratio": fund.get("pe_ratio"),
        "debt_to_equity": fund.get("debt_to_equity"),
        "short_pct": fund.get("short_pct"),
        "short_ratio": fund.get("short_ratio"),
        "inst_ownership": fund.get("inst_ownership"),
        "insider_ownership": fund.get("insider_ownership"),
        "insider_buys": fund.get("insider_buys", 0),
        "insider_sells": fund.get("insider_sells", 0),
        "insider_net": fund.get("insider_net", ""),
        "insider_transactions": fund.get("insider_transactions", []),
        "analyst_target": fund.get("analyst_target"),
        "analyst_upside": fund.get("analyst_upside"),
        "analyst_rec": fund.get("analyst_rec"),
        "num_analysts": fund.get("num_analysts"),
        "revenue_growth": fund.get("revenue_growth"),
        "profit_margin": fund.get("profit_margin"),
        "roe": fund.get("roe"),
        "fcf_yield": fund.get("fcf_yield"),
        "news_headlines": sent.get("headlines", []),
        "reddit_headlines": sent.get("reddit_headlines", []),
        "fmp_headlines": sent.get("fmp_headlines", []),
        "news_score": sent.get("news_score"),
        "reddit_score": sent.get("reddit_score"),
        "fmp_news_score": sent.get("fmp_news_score"),
        # FMP fundamental enhancements
        "fmp_available": fund.get("fmp_available", False),
        "earnings_beat_rate": fund.get("earnings_beat_rate"),
        "quarterly_trend": fund.get("quarterly_trend"),
        "estimate_revision": fund.get("estimate_revision"),
        "peg_ratio": fund.get("peg_ratio"),
        "sector_pe": fund.get("sector_pe"),
        "pe_vs_sector": fund.get("pe_vs_sector"),
        "recent_upgrades": fund.get("recent_upgrades", 0),
        "recent_downgrades": fund.get("recent_downgrades", 0),
        "next_earnings_date": fund.get("next_earnings_date"),
        "earnings_proximity_days": fund.get("earnings_proximity_days"),
        # Risk overlay flags
        "parabolic_penalty": risk_overlay.parabolic_penalty if risk_overlay else 0.0,
        "is_parabolic": risk_overlay.is_parabolic if risk_overlay else False,
        "earnings_near": risk_overlay.earnings_near if risk_overlay else False,
        "earnings_imminent": risk_overlay.earnings_imminent if risk_overlay else False,
        "cap_tier": risk_overlay.cap_tier if risk_overlay else "unknown",
        "confidence_discount": risk_overlay.confidence_discount if risk_overlay else 1.0,
        "max_weight_scale": risk_overlay.max_weight_scale if risk_overlay else 1.0,
    }
    result.update(forecast_data)
    return result


def analyse_portfolio(holdings: list[dict]) -> tuple[list[dict], dict, list[dict]]:
    """Run analysis on all holdings with portfolio-level risk assessment.

    Returns (results, risk_data, position_weights) where:
        results: per-holding analysis list
        risk_data: correlation matrix, sector concentration, and warnings
        position_weights: inverse-volatility suggested allocations
    """
    results = []
    for holding in holdings:
        result = analyse_holding(holding)
        results.append(result)

    # Portfolio-level risk analysis
    try:
        from engine.portfolio_risk import assess_portfolio_risk
        risk_data = assess_portfolio_risk(results, holdings)
    except Exception:
        risk_data = {
            "correlation_matrix": None,
            "high_correlations": [],
            "sector_weights": {},
            "concentration_warnings": [],
            "risk_score": 0.0,
        }

    # Position sizing via inverse-volatility weighting
    try:
        from engine.position_sizing import calculate_inverse_vol_weights
        position_weights = calculate_inverse_vol_weights(holdings, results)
    except Exception:
        position_weights = []

    return results, risk_data, position_weights
