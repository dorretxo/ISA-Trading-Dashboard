"""Multi-factor scoring engine and recommendation logic."""

import concurrent.futures
import logging
import threading
import time

import config
from engine import technical, fundamental, sentiment, stops
from engine.fscore_utils import is_f_score_actionable, normalize_f_score_coverage
from engine.forecasting import forecast_dual_horizon
from engine.regime import get_regime_adjusted_weights
from utils.data_fetch import get_current_price, get_daily_change
from utils.safe_numeric import safe_float

_logger = logging.getLogger(__name__)

# Per-component timeout (seconds) — prevents any single sub-analysis from blocking
_WEIGHT_CACHE_LOCK = threading.Lock()
_WEIGHT_CACHE: dict[tuple[str, str], tuple[dict[str, float] | None, float]] = {}


def _component_timeout() -> int:
    """Return the current per-component scoring timeout in seconds."""
    try:
        return max(1, int(getattr(config, "SCORING_COMPONENT_TIMEOUT", 45)))
    except Exception:
        return 45


def _get_scoring_weights(source: str = "portfolio", horizon: str = "90d") -> dict[str, float]:
    """Return scoring weights with a short process-local cache."""
    ttl = float(getattr(config, "SCORING_ADAPTIVE_WEIGHT_CACHE_TTL", 900))
    key = (source, horizon)
    now = time.time()
    with _WEIGHT_CACHE_LOCK:
        cached = _WEIGHT_CACHE.get(key)
        if cached and now - cached[1] < ttl and cached[0] is not None:
            return dict(cached[0])

    adjusted_weights = None
    try:
        from engine.discovery_backtest import get_adaptive_weights
        adjusted_weights = get_adaptive_weights(source=source, horizon=horizon)
    except Exception:
        adjusted_weights = None

    if adjusted_weights is None:
        try:
            adjusted_weights = get_regime_adjusted_weights(config.WEIGHTS)
        except Exception:
            adjusted_weights = dict(config.WEIGHTS)

    pillars = ("technical", "fundamental", "sentiment", "forecast")
    normalized = {
        p: float((adjusted_weights or {}).get(p, config.WEIGHTS.get(p, 0.0)) or 0.0)
        for p in pillars
    }
    total = sum(max(0.0, v) for v in normalized.values())
    if total <= 0:
        normalized = dict(config.WEIGHTS)
    else:
        normalized = {k: max(0.0, v) / total for k, v in normalized.items()}

    with _WEIGHT_CACHE_LOCK:
        _WEIGHT_CACHE[key] = (dict(normalized), now)
    return normalized


def clear_runtime_caches() -> None:
    """Clear process-local scoring caches."""
    with _WEIGHT_CACHE_LOCK:
        _WEIGHT_CACHE.clear()


def _run_component(func, *args, component_name: str = "", timeout: int = 0, **kwargs):
    """Run a scoring component with timeout protection and error isolation.

    Uses multiprocessing-based ProcessPoolExecutor to get true preemptive timeout,
    falling back to ThreadPoolExecutor if process-based execution fails (e.g. pickling issues).
    Returns (result, elapsed_seconds, error_string_or_None).
    """
    timeout = timeout or _component_timeout()
    t0 = time.time()

    # Strategy 1: Thread-based (fast, but GIL can block timeout)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(func, *args, **kwargs)
        result = future.result(timeout=timeout)
        elapsed = time.time() - t0
        pool.shutdown(wait=False)
        return result, elapsed, None
    except concurrent.futures.TimeoutError:
        elapsed = time.time() - t0
        # Check if the thread is genuinely stuck (GIL-blocked C extension)
        # If so, the elapsed time will be very close to timeout — that's fine.
        # The key fix: don't wait for the stuck thread.
        pool.shutdown(wait=False)
        _logger.warning("[scoring] %s timed out after %.1fs (limit %ds)",
                        component_name, elapsed, timeout)
        return None, elapsed, f"timeout after {timeout}s"
    except Exception as e:
        elapsed = time.time() - t0
        _logger.warning("[scoring] %s failed after %.1fs: %s: %s",
                        component_name, elapsed, type(e).__name__, e)
        pool.shutdown(wait=False)
        return None, elapsed, str(e)


def analyse_holding(holding: dict) -> dict:
    """Run full analysis on a single holding. Returns all data needed for the UI."""
    ticker = holding["ticker"]
    name = holding.get("name", ticker)
    _hold_start = time.time()
    _component_timings: dict[str, float] = {}
    _component_errors: list[str] = []
    component_timeout = _component_timeout()
    yahoo_info = holding.get("_yahoo_info") or holding.get("_stage5b_info") or None
    allow_yahoo_metadata = bool(holding.get("_allow_yahoo_metadata_network", True))

    if getattr(config, "SCORING_PARALLEL_COMPONENTS", True):
        component_jobs = {
            "technical": (technical.analyse, (ticker,), {}, {"score": 0.0, "reasons": [], "current_price": None, "rsi": None, "atr": None}),
            "fundamental": (
                fundamental.analyse,
                (ticker,),
                {"info": yahoo_info, "allow_yahoo_network": allow_yahoo_metadata},
                {"score": 0.0, "reasons": []},
            ),
            "sentiment": (
                sentiment.analyse,
                (ticker,),
                {"company_name": name},
                {"score": 0.0, "reasons": [], "sentiment_confidence": 0.0, "article_count": 0, "active_sources": 0},
            ),
        }
        component_results: dict[str, dict] = {}
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(component_jobs))
        futures = {}
        try:
            for comp_name, (func, args, kwargs, fallback) in component_jobs.items():
                futures[pool.submit(func, *args, **kwargs)] = (comp_name, time.time(), fallback)

            for future, (comp_name, started, fallback) in list(futures.items()):
                try:
                    remaining = max(0.1, component_timeout - (time.time() - started))
                    value = future.result(timeout=remaining)
                    _component_timings[comp_name] = time.time() - started
                    component_results[comp_name] = value or fallback
                except concurrent.futures.TimeoutError:
                    _component_timings[comp_name] = time.time() - started
                    _component_errors.append(f"{comp_name}: timeout after {component_timeout}s")
                    _logger.warning("[scoring] %s(%s) timed out after %.1fs (limit %ds)",
                                    comp_name, ticker, _component_timings[comp_name], component_timeout)
                    component_results[comp_name] = fallback
                except Exception as e:
                    _component_timings[comp_name] = time.time() - started
                    _component_errors.append(f"{comp_name}: {e}")
                    _logger.warning("[scoring] %s(%s) failed after %.1fs: %s: %s",
                                    comp_name, ticker, _component_timings[comp_name], type(e).__name__, e)
                    component_results[comp_name] = fallback
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        tech = component_results.get("technical") or component_jobs["technical"][3]
        fund = component_results.get("fundamental") or component_jobs["fundamental"][3]
        sent = component_results.get("sentiment") or component_jobs["sentiment"][3]
    else:
        # Run all three analysis modules with timeout protection
        tech, _t, _e = _run_component(technical.analyse, ticker, component_name=f"technical({ticker})")
        _component_timings["technical"] = _t
        if tech is None:
            _component_errors.append(f"technical: {_e}")
            tech = {"score": 0.0, "reasons": [], "current_price": None, "rsi": None, "atr": None}

        fund, _t, _e = _run_component(
            fundamental.analyse,
            ticker,
            component_name=f"fundamental({ticker})",
            info=yahoo_info,
            allow_yahoo_network=allow_yahoo_metadata,
        )
        _component_timings["fundamental"] = _t
        if fund is None:
            _component_errors.append(f"fundamental: {_e}")
            fund = {"score": 0.0, "reasons": []}

        sent, _t, _e = _run_component(
            sentiment.analyse, ticker, component_name=f"sentiment({ticker})",
            company_name=name,
        )
        _component_timings["sentiment"] = _t
        if sent is None:
            _component_errors.append(f"sentiment: {_e}")
            sent = {"score": 0.0, "reasons": [], "sentiment_confidence": 0.0,
                    "article_count": 0, "active_sources": 0}

    if _component_errors:
        _logger.info("[scoring] %s: %d component errors: %s (timings: %s)",
                     ticker, len(_component_errors),
                     "; ".join(_component_errors),
                     {k: f"{v:.1f}s" for k, v in _component_timings.items()})

    transcript_text = (
        holding.get("transcript_text")
        or holding.get("earnings_transcript")
        or holding.get("qa_transcript")
    )
    qa_sentiment_score = None
    if transcript_text:
        try:
            qa_sentiment_score = sentiment.score_qa_sentiment(transcript_text)
        except Exception:
            qa_sentiment_score = None

    # Get price data — sanitise to prevent NaN propagation
    current_price = safe_float(tech.get("current_price")) or safe_float(get_current_price(ticker))
    daily_change = safe_float(get_daily_change(ticker))

    # Calculate stop-loss (support confluence model) and take-profit
    stop = stops.calculate_stop_loss(ticker, tech.get("atr"), current_price,
                                     sma_200=tech.get("sma_200"),
                                     sma_50=tech.get("sma_50"),
                                     bb_lower=tech.get("bb_lower"))
    target = stops.calculate_take_profit(ticker, current_price, stop["stop_loss"])

    # Run dual-horizon MoE price forecast and convert to -1..+1 scores
    forecast_data = {}
    forecast_score = 0.0
    forecast_reasons = []
    try:
        _fc_timeout = getattr(config, "SCORING_FORECAST_TIMEOUT", 60)
        dual, _fc_t, _fc_e = _run_component(
            forecast_dual_horizon, ticker,
            component_name=f"forecast({ticker})",
            timeout=_fc_timeout,
        )
        _component_timings["forecast"] = _fc_t
        if dual is None:
            raise RuntimeError(_fc_e or "forecast returned None")
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

        # Blended forecast score: 15% short-horizon + 85% long-horizon
        # The long horizon (63 trading days ≈ 90 calendar days) aligns with
        # the 90-day holding cycle and should dominate. The 5-day forecast
        # is noise for a 3-month thesis — kept at 15% for near-term confirmation.
        score_short = max(-1.0, min(1.0, fc_short.pct_change / config.FORECAST_SCORE_SCALE))
        score_long_scale = getattr(config, "FORECAST_SCORE_SCALE_LONG", 15.0)
        score_long = (
            max(-1.0, min(1.0, fc_long.pct_change / score_long_scale))
            if fc_long is not None else score_short
        )
        forecast_score = 0.15 * score_short + 0.85 * score_long

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
    except Exception as e:
        _component_errors.append(f"forecast: {e}")
        _logger.warning("[scoring] Forecast failed for %s: %s: %s", ticker, type(e).__name__, e)
        forecast_data = {"forecast_price": None, "expected_return_90d": 0.0, "forecast_unavailable": True}

    # Get weights: prefer adaptive (backtest-driven) → regime-adjusted → config defaults
    adjusted_weights = _get_scoring_weights(source="portfolio", horizon="90d")

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
        risk_overlay = apply_risk_overlay(
            fund,
            ticker,
            allow_yahoo_earnings=allow_yahoo_metadata,
        )
        aggregate_score -= risk_overlay.parabolic_penalty
    except Exception as _ro_e:
        _logger.debug("[scoring] Risk overlay failed for %s: %s", ticker, _ro_e)

    # Asymmetric / binary outcome flag (metadata only — no score impact)
    # Detects stocks where risk profile is binary (big move in either direction),
    # e.g. geopolitical trades priced above consensus with extreme technicals.
    asymmetric_risk_flag = False
    asymmetric_risk_reason = None

    _analyst_upside = fund.get("analyst_upside")
    _rsi = tech.get("rsi")
    _is_parabolic = risk_overlay.is_parabolic if risk_overlay else False
    _short_pct = fund.get("short_pct")

    if _analyst_upside is not None and _analyst_upside < -15 and _rsi is not None and _rsi > 70:
        asymmetric_risk_flag = True
        asymmetric_risk_reason = f"Trading {abs(_analyst_upside):.0f}% above consensus at RSI {_rsi:.0f}"
    elif _is_parabolic and _short_pct is not None and _short_pct > 0.10:
        asymmetric_risk_flag = True
        asymmetric_risk_reason = f"Parabolic move with {_short_pct:.0%} short interest"

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

    # F-score hard downgrade (Piotroski 2000): names with very low F-score
    # and decent coverage are likely accounting-weak — cap at KEEP even if
    # aggregate is bullish.  Gated by `config.F_SCORE_GATE_ENABLED`.
    if getattr(config, "F_SCORE_GATE_ENABLED", False):
        _fs_raw = fund.get("f_score")
        _fs = safe_float(_fs_raw, default=None)
        _fs_cov = normalize_f_score_coverage(fund.get("f_score_coverage"))
        if (
            _fs is not None
            and is_f_score_actionable(_fs, _fs_cov, config_module=config)
            and _fs <= 3
            and action in ("STRONG BUY", "BUY")
        ):
            _logger.info("F-score gate: %s F=%s cov=%.2f — action capped at KEEP", ticker, _fs, _fs_cov)
            action = "KEEP"

    # Exit-action smoother: hysteresis + persistence on KEEP↔SELL↔STRONG SELL
    # (Constantinides 1986; Davis-Norman 1990; Wald 1947; Kaminski-Lo 2014).
    # BUY / STRONG BUY pass through unchanged.
    smoother_payload: dict = {
        "smoothed_action": action,
        "smoothing_reason": "smoother_disabled",
        "smoothing_since_date": None,
        "score_vol_30d": None,
        "persistence_m": 0,
        "persistence_n": 0,
        "smoothing_band_low": None,
        "smoothing_band_high": None,
    }
    if getattr(config, "EXIT_SMOOTHER_ENABLED", True) and action in ("KEEP", "SELL", "STRONG SELL"):
        try:
            from engine import score_history, action_smoother
            from engine.stops import _get_vix_percentile
            history = score_history.get_score_series(
                ticker,
                n_days=max(getattr(config, "EXIT_SMOOTHER_VOL_LOOKBACK", 30),
                           getattr(config, "EXIT_SMOOTHER_PERSISTENCE_N", 5)) + 5,
                source="portfolio",
            )
            score_vol = score_history.compute_score_vol(
                history,
                lookback=getattr(config, "EXIT_SMOOTHER_VOL_LOOKBACK", 30),
            )
            n_window = int(getattr(config, "EXIT_SMOOTHER_PERSISTENCE_N", 5))
            m_sell, since_sell = score_history.count_recent_action(
                history, score_history.DOWN_ACTIONS, n=n_window
            )
            m_strong, _ = score_history.count_recent_action(
                history, score_history.STRONG_DOWN_ACTIONS, n=n_window
            )
            prev_smoothed = score_history.get_prev_smoothed_action(history)

            try:
                vix_pct = float(_get_vix_percentile())
            except Exception:
                vix_pct = 50.0

            smoothed = action_smoother.smooth_exit_action(
                raw_score=aggregate_score,
                base_action=action,
                prev_smoothed_action=prev_smoothed,
                score_vol=score_vol,
                vix_pct=vix_pct,
                persistence_m_sell=m_sell,
                persistence_m_strong=m_strong,
                persistence_n=n_window,
                persistence_since=since_sell,
                cusum_urgent=False,  # exit_engine reconciler applies CUSUM override later
                cusum_score=0.0,
            )
            smoother_payload = {
                "smoothed_action": smoothed.action,
                "smoothing_reason": smoothed.reason,
                "smoothing_since_date": smoothed.since_date,
                "score_vol_30d": round(score_vol, 4) if score_vol is not None else None,
                "persistence_m": smoothed.persistence_m,
                "persistence_n": smoothed.persistence_n,
                "smoothing_band_low": smoothed.band_low,
                "smoothing_band_high": smoothed.band_high,
            }
            # Replace the action label only if the smoother changed it
            if smoothed.action != action:
                _logger.info(
                    "Exit smoother: %s %s -> %s (reason=%s, score=%.3f, sigma=%s, vix_pct=%.0f)",
                    ticker, action, smoothed.action, smoothed.reason,
                    aggregate_score,
                    f"{score_vol:.3f}" if score_vol is not None else "n/a",
                    vix_pct,
                )
                action = smoothed.action
        except Exception as _sm_e:
            _logger.debug("[scoring] Exit smoother failed for %s: %s", ticker, _sm_e)

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
        "base_action": action,
        "final_action": action,
        "aggregate_score": round(aggregate_score, 3),
        # Exit-action smoother diagnostics (read by app.py + persisted to signal_backtest)
        **smoother_payload,
        "stop_loss": stop["stop_loss"],
        "structural_stop_loss": stop["stop_loss"],
        "stop_method": stop["method"],
        "structural_stop_method": stop["method"],
        "stop_distance_pct": stop.get("stop_distance_pct"),
        "support_levels": stop.get("support_levels", {}),
        "regime_info": stop.get("regime", {}),
        "trailing_exit_stop": None,
        "trailing_exit_method": None,
        "take_profit": target["take_profit"],
        "target_method": target["method"],
        "why": why,
        # Detailed sub-scores for drill-down
        "technical_score": round(tech["score"], 3),
        "fundamental_score": round(fund["score"], 3),
        "sentiment_score": round(sent["score"], 3),
        "sentiment_confidence": sent.get("sentiment_confidence", 1.0),
        "article_count": sent.get("article_count", 0),
        "active_sources": sent.get("active_sources", 0),
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
        "quality_score_fundamental": fund.get("quality_score_fundamental"),
        "gross_profitability": fund.get("gross_profitability"),
        # Enterprise-grade factor bundle (roadmap items #1, #2)
        "enterprise_value": fund.get("enterprise_value"),
        "ev_ebit": fund.get("ev_ebit"),
        "ev_ebitda": fund.get("ev_ebitda"),
        "ebit_yield": fund.get("ebit_yield"),
        "ev_ebit_score": fund.get("ev_ebit_score"),
        "gpa": fund.get("gpa"),
        "gpa_score": fund.get("gpa_score"),
        "f_score": fund.get("f_score"),
        "f_score_gate": fund.get("f_score_gate", False),
        "f_score_score": fund.get("f_score_score"),
        "f_score_coverage": fund.get("f_score_coverage"),
        "fcf_to_assets": fund.get("fcf_to_assets"),
        "earnings_stability": fund.get("earnings_stability"),
        "eps_growth_variance_5y": fund.get("eps_growth_variance_5y"),
        "news_headlines": sent.get("headlines", []),
        "reddit_headlines": sent.get("reddit_headlines", []),
        "fmp_headlines": sent.get("fmp_headlines", []),
        "news_score": sent.get("news_score"),
        "reddit_score": sent.get("reddit_score"),
        "fmp_news_score": sent.get("fmp_news_score"),
        "qa_sentiment_score": qa_sentiment_score,
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
        # Dividend safety
        "dividend_yield": fund.get("dividend_yield"),
        "payout_ratio": fund.get("payout_ratio"),
        "ex_dividend_date": fund.get("ex_dividend_date"),
        "ex_dividend_days": fund.get("ex_dividend_days"),
        "five_year_avg_yield": fund.get("five_year_avg_yield"),
        # Balance sheet strength
        "current_ratio": fund.get("current_ratio"),
        "net_debt_ebitda": fund.get("net_debt_ebitda"),
        "cash_to_debt": fund.get("cash_to_debt"),
        "balance_sheet_grade": fund.get("balance_sheet_grade"),
        # Distress / quality-of-earnings gates (Altman 1968; Sloan 1996; CGS 2008)
        "altman_z": fund.get("altman_z"),
        "altman_zone": fund.get("altman_zone"),
        "altman_coverage": fund.get("altman_coverage", 0.0),
        "accruals_factor_score": fund.get("accruals_factor_score"),
        "investment_factor_score": fund.get("investment_factor_score"),
        # Governance red flag
        "governance_flag": fund.get("governance_flag", False),
        "governance_reasons": fund.get("governance_reasons", []),
        # Asymmetric / binary outcome flag
        "asymmetric_risk_flag": asymmetric_risk_flag,
        "asymmetric_risk_reason": asymmetric_risk_reason,
        # Placeholders: discovery candidates populate these; holdings get None
        "entry_price": None,
        "entry_method": None,
        "sizing_method": None,
        "kelly_cap_fraction": None,
        # Diagnostic metadata (for pipeline observability)
        "_scoring_time_s": round(time.time() - _hold_start, 2),
        "_component_timings": {k: round(v, 2) for k, v in _component_timings.items()},
        "_component_errors": _component_errors if _component_errors else None,
        # Raw earnings data passthrough for PEAD factor
        "_earnings_surprises": fund.get("_earnings_surprises"),
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
