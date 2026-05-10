"""Shared factor math for quality, confidence adjustment, and ML features.

This module centralises the prompt-driven factor logic so discovery,
fundamental scoring, and the ML ranker all consume the same definitions.

Institutional factor coverage (Jensen, Kelly & Pedersen 2023):
- Quality (QMJ): gross profitability, ROE, FCF/assets, earnings stability
- Value: P/E, PEG, FCF yield, P/B, P/S
- Momentum: 12-1 month, residual momentum (Blitz, Huij & Martens 2011)
- Volatility: realized vol, idiosyncratic vol (Ang et al. 2006)
- Investment: asset growth (Cooper, Gulen & Schill 2008)
- Accruals: (net income - OCF) / total assets (Sloan 1996)
- Leverage: D/E, interest coverage (Bhandari 1988)
- Reversal: 1-month contrarian signal (Jegadeesh 1990)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

import config


def cross_sectional_zscore(values: list[float] | np.ndarray, clip: float = 3.0) -> np.ndarray:
    """Return cross-sectional z-scores clipped to a stable range."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(arr)
    z = (arr - mean) / std
    return np.clip(z, -clip, clip)


def _clip_score(value: float, scale: float = 2.5) -> float:
    """Map a z-like input to [-1, 1] with a conservative clip."""
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value / scale, -1.0, 1.0))


def _first_valid(*values: Any) -> float | None:
    """Return the first numeric, finite value from a list of candidates."""
    for value in values:
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(num):
            return num
    return None


def _latest_total_assets(info: dict, balance_sheet_statements: list[dict] | None = None) -> float | None:
    """Find the most recent total-assets figure from the available sources."""
    total_assets = _first_valid(
        info.get("totalAssets"),
        info.get("totalAssetsRaw"),
    )
    if total_assets and total_assets > 0:
        return total_assets

    if balance_sheet_statements:
        for statement in balance_sheet_statements:
            total_assets = _first_valid(statement.get("totalAssets"))
            if total_assets and total_assets > 0:
                return total_assets
    return None


def _extract_eps_series(annual_income_statements: list[dict] | None) -> list[float]:
    """Extract up to 5 annual EPS observations in chronological order."""
    if not annual_income_statements:
        return []

    annual_eps: list[tuple[str, float]] = []
    for statement in annual_income_statements:
        eps = _first_valid(statement.get("eps"), statement.get("epsdiluted"))
        date_value = str(statement.get("date", ""))
        if eps is None:
            continue
        annual_eps.append((date_value, eps))

    if not annual_eps:
        return []

    annual_eps.sort(key=lambda item: item[0])
    values = [eps for _, eps in annual_eps[-5:]]
    return values


def compute_eps_growth_variance(annual_income_statements: list[dict] | None) -> tuple[float | None, float | None]:
    """Return the 5-year EPS growth variance and a higher-is-better stability score."""
    eps_series = _extract_eps_series(annual_income_statements)
    if len(eps_series) < 4:
        return None, None

    growth_rates: list[float] = []
    for prev, curr in zip(eps_series[:-1], eps_series[1:]):
        denom = abs(prev)
        if denom < 1e-6:
            continue
        growth_rates.append((curr - prev) / denom)

    if len(growth_rates) < 3:
        return None, None

    variance = float(np.var(growth_rates, ddof=1))
    # Lower variance is better; 0.15 is a steady grower, 0.60 is highly unstable.
    stability_z = (0.30 - variance) / 0.15
    return variance, _clip_score(stability_z, scale=2.0)


def compute_fundamental_quality_metrics(
    info: dict,
    annual_income_statements: list[dict] | None = None,
    balance_sheet_statements: list[dict] | None = None,
) -> dict:
    """Compute a QMJ-style quality composite from fundamental inputs.

    Components follow the prompt directly:
    - Gross Profitability = Gross Profit / Total Assets
    - Return on Equity
    - Free Cash Flow / Total Assets
    - Earnings Stability = inverse of 5-year EPS growth variance
    """
    total_assets = _latest_total_assets(info, balance_sheet_statements)
    gross_profit = _first_valid(info.get("grossProfits"), info.get("grossProfit"))
    roe = _first_valid(info.get("returnOnEquity"))
    free_cashflow = _first_valid(info.get("freeCashflow"))
    eps_growth_variance_5y, earnings_stability = compute_eps_growth_variance(annual_income_statements)

    gross_profitability = None
    fcf_to_assets = None
    component_scores: dict[str, float] = {}
    component_weights: dict[str, float] = {
        "gross_profitability": 0.35,
        "roe": 0.25,
        "fcf_to_assets": 0.25,
        "earnings_stability": 0.15,
    }

    if total_assets and total_assets > 0 and gross_profit is not None:
        gross_profitability = gross_profit / total_assets
        gp_z = (gross_profitability - 0.35) / 0.20
        component_scores["gross_profitability"] = _clip_score(gp_z)

    if roe is not None:
        roe_z = (roe - 0.12) / 0.15
        component_scores["roe"] = _clip_score(roe_z)

    if total_assets and total_assets > 0 and free_cashflow is not None:
        fcf_to_assets = free_cashflow / total_assets
        fcf_z = (fcf_to_assets - 0.05) / 0.08
        component_scores["fcf_to_assets"] = _clip_score(fcf_z)

    if earnings_stability is not None:
        component_scores["earnings_stability"] = earnings_stability

    if component_scores:
        total_weight = sum(component_weights[key] for key in component_scores)
        quality_score = sum(
            component_scores[key] * component_weights[key]
            for key in component_scores
        ) / total_weight
    else:
        quality_score = None

    return {
        "quality_score": None if quality_score is None else float(np.clip(quality_score, -1.0, 1.0)),
        "gross_profitability": gross_profitability,
        "roe": roe,
        "fcf_to_assets": fcf_to_assets,
        "eps_growth_variance_5y": eps_growth_variance_5y,
        "earnings_stability": earnings_stability,
        "component_scores": component_scores,
        "component_count": len(component_scores),
    }


def compute_investment_factor(info: dict, balance_sheet_statements: list[dict] | None = None) -> float | None:
    """Asset growth factor — Cooper, Gulen & Schill (2008).

    Firms with high asset growth underperform by 7-10% annually.
    Returns a score in [-1, 1] where negative = high growth (bearish).
    """
    if balance_sheet_statements and len(balance_sheet_statements) >= 2:
        ta_curr = _first_valid(balance_sheet_statements[0].get("totalAssets"))
        ta_prev = _first_valid(balance_sheet_statements[1].get("totalAssets"))
        if ta_curr and ta_prev and ta_prev > 0:
            asset_growth = (ta_curr / ta_prev) - 1.0
            # Low asset growth is bullish: median ~8%, >30% is aggressive
            return _clip_score((0.08 - asset_growth) / 0.15, scale=2.0)

    # Fallback: yfinance totalAssets vs prior (if only current available, skip)
    total_assets = _first_valid(info.get("totalAssets"))
    if total_assets is None:
        return None
    return None  # Need two periods for growth


def compute_accruals_factor(info: dict) -> float | None:
    """Accruals anomaly — Sloan (1996).

    Accruals = (net income - operating cash flow) / total assets.
    High accruals → earnings manipulation → underperformance (7-10% spread).
    Returns score in [-1, 1] where negative = high accruals (bearish).
    """
    net_income = _first_valid(info.get("netIncomeToCommon"), info.get("netIncome"))
    ocf = _first_valid(info.get("operatingCashflow"), info.get("totalCashFromOperatingActivities"))
    total_assets = _first_valid(info.get("totalAssets"))

    if net_income is None or ocf is None or not total_assets or total_assets <= 0:
        return None

    accruals = (net_income - ocf) / total_assets
    # Low accruals is bullish: median ~-0.03, positive accruals are red flag
    return _clip_score((-accruals - 0.03) / 0.06, scale=2.0)


def compute_leverage_factor(info: dict) -> float | None:
    """Leverage/solvency factor — Bhandari (1988), Ohlson (1980).

    Combines D/E and interest coverage for financial health signal.
    Returns score in [-1, 1] where positive = low leverage (bullish).
    """
    de_ratio = _first_valid(info.get("debtToEquity"))
    if de_ratio is not None and abs(de_ratio) > 10:
        de_ratio = de_ratio / 100.0  # yfinance sometimes returns as percentage

    interest_coverage = _first_valid(info.get("interestCoverage"))

    components: list[float] = []

    if de_ratio is not None:
        # Low D/E is bullish: median ~0.8, >2.0 is highly leveraged
        components.append(_clip_score((0.80 - de_ratio) / 0.60, scale=2.0))

    if interest_coverage is not None and interest_coverage > 0:
        # High coverage is bullish: >5x is comfortable, <2x is danger
        components.append(_clip_score((interest_coverage - 5.0) / 5.0, scale=2.0))

    if not components:
        return None
    return float(np.clip(np.mean(components), -1.0, 1.0))


def compute_reversal_factor(result: dict) -> float | None:
    """Conditional short-term reversal — Jegadeesh (1990), Huang et al. (2022).

    1-month return as contrarian signal, activated conditionally:
    - Full signal when VIX percentile > 60 (high-vol regimes amplify reversal)
    - Attenuated in calm markets (reversal is ~3x weaker in low-vol regimes)
    - Weighted by distance from SMA-50 normalized by realized vol

    Returns score in [-1, 1] where positive = contrarian buy signal.
    """
    ret_1m = _first_valid(
        result.get("return_1m"),
        result.get("_ret_21d"),
        result.get("_ret_30d"),
        result.get("return_30d_prior"),
    )
    if ret_1m is None:
        return None
    # Normalize: yfinance returns can be % or decimal
    if abs(ret_1m) > 2.0:
        ret_1m = ret_1m / 100.0

    # Base reversal signal (inverted: negative recent return = bullish)
    base_reversal = _clip_score(-ret_1m / 0.10, scale=2.0)

    # --- Regime conditioning (Huang, Liu, Rhee & Zhang 2022) ---
    vix_pct = _first_valid(result.get("vix_percentile"), result.get("_vix_percentile"))
    rsi = _first_valid(result.get("rsi"), result.get("_rsi"))

    # Regime multiplier: full signal in high-vol, attenuated in low-vol
    if vix_pct is not None:
        if vix_pct > 0.60:
            regime_mult = 1.0  # High-vol: reversal is strong
        elif vix_pct > 0.40:
            regime_mult = 0.6  # Normal: moderate signal
        else:
            regime_mult = 0.3  # Calm: reversal is weak
    else:
        regime_mult = 0.6  # Unknown regime: moderate default

    # RSI confirmation: oversold stocks have stronger reversal potential
    if rsi is not None and rsi < 30 and base_reversal > 0:
        regime_mult = min(1.0, regime_mult * 1.3)  # Boost for oversold + reversal
    elif rsi is not None and rsi > 70 and base_reversal < 0:
        regime_mult = min(1.0, regime_mult * 1.3)  # Boost for overbought + mean-revert down

    # Distance from SMA-50 normalization (deeper oversold = stronger signal)
    sma50_dist = _first_valid(result.get("_sma50_distance_pct"))
    vol_20d = _first_valid(result.get("vol_20d"), result.get("_vol_20d"))
    if sma50_dist is not None and vol_20d is not None and vol_20d > 0.001:
        # Normalize distance by vol to get "how many vol units from SMA-50"
        vol_normalized_dist = abs(sma50_dist) / vol_20d
        # Deeper deviation from SMA-50 strengthens reversal (capped at 1.5x)
        depth_mult = min(1.5, 1.0 + vol_normalized_dist * 0.1)
        regime_mult *= depth_mult

    return float(np.clip(base_reversal * regime_mult, -1.0, 1.0))


def compute_extended_value_score(info: dict, result: dict) -> dict[str, float | None]:
    """Extended value factor — Fama & French (1993), Lakonishok et al. (1994).

    Adds P/B and P/S to the existing P/E, PEG, FCF yield value composite.
    Returns individual scores and a blended extended_value_score.
    """
    pb = _first_valid(info.get("priceToBook"), result.get("_price_to_book"))
    ps = _first_valid(info.get("priceToSalesTrailing12Months"), result.get("_price_to_sales"))

    pb_score = None
    ps_score = None

    if pb is not None and pb > 0:
        # Low P/B is value: median ~3.0, <1.5 is cheap
        pb_score = _clip_score((3.0 - pb) / 3.0, scale=2.0)

    if ps is not None and ps > 0:
        # Low P/S is value: median ~2.5, <1.0 is cheap
        ps_score = _clip_score((2.5 - ps) / 2.5, scale=2.0)

    return {
        "pb_score": pb_score,
        "ps_score": ps_score,
    }


def compute_residual_momentum(result: dict) -> float | None:
    """Residual momentum — Blitz, Huij & Martens (2011).

    Proxy: raw momentum minus beta-implied market return.
    More persistent and less crash-prone than raw momentum.
    Returns score in [-1, 1].
    """
    ret_90d = _first_valid(result.get("return_90d_prior"), result.get("_ret_90d"))
    beta = _first_valid(result.get("beta_90d"), result.get("_beta"), result.get("beta"))

    if ret_90d is None:
        return None
    if abs(ret_90d) > 2.0:
        ret_90d = ret_90d / 100.0

    # Market proxy: assume ~3% per quarter (12% annualized)
    market_return_90d = 0.03
    if beta is not None:
        expected_return = beta * market_return_90d
    else:
        expected_return = market_return_90d

    residual = ret_90d - expected_return
    return _clip_score(residual / 0.15, scale=2.0)


def compute_idiosyncratic_vol(result: dict) -> float | None:
    """Idiosyncratic volatility — Ang, Hodrick, Xing & Zhang (2006).

    Proxy: total vol minus beta-implied systematic vol.
    Low idio-vol premium: stocks with low unexplained volatility outperform.
    Returns score in [-1, 1] where positive = low idio-vol (bullish).
    """
    vol_20d = _first_valid(result.get("vol_20d"), result.get("_vol_20d"))
    beta = _first_valid(result.get("beta_90d"), result.get("_beta"), result.get("beta"))

    if vol_20d is None:
        return None
    if abs(vol_20d) > 2.0:
        vol_20d = vol_20d / 100.0

    # Proxy for idiosyncratic vol: total_vol * sqrt(1 - R²)
    # Approximate R² from beta: R² ≈ beta² * market_vol² / total_vol²
    market_vol = 0.16  # ~16% annualized market vol
    if beta is not None:
        systematic_vol = abs(beta) * market_vol / (252 ** 0.5) * (20 ** 0.5)  # 20-day scale
        idio_vol = max(0, vol_20d - systematic_vol)
    else:
        idio_vol = vol_20d * 0.7  # Assume ~70% is idiosyncratic

    # Low idio-vol is bullish
    return _clip_score((0.02 - idio_vol) / 0.015, scale=2.0)


def compute_downside_vol(returns, threshold: float = 0.0) -> float | None:
    """Annualised downside semi-volatility from daily returns.

    This is the "bad volatility" component missing from plain low-beta screens:
    a stock can have low total variance but still gap down in ways that matter
    for a ready-to-buy screener.
    """
    if returns is None:
        return None
    try:
        arr = np.asarray(list(returns), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size < 5:
        return None
    downside = arr[arr < threshold] - threshold
    if downside.size < 2:
        return 0.0
    return float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(252))


def compute_max_drawdown(prices, lookback_days: int = 252) -> float | None:
    """Peak-to-trough drawdown over the trailing window as a positive fraction."""
    if prices is None:
        return None
    try:
        arr = np.asarray(list(prices), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    if arr.size < 5:
        return None
    arr = arr[-int(max(5, lookback_days)):]
    running_peak = np.maximum.accumulate(arr)
    drawdowns = arr / running_peak - 1.0
    return float(abs(np.min(drawdowns)))


def compute_bab_factor(result: dict) -> float | None:
    """Betting-Against-Beta style defensive factor.

    Positive values prefer low-beta, lower-volatility stocks.  This is used as
    a risk-control sleeve rather than a standalone reason to buy.
    """
    beta = _first_valid(result.get("beta_90d"), result.get("_beta"), result.get("beta"))
    vol_score = _first_valid(result.get("volatility_factor_score"))
    vol_20d = _first_valid(result.get("vol_20d"), result.get("_vol_20d"))
    downside_vol = _first_valid(result.get("downside_vol_60d"), result.get("_downside_vol_60d"))
    if downside_vol is None:
        downside_vol = compute_downside_vol(
            result.get("returns_60d") or result.get("_returns_60d") or result.get("daily_returns_60d")
        )
    max_dd = _first_valid(result.get("max_dd_252d"), result.get("_max_dd_252d"))
    if max_dd is None:
        max_dd = compute_max_drawdown(result.get("prices_252d") or result.get("_prices_252d"))
    components: list[float] = []

    if beta is not None:
        beta = min(2.5, max(-0.5, beta))
        components.append(float(np.clip((1.0 - beta) / 0.8, -1.0, 1.0)))
    if vol_score is not None:
        components.append(float(np.clip(vol_score, -1.0, 1.0)))
    elif vol_20d is not None:
        vol = vol_20d / 100.0 if abs(vol_20d) > 2.0 else vol_20d
        components.append(float(np.clip((0.25 - vol) / 0.20, -1.0, 1.0)))
    if downside_vol is not None:
        dv = downside_vol / 100.0 if abs(downside_vol) > 2.0 else downside_vol
        components.append(float(np.clip((0.18 - dv) / 0.18, -1.0, 1.0)))
    if max_dd is not None:
        dd = abs(max_dd / 100.0 if abs(max_dd) > 2.0 else max_dd)
        components.append(float(np.clip((0.25 - dd) / 0.25, -1.0, 1.0)))

    if not components:
        return None
    return float(np.clip(np.mean(components), -1.0, 1.0))


def compute_turnover_cost_score(result: dict) -> float | None:
    """Trading-cost / churn proxy inspired by anomaly turnover evidence.

    Higher is better: liquid, larger, calmer names with less short-term chase.
    This is intentionally a small gate-like feature, not a primary alpha factor.
    """
    avg_dollar_volume = _first_valid(result.get("avg_dollar_volume"), result.get("_avg_dollar_volume"))
    market_cap = _first_valid(result.get("market_cap"), result.get("_market_cap"))
    vol_20d = _first_valid(result.get("vol_20d"), result.get("_vol_20d"))
    ret_10d = _first_valid(result.get("return_10d_prior"), result.get("_ret_10d"), result.get("ret_10d"))
    ret_90d = _first_valid(result.get("return_90d_prior"), result.get("_ret_90d"), result.get("ret_90d"))

    components: list[float] = []
    if avg_dollar_volume is not None and avg_dollar_volume > 0:
        components.append(float(np.clip((math.log10(max(avg_dollar_volume, 1.0)) - 7.0) / 2.0, -1.0, 1.0)))
    if market_cap is not None and market_cap > 0:
        components.append(float(np.clip((math.log10(max(market_cap, 1.0)) - 9.0) / 2.5, -1.0, 1.0)))
    if vol_20d is not None:
        vol = vol_20d / 100.0 if abs(vol_20d) > 2.0 else vol_20d
        components.append(float(np.clip((0.35 - vol) / 0.35, -1.0, 1.0)))
    if ret_10d is not None:
        r10 = ret_10d / 100.0 if abs(ret_10d) > 2.0 else ret_10d
        r90 = 0.0
        if ret_90d is not None:
            r90 = ret_90d / 100.0 if abs(ret_90d) > 2.0 else ret_90d
        short_chase = max(0.0, abs(r10) - abs(r90) / 3.0)
        components.append(float(np.clip(-short_chase / 0.12, -1.0, 1.0)))

    if not components:
        return None
    return float(np.clip(np.mean(components), -1.0, 1.0))


def compute_pead_factor(result: dict) -> dict[str, float | None]:
    """Post-Earnings Announcement Drift factor — Martineau (2022).

    PEAD is the strongest documented anomaly at the 60-90 day horizon.
    Combines three components:
    - SUE (Standardized Unexpected Earnings): surprise / std(surprises)
    - Analyst revision momentum: 3-month consensus change direction
    - Earnings acceleration: quarter-over-quarter beat rate trend

    Returns a dict with individual scores and blended pead_factor_score,
    each in [-1, 1] or None if insufficient data.
    """
    sue_window = getattr(config, "PEAD_SUE_WINDOW_QUARTERS", 8)

    # --- 1. SUE: Standardized Unexpected Earnings ---
    sue_score = None
    surprises_raw = result.get("_earnings_surprises") or result.get("earnings_surprises")
    if surprises_raw and isinstance(surprises_raw, list) and len(surprises_raw) >= 2:
        surprise_pcts: list[float] = []
        for entry in surprises_raw[:sue_window]:
            actual = _first_valid(entry.get("actualEarningResult"), entry.get("actual"))
            estimated = _first_valid(entry.get("estimatedEarning"), entry.get("estimated"))
            if actual is not None and estimated is not None:
                denom = max(abs(estimated), 0.01)
                surprise_pcts.append((actual - estimated) / denom)

        if len(surprise_pcts) >= 2:
            latest_surprise = surprise_pcts[0]
            surprise_std = float(np.std(surprise_pcts, ddof=1)) if len(surprise_pcts) >= 3 else 0.10
            surprise_std = max(surprise_std, 0.01)  # floor to avoid division by zero
            sue_raw = latest_surprise / surprise_std
            sue_score = _clip_score(sue_raw, scale=3.0)

    # --- 2. Analyst revision momentum ---
    revision_momentum_3m = None
    estimate_revision = result.get("estimate_revision")
    if estimate_revision is not None:
        if isinstance(estimate_revision, (int, float)):
            rev_pct = float(estimate_revision)
        else:
            try:
                rev_pct = float(str(estimate_revision).replace("%", "").replace("+", "").strip())
            except (TypeError, ValueError):
                rev_pct = None
        if rev_pct is not None:
            # ±10% revision maps to ±1.0 score
            revision_momentum_3m = _clip_score(rev_pct / 10.0, scale=1.0)

    # --- 3. Earnings acceleration (beat rate trend) ---
    earnings_accel = None
    beat_rate_str = result.get("earnings_beat_rate")
    if beat_rate_str is not None:
        if isinstance(beat_rate_str, (int, float)):
            beat_ratio = float(beat_rate_str)
        else:
            try:
                parts = str(beat_rate_str).split("/")
                if len(parts) == 2:
                    beat_ratio = float(parts[0]) / float(parts[1])
                else:
                    beat_ratio = float(beat_rate_str)
            except (TypeError, ValueError, ZeroDivisionError):
                beat_ratio = None
        if beat_ratio is not None:
            # 0.75 = 3/4 beats → positive; 0.25 = 1/4 → negative
            earnings_accel = _clip_score((beat_ratio - 0.50) / 0.25, scale=1.0)

    # --- Blend: SUE 50%, Revision 30%, Acceleration 20% ---
    components: list[tuple[float, float]] = []
    if sue_score is not None:
        components.append((sue_score, 0.50))
    if revision_momentum_3m is not None:
        components.append((revision_momentum_3m, 0.30))
    if earnings_accel is not None:
        components.append((earnings_accel, 0.20))

    pead_factor_score = None
    if components:
        total_weight = sum(w for _, w in components)
        raw = sum(s * w for s, w in components) / total_weight
        pead_factor_score = float(np.clip(raw, -1.0, 1.0))

    return {
        "pead_factor_score": pead_factor_score,
        "sue_score": sue_score,
        "revision_momentum_3m": revision_momentum_3m,
        "earnings_accel": earnings_accel,
    }


def compute_all_extended_factors(
    info: dict,
    result: dict,
    balance_sheet_statements: list[dict] | None = None,
) -> dict[str, float | None]:
    """Compute all extended institutional factors from available data.

    Returns a dict of factor scores, each in [-1, 1] or None if data insufficient.
    """
    extended_value = compute_extended_value_score(info, result)
    pead = compute_pead_factor(result)

    return {
        "investment_factor_score": compute_investment_factor(info, balance_sheet_statements),
        "accruals_factor_score": compute_accruals_factor(info),
        "leverage_factor_score": compute_leverage_factor(info),
        "reversal_factor_score": compute_reversal_factor(result),
        "residual_momentum_score": compute_residual_momentum(result),
        "idiosyncratic_vol_score": compute_idiosyncratic_vol(result),
        "pb_score": extended_value["pb_score"],
        "ps_score": extended_value["ps_score"],
        "pead_factor_score": pead["pead_factor_score"],
        "sue_score": pead["sue_score"],
        "revision_momentum_3m": pead["revision_momentum_3m"],
    }


def compute_factor_scores_from_result(result: dict) -> dict[str, float | None]:
    """Derive the factor-score bundle used by discovery ranking and ML."""
    quality_score = result.get("quality_score_fundamental")
    if quality_score is None:
        quality_score = result.get("_quality_score_fundamental")

    # Quality gets additional contribution from first-class GPA (Novy-Marx 2013)
    # and Piotroski F-score (Piotroski 2000) when available.  We blend rather
    # than replace so legacy consumers still see a continuous series.
    gpa_score = _first_valid(result.get("gpa_score"))
    f_score_score = _first_valid(result.get("f_score_score"))
    _extra_q: list[float] = []
    if quality_score is not None:
        _extra_q.append(float(np.clip(quality_score, -1.0, 1.0)))
    if gpa_score is not None:
        _extra_q.append(float(np.clip(gpa_score, -1.0, 1.0)))
    if f_score_score is not None:
        _extra_q.append(float(np.clip(f_score_score, -1.0, 1.0)))
    if _extra_q:
        quality_score = float(np.mean(_extra_q))

    value_components: list[float] = []
    pe_ratio = _first_valid(result.get("pe_ratio"))
    if pe_ratio is not None and pe_ratio > 0:
        value_components.append(float(np.clip((18.0 - pe_ratio) / 18.0, -1.0, 1.0)))

    peg_ratio = _first_valid(result.get("peg_ratio"))
    if peg_ratio is not None and peg_ratio > 0:
        value_components.append(float(np.clip((1.25 - peg_ratio) / 1.25, -1.0, 1.0)))

    fcf_yield = _first_valid(result.get("fcf_yield"))
    if fcf_yield is not None:
        value_components.append(float(np.clip((fcf_yield - 0.03) / 0.08, -1.0, 1.0)))

    # Extended value: P/B and P/S (Fama-French 1993, Lakonishok et al. 1994)
    pb_score = _first_valid(result.get("pb_score"), result.get("_pb_score"))
    if pb_score is not None:
        value_components.append(float(np.clip(pb_score, -1.0, 1.0)))
    ps_score = _first_valid(result.get("ps_score"), result.get("_ps_score"))
    if ps_score is not None:
        value_components.append(float(np.clip(ps_score, -1.0, 1.0)))

    # Enterprise-grade value anchor (Loughran & Wellman 2011; Greenblatt 2006)
    # NOTE: EV/EBIT also contributes to the fundamental *pillar* directly via
    # `fundamental.analyse` (enterprise-pillar delta).  Keep single-weight
    # here to avoid double-counting when pillar + factor-tilt both feed alpha.
    ev_ebit_score = _first_valid(result.get("ev_ebit_score"))
    if ev_ebit_score is not None:
        value_components.append(float(np.clip(ev_ebit_score, -1.0, 1.0)))

    value_score = float(np.mean(value_components)) if value_components else 0.0

    momentum_score = result.get("momentum_score")
    if momentum_score is None:
        momentum_score = result.get("_momentum_score")
    if momentum_score is not None:
        momentum_score = float(momentum_score)
        if 0.0 <= momentum_score <= 1.0:
            momentum_score = (momentum_score * 2.0) - 1.0
    else:
        ret_90d = _first_valid(result.get("return_90d_prior"), result.get("_ret_90d"))
        ret_30d = _first_valid(result.get("return_30d_prior"), result.get("_ret_30d"))
        components = []
        if ret_90d is not None:
            ret_90d = ret_90d / 100.0 if abs(ret_90d) > 1.0 else ret_90d
            components.append(float(np.clip(ret_90d / 0.25, -1.0, 1.0)))
        if ret_30d is not None:
            ret_30d = ret_30d / 100.0 if abs(ret_30d) > 1.0 else ret_30d
            components.append(float(np.clip(ret_30d / 0.15, -1.0, 1.0)))
        momentum_score = float(np.mean(components)) if components else 0.0

    volatility = _first_valid(result.get("vol_20d"), result.get("_vol_20d"))
    if volatility is not None and abs(volatility) > 2.0:
        volatility = volatility / 100.0
    volatility_score = None
    if volatility is not None and volatility >= 0:
        volatility_score = float(np.clip((0.25 - volatility) / 0.20, -1.0, 1.0))
    else:
        beta = _first_valid(result.get("beta_90d"), result.get("_beta"))
        if beta is not None:
            volatility_score = float(np.clip((1.0 - beta) / 0.8, -1.0, 1.0))

    # Extended factors (may be None if data insufficient)
    investment = _first_valid(result.get("investment_factor_score"), result.get("_investment_factor_score"))
    accruals = _first_valid(result.get("accruals_factor_score"), result.get("_accruals_factor_score"))
    leverage = _first_valid(result.get("leverage_factor_score"), result.get("_leverage_factor_score"))
    reversal = _first_valid(result.get("reversal_factor_score"), result.get("_reversal_factor_score"))
    residual_mom = _first_valid(result.get("residual_momentum_score"), result.get("_residual_momentum_score"))
    idio_vol = _first_valid(result.get("idiosyncratic_vol_score"), result.get("_idiosyncratic_vol_score"))

    if getattr(config, "QMJ_LITE_ENABLED", True):
        gp_anchor = _first_valid(
            gpa_score,
            result.get("gpa"),
            result.get("gross_profitability"),
            result.get("_gross_profitability"),
        )
        profitability = None
        if gp_anchor is not None:
            if gp_anchor == gpa_score:
                profitability = float(np.clip(gp_anchor, -1.0, 1.0))
            else:
                profitability = float(np.clip((gp_anchor - 0.30) / 0.20, -1.0, 1.0))

        safety_vals = [
            _first_valid(result.get("earnings_stability"), result.get("_earnings_stability")),
            leverage,
        ]
        safety_parts = [float(np.clip(x, -1.0, 1.0)) for x in safety_vals if x is not None]
        safety = float(np.mean(safety_parts)) if safety_parts else None

        qmj_components = [
            x for x in (
                profitability,
                None if f_score_score is None else float(np.clip(f_score_score, -1.0, 1.0)),
                safety,
            )
            if x is not None
        ]
        qmj_min_components = int(getattr(config, "QMJ_LITE_MIN_COMPONENTS", 2))
        qmj_factor_score = float(np.mean(qmj_components)) if len(qmj_components) >= qmj_min_components else None
    else:
        qmj_components = []
        for q in (
            quality_score,
            gpa_score,
            f_score_score,
            _first_valid(result.get("earnings_stability"), result.get("_earnings_stability")),
            leverage,
        ):
            if q is not None:
                qmj_components.append(float(np.clip(q, -1.0, 1.0)))
        qmj_factor_score = float(np.mean(qmj_components)) if qmj_components else None

    bab_context = dict(result)
    if volatility_score is not None:
        bab_context["volatility_factor_score"] = volatility_score
    bab_factor_score = compute_bab_factor(bab_context)
    turnover_cost_score = compute_turnover_cost_score(result)

    # PEAD factor (Martineau 2022) — compute from result earnings data
    pead = compute_pead_factor(result)
    pead_score = pead["pead_factor_score"]
    sue = pead["sue_score"]
    rev_mom = pead["revision_momentum_3m"]

    return {
        "quality_factor_score": None if quality_score is None else float(np.clip(quality_score, -1.0, 1.0)),
        "qmj_factor_score": None if qmj_factor_score is None else float(np.clip(qmj_factor_score, -1.0, 1.0)),
        "qmj_component_count": len(qmj_components),
        "value_factor_score": value_score,
        "momentum_factor_score": float(np.clip(momentum_score, -1.0, 1.0)),
        "volatility_factor_score": 0.0 if volatility_score is None else volatility_score,
        "bab_factor_score": None if bab_factor_score is None else float(np.clip(bab_factor_score, -1.0, 1.0)),
        "turnover_cost_score": None if turnover_cost_score is None else float(np.clip(turnover_cost_score, -1.0, 1.0)),
        # Extended institutional factors
        "investment_factor_score": None if investment is None else float(np.clip(investment, -1.0, 1.0)),
        "accruals_factor_score": None if accruals is None else float(np.clip(accruals, -1.0, 1.0)),
        "leverage_factor_score": None if leverage is None else float(np.clip(leverage, -1.0, 1.0)),
        "reversal_factor_score": None if reversal is None else float(np.clip(reversal, -1.0, 1.0)),
        "residual_momentum_score": None if residual_mom is None else float(np.clip(residual_mom, -1.0, 1.0)),
        "idiosyncratic_vol_score": None if idio_vol is None else float(np.clip(idio_vol, -1.0, 1.0)),
        # PEAD factor (Martineau 2022)
        "pead_factor_score": None if pead_score is None else float(np.clip(pead_score, -1.0, 1.0)),
        "sue_score": None if sue is None else float(np.clip(sue, -1.0, 1.0)),
        "revision_momentum_3m": None if rev_mom is None else float(np.clip(rev_mom, -1.0, 1.0)),
        # Enterprise-grade factors (roadmap items #1, #2)
        "ev_ebit_factor_score": None if ev_ebit_score is None else float(np.clip(ev_ebit_score, -1.0, 1.0)),
        "gpa_factor_score": None if gpa_score is None else float(np.clip(gpa_score, -1.0, 1.0)),
        "f_score_factor_score": None if f_score_score is None else float(np.clip(f_score_score, -1.0, 1.0)),
        "f_score_gate": bool(result.get("f_score_gate", False)),
    }


def factor_tilt_adjustment(
    factor_scores: dict[str, float | None],
    factor_returns: dict | None,
    macro_regime: dict | None = None,
) -> float:
    """Translate factor momentum + macro regime into a bounded alpha adjustment.

    Core factors (quality, value, momentum) get 80% weight.
    Extended factors (investment, leverage, accruals) get 20% bonus when available.
    Macro regime tilts adjust the output by ±3% (Arnott et al. 2019).
    """
    if not factor_returns:
        return 0.0

    quality = factor_scores.get("quality_factor_score") or 0.0
    value = factor_scores.get("value_factor_score") or 0.0
    momentum = factor_scores.get("momentum_factor_score") or 0.0

    core_tilt = (
        0.35 * quality * float(factor_returns.get("quality_tilt", 0.0) or 0.0)
        + 0.30 * value * float(factor_returns.get("value_tilt", 0.0) or 0.0)
        + 0.35 * momentum * float(factor_returns.get("momentum_tilt", 0.0) or 0.0)
    )

    # PEAD contribution (Martineau 2022) — direct additive when available
    pead = factor_scores.get("pead_factor_score") or 0.0
    pead_weight = getattr(config, "PEAD_FACTOR_WEIGHT", 0.12)
    pead_bonus = pead * pead_weight if getattr(config, "PEAD_ENABLED", True) else 0.0

    # Extended factor bonus: investment + leverage + accruals when available
    extended_components: list[float] = []
    for key in ("investment_factor_score", "leverage_factor_score", "accruals_factor_score", "bab_factor_score"):
        score = factor_scores.get(key)
        if score is not None:
            extended_components.append(score)

    extended_bonus = float(np.mean(extended_components)) * 0.05 if extended_components else 0.0
    turnover_score = factor_scores.get("turnover_cost_score")
    turnover_bonus = (float(turnover_score) * 0.03) if turnover_score is not None else 0.0

    # Macro regime tilt (Arnott, Harvey, Kalesnik & Linnainmaa 2019)
    macro_tilt = 0.0
    if macro_regime and macro_regime.get("factor_tilts"):
        regime_tilts = macro_regime["factor_tilts"]
        # Apply regime tilts weighted by the stock's factor exposures
        macro_tilt += momentum * float(regime_tilts.get("momentum_tilt", 0.0))
        macro_tilt += quality * float(regime_tilts.get("quality_tilt", 0.0))
        inv_score = factor_scores.get("investment_factor_score") or 0.0
        macro_tilt += inv_score * float(regime_tilts.get("investment_tilt", 0.0))
        vol_score = factor_scores.get("volatility_factor_score") or 0.0
        macro_tilt += vol_score * float(regime_tilts.get("volatility_tilt", 0.0))
        rev_score = factor_scores.get("reversal_factor_score") or 0.0
        macro_tilt += rev_score * float(regime_tilts.get("reversal_tilt", 0.0))

    total_tilt = 0.12 * core_tilt + extended_bonus + turnover_bonus + pead_bonus + 0.03 * macro_tilt
    return float(np.clip(total_tilt, -0.25, 0.25))


def adjust_alpha_for_confidence(
    alpha: float,
    confidence: float,
    cross_sectional_mean: float = 0.0,
) -> tuple[float, float]:
    """Shrink alpha toward the cross-sectional mean without crushing it to zero."""
    floor = float(np.clip(getattr(config, "CONFIDENCE_FLOOR", 0.60), 0.0, 1.0))
    effective_confidence = max(floor, float(np.clip(confidence, 0.0, 1.0)))
    adjusted = cross_sectional_mean + effective_confidence * (alpha - cross_sectional_mean)
    return adjusted, effective_confidence
