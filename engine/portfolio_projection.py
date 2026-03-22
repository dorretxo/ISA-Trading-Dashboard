"""90-Day Portfolio Return Projection using Monte Carlo simulation.

Combines the MoE directional forecast (drift) with historical volatility
and cross-asset correlations to produce a realistic distribution of
portfolio returns over the next 90 trading days.

Output: per-ticker and portfolio-level projections with confidence intervals.
"""

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from utils.data_fetch import get_current_price, get_price_history

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECTION_HORIZON = 63          # Trading days (~90 calendar days)
N_SIMULATIONS = 5000             # Monte Carlo paths
CONFIDENCE_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90)  # Percentile bands

# Drift blending: how much to trust MoE vs pure historical
MOE_DRIFT_WEIGHT = 0.60         # 60% MoE signal, 40% historical drift


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TickerProjection:
    ticker: str
    current_price: float
    moe_predicted_price: float        # MoE point estimate at 90d
    moe_pct_change: float             # MoE predicted return %
    historical_annual_drift: float    # Annualized historical return
    annual_volatility: float          # Annualized vol
    # Monte Carlo distribution (percentiles)
    projected_prices: dict            # {percentile: price}
    projected_returns: dict           # {percentile: return %}
    expected_return_pct: float        # Mean of simulated returns
    prob_positive: float              # Probability of positive return


@dataclass
class PortfolioProjection:
    ticker_projections: list[TickerProjection]
    # Portfolio-level aggregates
    current_value: float
    projected_values: dict            # {percentile: value}
    projected_returns: dict           # {percentile: return %}
    expected_return_pct: float
    expected_value: float
    prob_positive: float
    # Simulation metadata
    n_simulations: int
    horizon_days: int


# ---------------------------------------------------------------------------
# MoE forecast integration
# ---------------------------------------------------------------------------

def _get_moe_forecast(ticker: str, horizon: int = PROJECTION_HORIZON) -> tuple[float | None, float | None]:
    """Get MoE price prediction and % change for the given horizon.

    Returns (predicted_price, pct_change) or (None, None) on failure.
    """
    try:
        from engine.forecasting import forecast
        fc = forecast(ticker, horizon_days=horizon)
        return fc.predicted_price, fc.pct_change
    except Exception as e:
        logger.debug("MoE forecast unavailable for %s: %s", ticker, e)
        return None, None


# ---------------------------------------------------------------------------
# Historical statistics
# ---------------------------------------------------------------------------

def _compute_stats(ticker: str) -> tuple[float, float, np.ndarray]:
    """Compute annualized drift, volatility, and daily returns array.

    Returns (annual_drift, annual_vol, daily_returns).
    """
    df = get_price_history(ticker)
    if df is None or df.empty or len(df) < 30:
        return 0.0, 0.30, np.array([])  # Default 30% vol

    closes = df["Close"].tail(252).values.astype(float)  # Up to 1 year
    daily_returns = np.diff(np.log(closes))  # Log returns

    if len(daily_returns) < 20:
        return 0.0, 0.30, daily_returns

    annual_drift = float(np.mean(daily_returns) * 252)
    annual_vol = float(np.std(daily_returns) * math.sqrt(252))

    return annual_drift, max(annual_vol, 0.05), daily_returns


# ---------------------------------------------------------------------------
# Correlation matrix from daily returns
# ---------------------------------------------------------------------------

def _build_correlation_matrix(
    tickers: list[str],
    returns_dict: dict[str, np.ndarray],
) -> np.ndarray:
    """Build a correlation matrix, filling missing data with identity."""
    n = len(tickers)
    if n <= 1:
        return np.eye(n)

    # Align to shortest common length
    lengths = [len(returns_dict.get(t, [])) for t in tickers]
    min_len = min(l for l in lengths if l > 0) if any(l > 0 for l in lengths) else 0

    if min_len < 20:
        return np.eye(n)

    aligned = np.zeros((min_len, n))
    for i, t in enumerate(tickers):
        r = returns_dict.get(t, np.zeros(min_len))
        aligned[:, i] = r[-min_len:]

    corr = np.corrcoef(aligned, rowvar=False)
    # Fix any NaN (from zero-variance series)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    return corr


# ---------------------------------------------------------------------------
# Cholesky decomposition for correlated simulations
# ---------------------------------------------------------------------------

def _cholesky_or_fallback(corr: np.ndarray) -> np.ndarray:
    """Compute Cholesky decomposition, falling back to eigen-repair if not PD."""
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        # Repair: eigenvalue clipping to make positive semi-definite
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        repaired = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        # Re-normalize to correlation matrix
        d = np.sqrt(np.diag(repaired))
        repaired = repaired / np.outer(d, d)
        np.fill_diagonal(repaired, 1.0)
        return np.linalg.cholesky(repaired)


# ---------------------------------------------------------------------------
# Monte Carlo simulation (correlated GBM paths)
# ---------------------------------------------------------------------------

def _simulate_portfolio(
    tickers: list[str],
    current_prices: list[float],
    drifts: list[float],
    vols: list[float],
    weights: list[float],
    corr_matrix: np.ndarray,
    horizon: int = PROJECTION_HORIZON,
    n_sims: int = N_SIMULATIONS,
) -> tuple[np.ndarray, np.ndarray]:
    """Run correlated Geometric Brownian Motion Monte Carlo.

    Returns:
        ticker_terminal: (n_sims, n_tickers) — terminal prices per sim
        portfolio_returns: (n_sims,) — portfolio-level returns per sim
    """
    n = len(tickers)
    L = _cholesky_or_fallback(corr_matrix)

    # Daily parameters
    dt = 1.0 / 252
    daily_drifts = np.array([(d - 0.5 * v**2) * dt for d, v in zip(drifts, vols)])
    daily_vols = np.array([v * math.sqrt(dt) for v in vols])

    # Generate correlated random shocks
    rng = np.random.default_rng(seed=42)
    Z = rng.standard_normal((n_sims, horizon, n))
    correlated_Z = Z @ L.T  # Apply Cholesky to correlate

    # Simulate GBM paths
    log_returns = daily_drifts[np.newaxis, np.newaxis, :] + daily_vols[np.newaxis, np.newaxis, :] * correlated_Z
    cumulative = np.cumsum(log_returns, axis=1)  # (n_sims, horizon, n)

    # Terminal prices
    prices_0 = np.array(current_prices)
    terminal_prices = prices_0 * np.exp(cumulative[:, -1, :])  # (n_sims, n)

    # Portfolio returns (weighted sum of per-ticker returns)
    ticker_returns = (terminal_prices - prices_0) / prices_0  # (n_sims, n)
    w = np.array(weights)
    portfolio_returns = ticker_returns @ w  # (n_sims,)

    return terminal_prices, portfolio_returns


# ---------------------------------------------------------------------------
# Main projection function
# ---------------------------------------------------------------------------

def project_portfolio_return(
    results: list[dict],
    holdings: list[dict],
    position_weights: list[dict] | None = None,
) -> PortfolioProjection:
    """Compute 90-day return projection for the current portfolio.

    Args:
        results: analyse_holding() output per holding
        holdings: portfolio.json holdings list
        position_weights: from calculate_inverse_vol_weights() (optional)

    Returns:
        PortfolioProjection with per-ticker and portfolio-level stats.
    """
    tickers = [r["ticker"] for r in results]
    n = len(tickers)

    # Gather per-ticker data
    current_prices = []
    moe_preds = []
    moe_pcts = []
    hist_drifts = []
    hist_vols = []
    returns_dict = {}

    for r in results:
        ticker = r["ticker"]
        price = r.get("current_price") or get_current_price(ticker) or 0
        current_prices.append(price)

        # MoE forecast
        moe_price, moe_pct = _get_moe_forecast(ticker, PROJECTION_HORIZON)
        moe_preds.append(moe_price or price)
        moe_pcts.append(moe_pct or 0.0)

        # Historical stats
        drift, vol, daily_ret = _compute_stats(ticker)
        hist_drifts.append(drift)
        hist_vols.append(vol)
        if len(daily_ret) > 0:
            returns_dict[ticker] = daily_ret

    # Blend MoE drift with historical drift
    blended_drifts = []
    for i in range(n):
        moe_annual = (moe_pcts[i] / 100) * (252 / PROJECTION_HORIZON)  # Annualize MoE signal
        blended = MOE_DRIFT_WEIGHT * moe_annual + (1 - MOE_DRIFT_WEIGHT) * hist_drifts[i]
        blended_drifts.append(blended)

    # Position weights
    if position_weights:
        weights = []
        pw_map = {pw["ticker"]: pw["current_weight"] for pw in position_weights}
        for t in tickers:
            weights.append(pw_map.get(t, 1.0 / n))
    else:
        # Compute from market values
        values = []
        for r, h in zip(results, holdings):
            price = r.get("current_price", 0)
            qty = h.get("quantity", 0)
            v = price * qty
            if h.get("currency") == "GBX":
                v *= 0.01
            values.append(v)
        total_v = sum(values) or 1.0
        weights = [v / total_v for v in values]

    # Correlation matrix
    corr = _build_correlation_matrix(tickers, returns_dict)

    # Run Monte Carlo
    terminal_prices, portfolio_returns = _simulate_portfolio(
        tickers, current_prices, blended_drifts, hist_vols, weights, corr,
    )

    # Build per-ticker projections
    ticker_projections = []
    for i, ticker in enumerate(tickers):
        prices_i = terminal_prices[:, i]
        returns_i = (prices_i - current_prices[i]) / current_prices[i] * 100

        proj_prices = {p: float(np.percentile(prices_i, p * 100)) for p in CONFIDENCE_LEVELS}
        proj_returns = {p: float(np.percentile(returns_i, p * 100)) for p in CONFIDENCE_LEVELS}

        ticker_projections.append(TickerProjection(
            ticker=ticker,
            current_price=current_prices[i],
            moe_predicted_price=moe_preds[i],
            moe_pct_change=moe_pcts[i],
            historical_annual_drift=hist_drifts[i],
            annual_volatility=hist_vols[i],
            projected_prices=proj_prices,
            projected_returns=proj_returns,
            expected_return_pct=float(np.mean(returns_i)),
            prob_positive=float(np.mean(returns_i > 0)),
        ))

    # Portfolio-level aggregation
    port_returns_pct = portfolio_returns * 100
    total_value = sum(
        r.get("current_price", 0) * h.get("quantity", 0) * (0.01 if h.get("currency") == "GBX" else 1.0)
        for r, h in zip(results, holdings)
    )
    projected_values = {
        p: total_value * (1 + np.percentile(portfolio_returns, p * 100))
        for p in CONFIDENCE_LEVELS
    }
    projected_returns = {
        p: float(np.percentile(port_returns_pct, p * 100))
        for p in CONFIDENCE_LEVELS
    }

    return PortfolioProjection(
        ticker_projections=ticker_projections,
        current_value=total_value,
        projected_values=projected_values,
        projected_returns=projected_returns,
        expected_return_pct=float(np.mean(port_returns_pct)),
        expected_value=total_value * (1 + float(np.mean(portfolio_returns))),
        prob_positive=float(np.mean(portfolio_returns > 0)),
        n_simulations=N_SIMULATIONS,
        horizon_days=PROJECTION_HORIZON,
    )


# ---------------------------------------------------------------------------
# Swap impact comparison
# ---------------------------------------------------------------------------

def project_swap_impact(
    results: list[dict],
    holdings: list[dict],
    swap_out_ticker: str,
    swap_in_ticker: str,
    position_weights: list[dict] | None = None,
) -> tuple[PortfolioProjection, PortfolioProjection]:
    """Compare 90-day projection before and after a proposed swap.

    Returns (current_projection, swapped_projection).
    """
    # Current portfolio projection
    current_proj = project_portfolio_return(results, holdings, position_weights)

    # Build modified portfolio: replace swap_out with swap_in
    mod_results = []
    mod_holdings = []
    swap_idx = None

    for i, (r, h) in enumerate(zip(results, holdings)):
        if r["ticker"] == swap_out_ticker:
            swap_idx = i
            # Replace with swap-in ticker
            swap_price = get_current_price(swap_in_ticker) or 0
            swap_value = r.get("current_price", 0) * h.get("quantity", 0)
            swap_qty = swap_value / swap_price if swap_price > 0 else 0

            mod_results.append({
                **r,
                "ticker": swap_in_ticker,
                "current_price": swap_price,
            })
            mod_holdings.append({
                **h,
                "ticker": swap_in_ticker,
                "quantity": swap_qty,
            })
        else:
            mod_results.append(r)
            mod_holdings.append(h)

    if swap_idx is None:
        logger.warning("Swap-out ticker %s not found in portfolio", swap_out_ticker)
        return current_proj, current_proj

    # Run projection on modified portfolio
    swapped_proj = project_portfolio_return(mod_results, mod_holdings, position_weights)

    return current_proj, swapped_proj
