"""Portfolio Construction Engine — joint optimization across all holdings.

Replaces per-stock scoring with a portfolio-aware allocation that maximises
risk-adjusted returns subject to practical constraints:

  - Expected returns from MoE forecasts + pillar scores
  - Covariance from historical daily returns
  - Max position weight (default 25%)
  - Sector concentration cap (default 40%)
  - FX cost penalty for non-GBP holdings
  - Min weight floor (don't hold dust positions)
  - Turnover penalty (discourage excessive rebalancing)

Uses scipy.optimize.minimize with SLSQP (supports equality + inequality
constraints).  Falls back to inverse-vol weights if optimisation fails.

Public API:
    optimize_portfolio(results, holdings, risk_data, position_weights, regime)
        -> PortfolioAllocation
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults (override via config.py)
# ---------------------------------------------------------------------------

MAX_WEIGHT = getattr(config, "MAX_POSITION_WEIGHT", 0.25)
MIN_WEIGHT = 0.02          # 2% floor — below this, don't bother holding
SECTOR_CAP = getattr(config, "DISCOVERY_SECTOR_CONCENTRATION_MAX", 0.40)
TURNOVER_PENALTY = 0.002   # 20bps penalty per unit of turnover
FX_COST_PER_LEG = getattr(config, "FX_FEE_TIER", 0.0075)
RISK_AVERSION = 2.0        # Lambda in mean-variance: higher = more conservative
LOOKBACK_DAYS = 180        # Historical covariance window

# FX rate cache (populated once per run)
_fx_cache: dict[str, float] = {}


def _get_fx_rate(currency: str) -> float:
    """Return the conversion rate from `currency` to GBP.

    GBP and GBX return 1.0 (GBX→GBP is handled separately as /100).
    EUR and USD are fetched from yfinance FX pairs. Cached for the session.
    """
    if currency in ("GBP", "GBX"):
        return 1.0

    if currency in _fx_cache:
        return _fx_cache[currency]

    # yfinance FX pair convention: XXXGBP=X
    pair = f"{currency}GBP=X"
    try:
        data = yf.download(pair, period="5d", progress=False, auto_adjust=True)
        if data is not None and not data.empty:
            rate = float(data["Close"].iloc[-1:].values[0])
            _fx_cache[currency] = rate
            logger.info("FX rate %s→GBP: %.4f", currency, rate)
            return rate
    except Exception as e:
        logger.warning("FX rate fetch failed for %s: %s", currency, e)

    # Fallback rates (approximate, better than 1:1)
    fallbacks = {"USD": 0.79, "EUR": 0.86}
    rate = fallbacks.get(currency, 1.0)
    _fx_cache[currency] = rate
    logger.warning("Using fallback FX rate %s→GBP: %.2f", currency, rate)
    return rate


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HoldingAllocation:
    """Optimised allocation for a single holding."""
    ticker: str
    name: str
    current_weight: float
    optimal_weight: float
    rebalance_delta: float       # optimal - current
    expected_return: float       # annualised
    volatility: float            # annualised
    sharpe_contribution: float   # marginal contribution to portfolio Sharpe
    sector: str
    currency: str
    action: str                  # from scoring engine
    aggregate_score: float
    fx_cost_if_rebalanced: float  # estimated FX cost to rebalance


@dataclass
class PortfolioAllocation:
    """Complete portfolio optimisation result."""
    holdings: list[HoldingAllocation]
    portfolio_expected_return: float    # annualised
    portfolio_volatility: float        # annualised
    portfolio_sharpe: float
    risk_free_rate: float
    method: str                        # 'mean_variance' | 'inverse_vol_fallback'
    sector_weights: dict[str, float]
    fx_exposure: dict[str, float]
    turnover: float                    # sum of |delta| / 2
    rebalance_trades: list[dict]       # concrete trade suggestions
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Expected returns estimation
# ---------------------------------------------------------------------------

def _estimate_expected_returns(
    results: list[dict],
    tickers: list[str],
) -> np.ndarray:
    """Estimate annualised expected returns from the unified 90-day outlook.

    Uses the pre-computed expected_return_90d from scoring.py (which is the
    63-trading-day MoE forecast ≈ 90 calendar days).  Falls back to aggregate
    score mapping when the forecast is unavailable.

    Blending (all in 90-day terms, then annualised):
      50%  MoE 90-day forecast   — direct price prediction
      25%  Aggregate score       — pillar-weighted quality signal
      25%  Historical 90d momentum — persistence / trend signal
    """
    mu = np.zeros(len(tickers))
    result_map = {r["ticker"]: r for r in results}

    for i, ticker in enumerate(tickers):
        r = result_map.get(ticker, {})

        # 1. MoE 90-day forecast (already in 90-day terms from scoring.py)
        moe_90d = r.get("expected_return_90d")
        if moe_90d is None:
            # Fallback: use short-horizon forecast extrapolated
            fc_pct = r.get("forecast_pct_change", 0) or 0
            horizon = r.get("forecast_horizon", getattr(config, "FORECAST_HORIZON_DAYS", 5))
            moe_90d = (fc_pct / 100) * (63 / max(horizon, 1))

        # 2. Aggregate score → 90-day expected return
        #    Score of +1.0 maps to +15% over 90 days (conservative)
        #    Score of -1.0 maps to -15% over 90 days
        agg = r.get("aggregate_score", 0) or 0
        score_90d = agg * 0.15

        # 3. Historical 90-day momentum (actual trailing return)
        try:
            data = yf.download(ticker, period="120d", progress=False, auto_adjust=True)
            if data is not None and len(data) >= 60:
                hist_90d = (float(data["Close"].iloc[-1:].values[0]) /
                            float(data["Close"].iloc[-63:-62].values[0]) - 1)
            else:
                hist_90d = 0.0
        except Exception:
            hist_90d = 0.0

        # Blend in 90-day terms, then annualise
        expected_90d = 0.50 * moe_90d + 0.25 * score_90d + 0.25 * hist_90d
        mu[i] = expected_90d * (252 / 63)  # annualise for mean-variance math

    return mu


# ---------------------------------------------------------------------------
# Covariance matrix
# ---------------------------------------------------------------------------

def _estimate_covariance(tickers: list[str]) -> np.ndarray:
    """Historical daily return covariance matrix, annualised.

    Uses shrinkage toward diagonal (Ledoit-Wolf style simple shrinkage)
    to improve conditioning for small sample sizes.

    IMPORTANT: Always returns an n×n matrix aligned to the input `tickers`
    list.  If yfinance drops a ticker, that ticker gets a diagonal fallback
    (30% vol, zero correlation) so the matrix dimensions stay consistent
    with the weight vector.
    """
    n = len(tickers)
    FALLBACK_VAR = 0.30 ** 2  # 30% annualised vol squared

    try:
        data = yf.download(tickers, period=f"{LOOKBACK_DAYS}d", progress=False, auto_adjust=True)
        if data is None or data.empty:
            raise ValueError("No price data")

        # Handle single vs multi ticker
        if n == 1:
            closes = data["Close"].to_frame(tickers[0])
        else:
            closes = data["Close"]

        # Identify which tickers we actually got usable data for
        # (yfinance may return columns with all NaN for delisted tickers)
        available = [t for t in tickers if t in closes.columns and not closes[t].isna().all()]
        missing = [t for t in tickers if t not in available]
        if missing:
            logger.warning("Covariance: missing price data for %s — using diagonal fallback for these", missing)

        if not available:
            raise ValueError("No tickers had price data")

        closes = closes[available].dropna()
        daily_returns = closes.pct_change(fill_method=None).dropna()

        if len(daily_returns) < 30:
            raise ValueError("Insufficient return data")

        # Compute sample covariance for available tickers
        avail_cov = daily_returns.cov().values * 252  # annualise

        # Simple shrinkage: blend toward diagonal
        shrinkage = 0.3
        diag = np.diag(np.diag(avail_cov))
        avail_cov = (1 - shrinkage) * avail_cov + shrinkage * diag

        # Build full n×n matrix aligned to original tickers list
        # Missing tickers get diagonal fallback (FALLBACK_VAR, zero correlation)
        avail_idx = {t: i for i, t in enumerate(available)}
        cov = np.eye(n) * FALLBACK_VAR
        for i, ti in enumerate(tickers):
            for j, tj in enumerate(tickers):
                if ti in avail_idx and tj in avail_idx:
                    cov[i, j] = avail_cov[avail_idx[ti], avail_idx[tj]]

        # Ensure positive semi-definite
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-8)
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return cov

    except Exception as e:
        logger.warning("Covariance estimation failed (%s), using diagonal fallback", e)
        # Fallback: assume 30% vol, zero correlation
        return np.eye(n) * FALLBACK_VAR


# ---------------------------------------------------------------------------
# Current weights
# ---------------------------------------------------------------------------

def _current_weights(holdings: list[dict], results: list[dict]) -> np.ndarray:
    """Calculate current portfolio weights from market values (all converted to GBP)."""
    result_map = {r["ticker"]: r for r in results}
    values = []

    for h in holdings:
        ticker = h["ticker"]
        r = result_map.get(ticker, {})
        price = r.get("current_price", 0) or 0
        qty = h.get("quantity", 0)
        currency = h.get("currency", "GBP")

        value = price * qty
        # Convert GBX pence to GBP pounds
        if currency == "GBX":
            value /= 100
        # Convert foreign currencies to GBP
        fx = _get_fx_rate(currency)
        value *= fx

        values.append(value)

    total = sum(values) or 1
    return np.array([v / total for v in values])


# ---------------------------------------------------------------------------
# Sector mapping
# ---------------------------------------------------------------------------

def _get_sectors(results: list[dict], tickers: list[str]) -> dict[str, str]:
    """Map tickers to sectors from analysis results or yfinance."""
    result_map = {r["ticker"]: r for r in results}
    sectors = {}
    for ticker in tickers:
        r = result_map.get(ticker, {})
        # Try from discovery/scoring data
        sector = None
        for key in ("sector",):
            sector = r.get(key)
            if sector:
                break
        if not sector:
            try:
                info = yf.Ticker(ticker).info
                sector = info.get("sector", "Unknown")
            except Exception:
                sector = "Unknown"
        sectors[ticker] = sector
    return sectors


# ---------------------------------------------------------------------------
# Mean-variance optimisation
# ---------------------------------------------------------------------------

def _mean_variance_optimize(
    mu: np.ndarray,
    cov: np.ndarray,
    current_w: np.ndarray,
    sectors: dict[str, str],
    tickers: list[str],
    currencies: list[str],
    risk_aversion: float = RISK_AVERSION,
    results: list[dict] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Solve for optimal weights maximising: mu'w - (lambda/2)*w'Cov*w - turnover_cost.

    Subject to:
      - weights sum to 1
      - each weight in [MIN_WEIGHT, MAX_WEIGHT]
      - sector concentration <= SECTOR_CAP
      - FX cost penalised in objective

    Returns (optimal_weights, warnings).
    """
    n = len(tickers)
    warnings = []

    # Build sector groups for constraints
    sector_groups = {}
    for i, ticker in enumerate(tickers):
        s = sectors.get(ticker, "Unknown")
        sector_groups.setdefault(s, []).append(i)

    # FX cost vector: cost of changing weight for non-GBP holdings
    fx_cost = np.array([
        FX_COST_PER_LEG if c not in ("GBP", "GBX") else 0.0
        for c in currencies
    ])

    def objective(w):
        # Expected return
        ret = mu @ w
        # Risk
        risk = w @ cov @ w
        # Turnover cost
        turnover = np.sum(np.abs(w - current_w))
        turnover_cost = TURNOVER_PENALTY * turnover
        # FX cost on rebalanced portion
        fx_rebal_cost = np.sum(fx_cost * np.abs(w - current_w))
        # Maximise return - risk - costs → minimise negative
        return -(ret - (risk_aversion / 2) * risk - turnover_cost - fx_rebal_cost)

    # Constraints
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # weights sum to 1
    ]

    # Sector concentration constraints
    for sector, indices in sector_groups.items():
        if len(indices) > 0:
            constraints.append({
                "type": "ineq",
                "fun": lambda w, idx=indices: SECTOR_CAP - sum(w[i] for i in idx),
            })

    # Bounds: each weight between MIN_WEIGHT and MAX_WEIGHT
    # Risk overlay may reduce the upper bound for small/micro cap holdings
    result_map = {r["ticker"]: r for r in results} if results else {}
    bounds = []
    for ticker in tickers:
        r = result_map.get(ticker, {})
        scale = r.get("max_weight_scale", 1.0)
        upper = MAX_WEIGHT * scale
        bounds.append((MIN_WEIGHT, max(MIN_WEIGHT, upper)))

    # Initial guess: current weights (feasible starting point)
    w0 = current_w.copy()
    # Clip to bounds
    w0 = np.clip(w0, MIN_WEIGHT, MAX_WEIGHT)
    w0 = w0 / w0.sum()  # re-normalise

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )

    if result.success:
        optimal_w = result.x
        # Clean up small numerical errors
        optimal_w = np.maximum(optimal_w, 0)
        optimal_w = optimal_w / optimal_w.sum()
    else:
        warnings.append(f"Optimisation did not converge: {result.message}")
        logger.warning("Mean-variance optimisation failed: %s", result.message)
        optimal_w = current_w

    return optimal_w, warnings


# ---------------------------------------------------------------------------
# Rebalance trade suggestions
# ---------------------------------------------------------------------------

def _build_rebalance_trades(
    tickers: list[str],
    names: list[str],
    current_w: np.ndarray,
    optimal_w: np.ndarray,
    total_value: float,
) -> list[dict]:
    """Generate concrete trade suggestions for rebalancing.

    Only suggests trades where the delta is significant (> 2% of portfolio).
    """
    trades = []
    for i, ticker in enumerate(tickers):
        delta = optimal_w[i] - current_w[i]
        if abs(delta) < 0.02:  # ignore < 2% changes
            continue

        trade_value = delta * total_value
        direction = "BUY" if delta > 0 else "TRIM"

        trades.append({
            "ticker": ticker,
            "name": names[i],
            "direction": direction,
            "current_weight": round(current_w[i] * 100, 1),
            "optimal_weight": round(optimal_w[i] * 100, 1),
            "delta_pct": round(delta * 100, 1),
            "trade_value": round(abs(trade_value), 2),
        })

    # Sort: largest absolute delta first
    trades.sort(key=lambda t: abs(t["delta_pct"]), reverse=True)
    return trades


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize_portfolio(
    results: list[dict],
    holdings: list[dict],
    risk_data: dict | None = None,
    position_weights: list[dict] | None = None,
    regime: dict | None = None,
) -> PortfolioAllocation:
    """Run full portfolio optimisation.

    Args:
        results: per-holding analysis from scoring.analyse_portfolio()
        holdings: raw portfolio.json holdings
        risk_data: from portfolio_risk.assess_portfolio_risk()
        position_weights: from position_sizing (used as fallback)
        regime: VIX regime dict

    Returns:
        PortfolioAllocation with optimal weights and trade suggestions.
    """
    tickers = [h["ticker"] for h in holdings]
    names = [h.get("name", h["ticker"]) for h in holdings]
    currencies = [h.get("currency", "GBP") for h in holdings]
    result_map = {r["ticker"]: r for r in results}
    n = len(tickers)

    # Risk-free rate (approximate from TNX if available)
    risk_free = 0.04  # 4% default
    try:
        tnx = yf.download("^TNX", period="5d", progress=False, auto_adjust=True)
        if tnx is not None and not tnx.empty:
            risk_free = float(tnx["Close"].iloc[-1:].values[0]) / 100
    except Exception:
        pass

    # Current weights
    current_w = _current_weights(holdings, results)

    # Total portfolio value (GBP)
    total_value = 0
    for h in holdings:
        r = result_map.get(h["ticker"], {})
        price = r.get("current_price", 0) or 0
        qty = h.get("quantity", 0)
        currency = h.get("currency", "GBP")
        val = price * qty
        if currency == "GBX":
            val /= 100
        val *= _get_fx_rate(currency)
        total_value += val

    # Expected returns
    mu = _estimate_expected_returns(results, tickers)

    # Covariance
    cov = _estimate_covariance(tickers)

    # Sectors
    sectors = _get_sectors(results, tickers)

    # Adjust risk aversion by regime
    risk_aversion = RISK_AVERSION
    regime_label = (regime or {}).get("regime_label", "NEUTRAL")
    if regime_label == "BEAR":
        risk_aversion *= 1.5  # more conservative in bear markets
    elif regime_label == "BULL":
        risk_aversion *= 0.8  # slightly more aggressive in bull

    # Optimise
    method = "mean_variance"
    optimal_w, warnings = _mean_variance_optimize(
        mu, cov, current_w, sectors, tickers, currencies, risk_aversion,
        results=results,
    )

    # Check if optimisation produced meaningful change
    if np.allclose(optimal_w, current_w, atol=0.01):
        warnings.append("Portfolio is already near-optimal — no significant rebalancing needed")

    # Portfolio-level stats
    port_ret = float(mu @ optimal_w)
    port_vol = float(np.sqrt(optimal_w @ cov @ optimal_w))
    port_sharpe = (port_ret - risk_free) / port_vol if port_vol > 0 else 0

    # Per-holding volatilities
    vols = np.sqrt(np.diag(cov))

    # Marginal contribution to risk (MCTR)
    port_var = optimal_w @ cov @ optimal_w
    if port_var > 0:
        mctr = (cov @ optimal_w) / np.sqrt(port_var)
    else:
        mctr = np.zeros(n)

    # Sharpe contribution per holding
    sharpe_contrib = np.zeros(n)
    for i in range(n):
        if vols[i] > 0:
            holding_sharpe = (mu[i] - risk_free) / vols[i]
            sharpe_contrib[i] = holding_sharpe * optimal_w[i]

    # Build per-holding allocations
    holding_allocs = []
    for i, ticker in enumerate(tickers):
        r = result_map.get(ticker, {})
        delta = optimal_w[i] - current_w[i]
        fx_cost = abs(delta) * FX_COST_PER_LEG if currencies[i] not in ("GBP", "GBX") else 0

        holding_allocs.append(HoldingAllocation(
            ticker=ticker,
            name=names[i],
            current_weight=round(float(current_w[i]), 4),
            optimal_weight=round(float(optimal_w[i]), 4),
            rebalance_delta=round(float(delta), 4),
            expected_return=round(float(mu[i]), 4),
            volatility=round(float(vols[i]), 4),
            sharpe_contribution=round(float(sharpe_contrib[i]), 4),
            sector=sectors.get(ticker, "Unknown"),
            currency=currencies[i],
            action=r.get("action", "KEEP"),
            aggregate_score=r.get("aggregate_score", 0),
            fx_cost_if_rebalanced=round(fx_cost, 4),
        ))

    # Sector weights under optimal allocation
    opt_sector_weights = {}
    for i, ticker in enumerate(tickers):
        s = sectors.get(ticker, "Unknown")
        opt_sector_weights[s] = opt_sector_weights.get(s, 0) + optimal_w[i]
    opt_sector_weights = {k: round(v, 4) for k, v in opt_sector_weights.items()}

    # FX exposure
    fx_exposure = {}
    for i, c in enumerate(currencies):
        norm_c = "GBP" if c == "GBX" else c
        fx_exposure[norm_c] = fx_exposure.get(norm_c, 0) + optimal_w[i]
    fx_exposure = {k: round(v, 4) for k, v in fx_exposure.items()}

    # Turnover
    turnover = float(np.sum(np.abs(optimal_w - current_w))) / 2

    # Rebalance trades
    rebalance_trades = _build_rebalance_trades(
        tickers, names, current_w, optimal_w, total_value,
    )

    return PortfolioAllocation(
        holdings=holding_allocs,
        portfolio_expected_return=round(port_ret, 4),
        portfolio_volatility=round(port_vol, 4),
        portfolio_sharpe=round(port_sharpe, 3),
        risk_free_rate=round(risk_free, 4),
        method=method,
        sector_weights=opt_sector_weights,
        fx_exposure=fx_exposure,
        turnover=round(turnover, 4),
        rebalance_trades=rebalance_trades,
        warnings=warnings,
    )


def optimize_with_candidate(
    results: list[dict],
    holdings: list[dict],
    candidate_ticker: str,
    replace_ticker: str,
    risk_data: dict | None = None,
    regime: dict | None = None,
) -> tuple[PortfolioAllocation, PortfolioAllocation]:
    """Compare current portfolio vs portfolio with a swap.

    Returns (current_allocation, proposed_allocation) for side-by-side comparison.
    """
    current_alloc = optimize_portfolio(results, holdings, risk_data, regime=regime)

    # Build hypothetical holdings/results with the swap
    new_holdings = [h for h in holdings if h["ticker"] != replace_ticker]
    replaced = next((h for h in holdings if h["ticker"] == replace_ticker), None)
    if replaced:
        new_holdings.append({
            "ticker": candidate_ticker,
            "name": candidate_ticker,
            "quantity": replaced["quantity"],
            "avg_buy_price": 0,  # new position
            "currency": replaced.get("currency", "GBP"),
        })

    # Re-run analysis for the candidate (simplified — use discovery data if available)
    new_results = [r for r in results if r["ticker"] != replace_ticker]
    # Add placeholder result for candidate
    try:
        from engine.scoring import analyse_holding
        cand_holding = new_holdings[-1]
        cand_result = analyse_holding(cand_holding)
        new_results.append(cand_result)
    except Exception as e:
        logger.warning("Could not analyse candidate %s: %s", candidate_ticker, e)
        return current_alloc, current_alloc

    proposed_alloc = optimize_portfolio(new_results, new_holdings, risk_data, regime=regime)

    return current_alloc, proposed_alloc
