"""Portfolio Construction Engine — multi-method ensemble allocator.

Replaces per-stock scoring with a portfolio-aware allocation that maximises
risk-adjusted returns subject to practical constraints.

Five optimisation methods vote, and the final weights are a Sharpe-weighted
ensemble (Bates & Granger 1969, Timmermann 2006):

  1. Mean-Variance (Markowitz 1952) — μ/Σ reward-risk tradeoff.
  2. Minimum Variance (Jagannathan & Ma 2003) — μ-free, robust to estimation
     error in expected returns.
  3. Risk Parity / Equal Risk Contribution (Maillard, Roncalli & Teïletche
     2010) — equal risk contribution per holding, μ-free.
  4. Black-Litterman (Black & Litterman 1992; He & Litterman 1999) — Bayesian
     fusion of a neutral prior with the scoring engine's views.
  5. Hierarchical Risk Parity (López de Prado 2016) — cluster-based
     inverse-variance split, numerically robust, no matrix inversion.

μ (expected returns) are calibrated via James-Stein shrinkage toward the
cross-sectional mean (Jorion 1986; Stein 1956), with the score→return scale
IC-anchored to realised 90-day returns from `signal_backtest`.

Σ (covariance) blends Ledoit-Wolf shrinkage + EWMA with the Gerber robust
co-movement statistic (Gerber, Markowitz, Pujara & Zhu 2022) and inflates
off-diagonal correlations under BEAR regimes (Longin & Solnik 2001).

All methods project onto a shared feasible set: per-position min/max, Kelly
caps, sector concentration, STRONG SELL → MIN_WEIGHT, SELL → 2·MIN_WEIGHT.

Public API:
    optimize_portfolio(results, holdings, risk_data, position_weights, regime)
        -> PortfolioAllocation
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

import config

logger = logging.getLogger(__name__)


def _resolve_yahoo_symbol(ticker: str) -> str:
    symbol = str(ticker or "").upper().strip()
    if not symbol:
        return ""
    try:
        from utils.global_universe import resolve_yahoo_ticker

        return resolve_yahoo_ticker(symbol)
    except Exception:
        return symbol


def _is_blocked_symbol(ticker: str) -> bool:
    if not ticker:
        return True
    try:
        from utils.global_universe import is_excluded_ticker

        return is_excluded_ticker(ticker)
    except Exception:
        return False


def _yahoo_download_map(tickers: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    symbol_to_originals: dict[str, list[str]] = {}
    for raw in tickers:
        original = str(raw or "").upper().strip()
        symbol = _resolve_yahoo_symbol(original)
        if not symbol or _is_blocked_symbol(symbol):
            continue
        symbol_to_originals.setdefault(symbol, []).append(original)
    return sorted(symbol_to_originals), symbol_to_originals


def _extract_close_series(data: pd.DataFrame, symbol: str, batch_len: int) -> pd.Series | None:
    if data is None or data.empty:
        return None
    symbol_key = str(symbol or "").upper().strip()
    if not isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns:
            return None
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            if symbol in close.columns:
                return close[symbol]
            if batch_len == 1 and len(close.columns):
                return close.iloc[:, 0]
            return None
        return close

    for level in range(data.columns.nlevels):
        raw_values = list(data.columns.get_level_values(level))
        upper_values = [str(v).upper().strip() for v in raw_values]
        if symbol_key not in upper_values:
            continue
        matched = raw_values[upper_values.index(symbol_key)]
        try:
            sliced = data.xs(matched, axis=1, level=level, drop_level=True)
        except Exception:
            continue
        if isinstance(sliced, pd.DataFrame) and "Close" in sliced.columns:
            close = sliced["Close"]
            return close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
        if isinstance(sliced, pd.Series):
            return sliced

    for level in range(data.columns.nlevels):
        raw_values = list(data.columns.get_level_values(level))
        upper_values = [str(v).upper().strip() for v in raw_values]
        if "CLOSE" not in upper_values:
            continue
        matched = raw_values[upper_values.index("CLOSE")]
        try:
            close = data.xs(matched, axis=1, level=level, drop_level=True)
        except Exception:
            continue
        if isinstance(close, pd.DataFrame):
            col_lookup = {str(col).upper().strip(): col for col in close.columns}
            if symbol_key in col_lookup:
                return close[col_lookup[symbol_key]]
            if batch_len == 1 and len(close.columns):
                return close.iloc[:, 0]
        elif isinstance(close, pd.Series):
            return close
    return None


def _close_frame_from_yahoo_download(
    data: pd.DataFrame,
    download_symbols: list[str],
    symbol_to_originals: dict[str, list[str]],
) -> pd.DataFrame:
    closes: dict[str, pd.Series] = {}
    for symbol in download_symbols:
        series = _extract_close_series(data, symbol, len(download_symbols))
        if series is None:
            continue
        series = pd.to_numeric(series, errors="coerce").dropna()
        if series.empty:
            continue
        for original in symbol_to_originals.get(symbol, [symbol]):
            closes[original] = series
    return pd.DataFrame(closes)

# ---------------------------------------------------------------------------
# Configuration defaults (override via config.py)
# ---------------------------------------------------------------------------

MAX_WEIGHT = getattr(config, "MAX_POSITION_WEIGHT", 0.25)
MIN_WEIGHT = 0.02          # 2% floor — below this, don't bother holding
SECTOR_CAP = getattr(
    config,
    "DISCOVERY_SECTOR_PCT_CAP",
    getattr(config, "DISCOVERY_SECTOR_CONCENTRATION_MAX", 0.30),
)
TURNOVER_PENALTY = 0.002   # 20bps flat penalty (fallback; overridden by Almgren-Chriss when data available)
FX_COST_PER_LEG = getattr(config, "FX_FEE_TIER", 0.0075)
RISK_AVERSION = 2.0        # Lambda in mean-variance: higher = more conservative
LOOKBACK_DAYS = 180        # Historical covariance window

# Ensemble/v2 flags (read via getattr so missing config is safe)
OPTIMIZER_MU_V2 = getattr(config, "OPTIMIZER_MU_V2", True)
OPTIMIZER_MU_WEIGHTS = getattr(config, "OPTIMIZER_MU_WEIGHTS", (0.70, 0.30))
OPTIMIZER_MU_SHRINKAGE_MAX = getattr(config, "OPTIMIZER_MU_SHRINKAGE_MAX", 0.40)
OPTIMIZER_COV_METHOD = getattr(config, "OPTIMIZER_COV_METHOD", "blend")
OPTIMIZER_BEAR_CORR_INFLATE = getattr(config, "OPTIMIZER_BEAR_CORR_INFLATE", 0.20)
OPTIMIZER_ENSEMBLE_METHODS = getattr(
    config,
    "OPTIMIZER_ENSEMBLE_METHODS",
    ["mean_variance", "min_variance", "risk_parity", "black_litterman", "hrp"],
)
OPTIMIZER_ENSEMBLE_TEMPERATURE = getattr(config, "OPTIMIZER_ENSEMBLE_TEMPERATURE", 0.5)
OPTIMIZER_METHOD_WEIGHT_MIN = getattr(config, "OPTIMIZER_METHOD_WEIGHT_MIN", 0.05)
OPTIMIZER_METHOD_WEIGHT_MAX = getattr(config, "OPTIMIZER_METHOD_WEIGHT_MAX", 0.50)
OPTIMIZER_BL_TAU = getattr(config, "OPTIMIZER_BL_TAU", 0.05)

# History file for ensemble method weighting (Sharpe-realised combiner)
OPTIMIZER_HISTORY_PATH = os.path.join("feature_cache", "optimizer_history.json")

# FX rate cache (populated once per run)
_fx_cache: dict[str, float] = {}

# Score-IC cache (populated once per optimiser run)
_score_ic_cache: dict = {"k": None, "n": 0}


# ---------------------------------------------------------------------------
# Transaction cost + FX
# ---------------------------------------------------------------------------

def estimate_transaction_cost(
    vol_ann: float,
    avg_dollar_volume: float,
    trade_value: float,
    currency: str = "GBP",
) -> float:
    """Almgren-Chriss (2001) stock-specific transaction cost estimate.

    Components:
    - Bid-ask spread: base 10bps + vol-proportional
    - Market impact: square-root of participation rate × volatility
    - FX round-trip: for non-GBP positions

    Returns estimated one-way cost as a fraction (e.g., 0.003 = 30bps).
    """
    spread = 0.001 + 0.01 * min(vol_ann, 1.0)
    impact = 0.0
    if avg_dollar_volume > 0 and trade_value > 0:
        participation = min(trade_value / avg_dollar_volume, 0.10)
        impact = 0.5 * vol_ann * (participation ** 0.5)
    fx = FX_COST_PER_LEG if currency not in ("GBP", "GBX") else 0.0
    return spread + impact + fx


def _get_fx_rate(currency: str) -> float:
    """Return the conversion rate from `currency` to GBP."""
    if currency in ("GBP", "GBX"):
        return 1.0

    if currency in _fx_cache:
        return _fx_cache[currency]

    pair = f"{currency}GBP=X"
    try:
        data = yf.download(pair, period="5d", progress=False, auto_adjust=True, timeout=30)
        if data is not None and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            rate = float(data["Close"].dropna().iloc[-1])
            _fx_cache[currency] = rate
            logger.info("FX rate %s->GBP: %.4f", currency, rate)
            return rate
    except Exception as e:
        logger.warning("FX rate fetch failed for %s: %s", currency, e)

    fallbacks = {"USD": 0.79, "EUR": 0.86}
    rate = fallbacks.get(currency, 1.0)
    _fx_cache[currency] = rate
    logger.warning("Using fallback FX rate %s->GBP: %.2f", currency, rate)
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
    rebalance_delta: float
    expected_return: float
    volatility: float
    sharpe_contribution: float
    sector: str
    currency: str
    action: str
    aggregate_score: float
    fx_cost_if_rebalanced: float


@dataclass
class PortfolioAllocation:
    """Complete portfolio optimisation result."""
    holdings: list[HoldingAllocation]
    portfolio_expected_return: float
    portfolio_volatility: float
    portfolio_sharpe: float
    risk_free_rate: float
    method: str
    sector_weights: dict[str, float]
    fx_exposure: dict[str, float]
    turnover: float
    rebalance_trades: list[dict]
    warnings: list[str] = field(default_factory=list)
    # Additive ensemble fields (present only after v2 upgrade; readers must default safely)
    per_method_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    method_weights: dict[str, float] = field(default_factory=dict)
    mu_version: str = "legacy"
    cov_method: str = "lw_ewma"
    history_run_count: int = 0


# ---------------------------------------------------------------------------
# Score→return IC calibration
# ---------------------------------------------------------------------------

def _load_score_to_return_scale(min_rows: int = 30) -> float:
    """Compute the realised cross-sectional slope of return_90d ~ aggregate_score.

    Reads `signal_backtest` rows where `evaluated_90d = 1` and fits an OLS
    slope via closed form.  Returns 0.15 (legacy fallback) when history is
    insufficient.  The slope is capped to the ±0.50 range to prevent a few
    outlier evaluations from blowing up μ.
    """
    cached = _score_ic_cache.get("k")
    if cached is not None and _score_ic_cache.get("n", 0) >= min_rows:
        return cached

    try:
        from engine.discovery_backtest import _connect, init_backtest_db
        init_backtest_db()
        with _connect() as conn:
            rows = conn.execute(
                "SELECT aggregate_score, return_90d "
                "FROM signal_backtest "
                "WHERE evaluated_90d = 1 "
                "  AND return_90d IS NOT NULL "
                "  AND aggregate_score IS NOT NULL"
            ).fetchall()
    except Exception as e:
        logger.debug("Score-IC load failed (%s) — using fallback scale 0.15", e)
        return 0.15

    if not rows or len(rows) < min_rows:
        _score_ic_cache.update(k=0.15, n=len(rows) if rows else 0)
        return 0.15

    x = np.array([float(r[0]) for r in rows], dtype=float)
    y = np.array([float(r[1]) for r in rows], dtype=float)

    # Drop non-finite
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < min_rows:
        _score_ic_cache.update(k=0.15, n=len(x))
        return 0.15

    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = float((x_c * x_c).sum())
    if denom <= 1e-9:
        _score_ic_cache.update(k=0.15, n=len(x))
        return 0.15

    slope = float((x_c * y_c).sum() / denom)
    # Clamp to avoid extreme IC estimates from small/noisy samples
    slope = max(-0.50, min(0.50, slope))
    # Sanity floor: if realised slope is near-zero, keep a small positive scale
    if abs(slope) < 0.02:
        slope = 0.05 * (1.0 if slope >= 0 else -1.0)

    _score_ic_cache.update(k=slope, n=len(x))
    logger.info("Score->90d IC calibrated: slope=%.3f on n=%d", slope, len(x))
    return slope


# ---------------------------------------------------------------------------
# Expected returns
# ---------------------------------------------------------------------------

def _estimate_expected_returns_legacy(
    results: list[dict],
    tickers: list[str],
) -> np.ndarray:
    """Legacy μ: 0.50·MoE + 0.25·score·0.15 + 0.25·hist_90d, annualised."""
    mu = np.zeros(len(tickers))
    result_map = {r["ticker"]: r for r in results}

    for i, ticker in enumerate(tickers):
        r = result_map.get(ticker, {})
        moe_90d = r.get("expected_return_90d")
        if moe_90d is None:
            fc_pct = r.get("forecast_pct_change", 0) or 0
            horizon = r.get("forecast_horizon", getattr(config, "FORECAST_HORIZON_DAYS", 5))
            moe_90d = (fc_pct / 100) * (63 / max(horizon, 1))

        agg = r.get("aggregate_score", 0) or 0
        score_90d = agg * 0.15

        try:
            yahoo_ticker = _resolve_yahoo_symbol(ticker)
            if _is_blocked_symbol(yahoo_ticker):
                raise ValueError("excluded/quarantined ticker")
            data = yf.download(yahoo_ticker, period="120d", progress=False, auto_adjust=True, timeout=30)
            if data is not None and len(data) >= 60:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                hist_90d = (float(data["Close"].dropna().iloc[-1]) /
                            float(data["Close"].dropna().iloc[-63]) - 1)
            else:
                hist_90d = 0.0
        except Exception:
            hist_90d = 0.0

        expected_90d = 0.50 * moe_90d + 0.25 * score_90d + 0.25 * hist_90d
        mu[i] = expected_90d * (252 / 63)

    nan_mask = np.isnan(mu)
    if nan_mask.any():
        mu = np.nan_to_num(mu, nan=0.0)
    return mu


def _estimate_expected_returns_v2(
    results: list[dict],
    tickers: list[str],
    *,
    confidence: bool = True,
) -> np.ndarray:
    """Calibrated μ with James-Stein shrinkage.

    Blend (in 90-day terms, then annualised ×252/63):
        expected_90d = w_moe · MoE_90d + w_score · (aggregate_score · k_ic)

    where:
      - MoE_90d is the 63-trading-day forecast already computed in scoring
      - k_ic is the realised cross-sectional slope from `signal_backtest`
        (falls back to 0.15 when history < 30 rows)
      - The raw-momentum leg is dropped: historical 90-day returns are a
        lagged factor, not a forward expected-return estimate (Jegadeesh &
        Titman 1993).

    Then:
      - μᵢ *= max(0.25, confidence_discountᵢ)    when confidence=True
      - James-Stein shrinkage toward the cross-sectional mean (Jorion 1986;
        Stein 1956), α capped at OPTIMIZER_MU_SHRINKAGE_MAX.
    """
    w_moe, w_score = OPTIMIZER_MU_WEIGHTS
    k_ic = _load_score_to_return_scale()

    result_map = {r["ticker"]: r for r in results}
    mu_90d = np.zeros(len(tickers))

    for i, ticker in enumerate(tickers):
        r = result_map.get(ticker, {})
        moe_90d = r.get("expected_return_90d")
        if moe_90d is None:
            fc_pct = r.get("forecast_pct_change", 0) or 0
            horizon = r.get("forecast_horizon", getattr(config, "FORECAST_HORIZON_DAYS", 5))
            moe_90d = (fc_pct / 100) * (63 / max(horizon, 1))

        agg = r.get("aggregate_score", 0) or 0
        score_90d = agg * k_ic

        expected_90d = w_moe * float(moe_90d) + w_score * float(score_90d)

        if confidence:
            cd = r.get("confidence_discount")
            if cd is not None:
                try:
                    expected_90d *= max(0.25, float(cd))
                except (TypeError, ValueError):
                    pass

        mu_90d[i] = expected_90d

    # Annualise
    mu = mu_90d * (252 / 63)

    # Replace NaN with 0 before shrinkage
    nan_mask = np.isnan(mu)
    if nan_mask.any():
        logger.warning(
            "v2 mu: NaN for %s - replacing with 0",
            [tickers[i] for i in np.where(nan_mask)[0]],
        )
        mu = np.nan_to_num(mu, nan=0.0)

    # James-Stein shrinkage — deferred to post-covariance stage where μ_mvp
    # is available.  See _james_stein_shrink(mu, cov, ...).  Legacy path
    # shrinks toward the cross-sectional arithmetic mean if cov is unknown.
    mu = _james_stein_shrink(mu, alpha_max=OPTIMIZER_MU_SHRINKAGE_MAX)
    return mu


def _james_stein_shrink(
    mu: np.ndarray,
    alpha_max: float = 0.40,
    cov: np.ndarray | None = None,
) -> np.ndarray:
    """Shrink μ toward the minimum-variance-portfolio mean (Jorion 1986).

    Jorion's empirical-Bayes estimator shrinks toward the expected return
    of the global minimum-variance portfolio μ_mvp = (1'Σ⁻¹μ) / (1'Σ⁻¹1),
    not the cross-sectional arithmetic mean.  Shrinking to the arithmetic
    mean over-weights high-μ/high-σ names because it ignores correlation
    structure.  When ``cov`` is not supplied we fall back to the grand
    mean (legacy behaviour) with a debug log.
    """
    n = len(mu)
    if n < 2:
        return mu

    target = None
    if cov is not None and cov.shape == (n, n):
        try:
            ones = np.ones(n)
            inv = np.linalg.pinv(cov)
            num = float(ones @ inv @ mu)
            den = float(ones @ inv @ ones)
            if abs(den) > 1e-12:
                target = num / den
        except Exception:
            target = None
    if target is None:
        target = float(np.mean(mu))
        logger.debug("JS shrinkage: cov not available, falling back to grand-mean target")

    disp = float(np.var(mu))
    if disp <= 1e-12:
        return mu
    alpha = min(alpha_max, 1.0 / (1.0 + disp * n))
    return (1.0 - alpha) * mu + alpha * target


def _estimate_expected_returns(results: list[dict], tickers: list[str]) -> np.ndarray:
    """Dispatcher honouring config.OPTIMIZER_MU_V2 (default True)."""
    if OPTIMIZER_MU_V2:
        return _estimate_expected_returns_v2(results, tickers, confidence=True)
    return _estimate_expected_returns_legacy(results, tickers)


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------

def _gerber_statistic(returns: pd.DataFrame, q: float = 0.5) -> np.ndarray:
    """Gerber co-movement statistic (Gerber, Markowitz, Pujara & Zhu 2022).

    For threshold H_i = q · σᵢ, counts joint up/down exceedances, ignores
    small-move periods, and normalises to a correlation matrix. Then rescales
    to a covariance matrix via the sample volatilities.

    Returns an annualised covariance matrix.
    """
    vals = returns.values
    T, n = vals.shape
    sigmas = np.std(vals, axis=0, ddof=0)
    # Thresholds per column
    H = q * sigmas
    # Avoid division by zero
    H = np.where(H <= 0, 1e-12, H)

    # Signed indicators: +1 if above +H, -1 if below -H, 0 otherwise
    # Shape (T, n)
    sign = np.zeros_like(vals)
    sign[vals > H[None, :]] = 1.0
    sign[vals < -H[None, :]] = -1.0

    # Concordant = sign[:,i]*sign[:,j] == 1  → +1
    # Discordant = sign[:,i]*sign[:,j] == -1 → -1
    # Neither exceeded = 0 (both zero)  → 0 (ignored)
    prod = sign[:, :, None] * sign[:, None, :]  # (T, n, n)
    concordant = (prod == 1).sum(axis=0).astype(float)
    discordant = (prod == -1).sum(axis=0).astype(float)
    total = concordant + discordant
    total = np.where(total <= 0, 1.0, total)

    corr = (concordant - discordant) / total
    # Diagonal should be 1
    np.fill_diagonal(corr, 1.0)

    # Ensure symmetry and clip to valid corr range
    corr = 0.5 * (corr + corr.T)
    corr = np.clip(corr, -1.0, 1.0)

    # Rescale to annualised covariance via daily-vol × 252
    D = np.diag(sigmas) * np.sqrt(252.0)
    cov = D @ corr @ D
    return cov


def _stress_cov_for_bear(cov: np.ndarray, inflate: float) -> np.ndarray:
    """Inflate off-diagonal correlations toward 1 by `inflate` (Longin & Solnik 2001)."""
    if inflate <= 0:
        return cov
    d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    # Only inflate off-diagonals; keep the diagonal at 1
    mask = ~np.eye(corr.shape[0], dtype=bool)
    corr_new = corr.copy()
    corr_new[mask] = corr[mask] + inflate * (1.0 - corr[mask])
    corr_new = np.clip(corr_new, -1.0, 1.0)
    np.fill_diagonal(corr_new, 1.0)
    return np.outer(d, d) * corr_new


def _fetch_fx_series(currency: str, start: str | None = None) -> pd.Series | None:
    """Fetch a daily FX series converting ``currency`` to GBP.

    Returns a pandas Series indexed by date.  Used to FX-harmonise per-ticker
    return series before computing covariance — without this, USD/EUR vols
    and correlations are misattributed to the stock rather than to FX.
    """
    if currency in ("GBP", "GBX"):
        return None
    try:
        pair = f"{currency}GBP=X"
        data = yf.download(
            pair,
            period=f"{LOOKBACK_DAYS + 30}d",
            progress=False,
            auto_adjust=True,
            timeout=30,
        )
        if data is None or data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        s = data["Close"].dropna()
        if s.empty:
            return None
        return s
    except Exception as e:
        logger.debug("FX series fetch failed %s->GBP: %s", currency, e)
        return None


def _gbp_harmonise_closes(
    closes: pd.DataFrame,
    currencies_by_ticker: dict[str, str] | None,
) -> pd.DataFrame:
    """Convert a per-ticker closes DataFrame into GBP-denominated closes.

    * GBP / GBX tickers pass through (GBX converted /100 so returns compose
      with other GBP series; return series is ratio-invariant so the /100
      is a no-op for returns, but kept for definitional clarity).
    * Non-GBP tickers are multiplied by a date-aligned FX spot series
      (ffilled with a 3-day limit to bridge missing FX days).
    * Tickers whose FX series cannot be fetched are dropped with a warning.
    """
    if currencies_by_ticker is None or closes is None or closes.empty:
        return closes

    out_cols: dict[str, pd.Series] = {}
    fx_cache: dict[str, pd.Series] = {}
    for tkr in closes.columns:
        series = closes[tkr]
        if series is None or series.dropna().empty:
            continue
        ccy = (currencies_by_ticker.get(tkr) or "GBP").upper()
        if ccy in ("GBP", "GBX"):
            out_cols[tkr] = series if ccy == "GBP" else (series * 0.01)
            continue
        if ccy not in fx_cache:
            fx = _fetch_fx_series(ccy)
            fx_cache[ccy] = fx if fx is not None else pd.Series(dtype=float)
        fx = fx_cache[ccy]
        if fx.empty:
            logger.warning(
                "Covariance: no FX series for %s (%s) — dropping from cov matrix",
                tkr, ccy,
            )
            continue
        aligned = fx.reindex(series.index).ffill(limit=3).bfill(limit=1)
        out_cols[tkr] = series * aligned
    if not out_cols:
        return closes
    return pd.DataFrame(out_cols).dropna(how="all")


def _estimate_covariance(
    tickers: list[str],
    regime_label: str = "NEUTRAL",
    method: str | None = None,
    currencies: list[str] | None = None,
) -> np.ndarray:
    """Annualised covariance estimate, aligned to `tickers`.

    method:
      - "lw_ewma"  → Ledoit-Wolf shrinkage (60%) blended with EWMA (40%)
      - "gerber"   → Gerber robust co-movement
      - "blend"    → 0.7 · LW_EWMA + 0.3 · Gerber  (default)

    When regime_label == "BEAR", off-diagonal correlations are inflated by
    OPTIMIZER_BEAR_CORR_INFLATE (Longin & Solnik 2001).

    For very small portfolios (n<5), the result is blended 0.8/0.2 with its
    diagonal for stability.  Output is PSD; missing tickers get a diagonal
    fallback (30% vol, zero correlation).
    """
    method = method or OPTIMIZER_COV_METHOD
    n = len(tickers)
    FALLBACK_VAR = 0.30 ** 2

    try:
        download_symbols, symbol_to_originals = _yahoo_download_map(tickers)
        if not download_symbols:
            raise ValueError("No active Yahoo symbols after alias/quarantine filtering")
        data = yf.download(download_symbols, period=f"{LOOKBACK_DAYS}d", progress=False, auto_adjust=True)
        if data is None or data.empty:
            raise ValueError("No price data")

        closes = _close_frame_from_yahoo_download(data, download_symbols, symbol_to_originals)
        if closes.empty:
            raise ValueError("No close price data")

        available = [t for t in tickers if t in closes.columns and not closes[t].isna().all()]
        missing = [t for t in tickers if t not in available]
        if missing:
            logger.warning("Covariance: missing price data for %s — using diagonal fallback for these", missing)

        if not available:
            raise ValueError("No tickers had price data")

        closes = closes[available]

        # --- FX-harmonise to GBP before computing returns ---
        # Currencies passed in from the caller; without them we fall back to
        # raw local-currency returns (legacy behaviour, logged for clarity).
        if currencies is not None:
            ccy_map = {t: c for t, c in zip(tickers, currencies)}
            try:
                closes = _gbp_harmonise_closes(closes, ccy_map)
                # Some tickers may have been dropped due to missing FX series.
                available = [t for t in available if t in closes.columns]
                if not available:
                    raise ValueError("All tickers dropped after FX harmonisation")
            except Exception as _fx_e:
                logger.warning("FX harmonisation failed (%s) — using local-ccy returns", _fx_e)
        else:
            logger.debug("Covariance: no currencies supplied — local-ccy returns (biased for multi-ccy books)")

        # Align calendars without dropping every pan-market holiday row:
        # per-series ffill(1) lets LSE/SPX holidays coexist, then drop rows
        # where every series is NaN.
        closes = closes.ffill(limit=1).dropna(how="all")
        daily_returns = closes.pct_change(fill_method=None).dropna(how="all")
        # Any residual NaN becomes zero-return for that day (equiv. to a
        # half-day for that venue); strictly dominates dropping the row.
        daily_returns = daily_returns.fillna(0.0)

        if len(daily_returns) < 30:
            raise ValueError("Insufficient return data")

        # --- Ledoit-Wolf + EWMA blend ---
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(daily_returns.values)
            lw_cov = lw.covariance_ * 252
            logger.debug("Covariance: Ledoit-Wolf shrinkage=%.3f", lw.shrinkage_)
        except Exception:
            lw_cov = daily_returns.cov().values * 252
            shrinkage = 0.3
            diag = np.diag(np.diag(lw_cov))
            lw_cov = (1 - shrinkage) * lw_cov + shrinkage * diag

        try:
            ewma_halflife = 42
            ewma_returns = daily_returns.ewm(halflife=ewma_halflife).cov()
            n_avail = len(available)
            last_idx = ewma_returns.index[-n_avail:]
            ewma_cov = ewma_returns.loc[last_idx].values * 252
            if ewma_cov.shape == lw_cov.shape:
                lw_ewma_cov = 0.60 * lw_cov + 0.40 * ewma_cov
            else:
                lw_ewma_cov = lw_cov
        except Exception:
            lw_ewma_cov = lw_cov

        # --- Method selection ---
        if method == "lw_ewma":
            avail_cov = lw_ewma_cov
        elif method == "gerber":
            try:
                avail_cov = _gerber_statistic(daily_returns)
            except Exception as e:
                logger.warning("Gerber estimation failed (%s) — falling back to LW+EWMA", e)
                avail_cov = lw_ewma_cov
        else:  # "blend" (default)
            try:
                gerber_cov = _gerber_statistic(daily_returns)
                avail_cov = 0.70 * lw_ewma_cov + 0.30 * gerber_cov
            except Exception as e:
                logger.warning("Gerber blend failed (%s) — using LW+EWMA only", e)
                avail_cov = lw_ewma_cov

        # --- Align to original ticker list (diagonal fallback for missing) ---
        avail_idx = {t: i for i, t in enumerate(available)}
        cov = np.eye(n) * FALLBACK_VAR
        for i, ti in enumerate(tickers):
            for j, tj in enumerate(tickers):
                if ti in avail_idx and tj in avail_idx:
                    cov[i, j] = avail_cov[avail_idx[ti], avail_idx[tj]]

        # --- Regime stress ---
        if regime_label == "BEAR":
            cov = _stress_cov_for_bear(cov, OPTIMIZER_BEAR_CORR_INFLATE)

        # --- Small-portfolio stabilisation ---
        if n < 5:
            diag = np.diag(np.diag(cov))
            cov = 0.8 * cov + 0.2 * diag

        # --- PSD enforcement ---
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-8)
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return cov

    except Exception as e:
        logger.warning("Covariance estimation failed (%s), using diagonal fallback", e)
        return np.eye(n) * FALLBACK_VAR


# ---------------------------------------------------------------------------
# Current weights & sectors
# ---------------------------------------------------------------------------

def _current_weights(holdings: list[dict], results: list[dict]) -> np.ndarray:
    """Calculate current portfolio weights from market values (all converted to GBP)."""
    result_map = {r["ticker"]: r for r in results}
    values = []
    for h in holdings:
        ticker = h["ticker"]
        r = result_map.get(ticker, {})
        price = r.get("current_price", 0) or 0
        if price != price:
            price = 0
        qty = h.get("quantity", 0)
        currency = h.get("currency", "GBP")
        value = price * qty
        if currency == "GBX":
            value /= 100
        fx = _get_fx_rate(currency)
        value *= fx
        values.append(value)
    total = sum(values) or 1
    return np.array([v / total for v in values])


def _get_sectors(results: list[dict], tickers: list[str]) -> dict[str, str]:
    """Map tickers to sectors from analysis results or yfinance."""
    result_map = {r["ticker"]: r for r in results}
    sectors = {}
    for ticker in tickers:
        r = result_map.get(ticker, {})
        sector = r.get("sector")
        if not sector:
            try:
                from utils.data_fetch import get_ticker_info

                info = get_ticker_info(ticker)
                sector = info.get("sector", "Unknown")
            except Exception:
                sector = "Unknown"
        sectors[ticker] = sector
    return sectors


# ---------------------------------------------------------------------------
# Shared bounds + common-constraint projection
# ---------------------------------------------------------------------------

def _build_bounds(
    tickers: list[str],
    results: list[dict],
) -> list[tuple[float, float]]:
    """Per-ticker (min, max) bounds respecting Kelly caps and action tiers."""
    result_map = {r["ticker"]: r for r in results}

    # Kelly caps (opt-in, requires enough backtest history)
    kelly_fractions: dict[str, float] = {}
    try:
        from engine.discovery_backtest import get_kelly_fractions
        kelly_fractions = get_kelly_fractions(source="portfolio") or {}
    except Exception:
        kelly_fractions = {}

    bounds: list[tuple[float, float]] = []
    for ticker in tickers:
        r = result_map.get(ticker, {})
        action = r.get("action", "KEEP")
        scale = r.get("max_weight_scale", 1.0) or 1.0
        upper = MAX_WEIGHT * float(scale)

        if kelly_fractions and action in kelly_fractions:
            upper = min(upper, float(kelly_fractions[action]))

        if action == "STRONG SELL":
            upper = MIN_WEIGHT
        elif action == "SELL":
            upper = min(upper, MIN_WEIGHT * 2)

        bounds.append((MIN_WEIGHT, max(MIN_WEIGHT, upper)))
    return bounds


def _rebalance_with_bounds(
    w: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    target: float = 1.0,
    frozen: np.ndarray | None = None,
) -> np.ndarray:
    """Move a vector back to the target sum while respecting per-name bounds."""
    w = np.clip(np.asarray(w, dtype=float), lower, upper)
    if frozen is None:
        frozen = np.zeros(len(w), dtype=bool)
    else:
        frozen = np.asarray(frozen, dtype=bool)

    for _ in range(100):
        diff = float(target - w.sum())
        if abs(diff) < 1e-8:
            break

        if diff > 0:
            room = np.clip(upper - w, 0.0, None)
        else:
            room = np.clip(w - lower, 0.0, None)
        room[frozen] = 0.0

        total_room = float(room.sum())
        if total_room <= 1e-12:
            break

        step = room / total_room
        if diff > 0:
            w = w + diff * step
        else:
            w = w - (-diff) * step
        w = np.clip(w, lower, upper)

    return np.clip(w, lower, upper)


def _redistribute_difference(
    w: np.ndarray,
    diff: float,
    lower: np.ndarray,
    upper: np.ndarray,
    sectors_map: dict[str, list[int]],
) -> np.ndarray:
    """Add or remove weight while respecting per-name and sector caps."""
    w = np.clip(np.asarray(w, dtype=float), lower, upper)
    tol = 1e-10

    if abs(diff) < tol:
        return w

    if diff < 0:
        remaining = -diff
        for _ in range(100):
            slack = np.clip(w - lower, 0.0, None)
            total_slack = float(slack.sum())
            if remaining <= tol or total_slack <= tol:
                break
            take = min(remaining, total_slack)
            w = w - take * (slack / total_slack)
            w = np.clip(w, lower, upper)
            remaining -= take
        return w

    remaining = diff
    for _ in range(100):
        if remaining <= tol:
            break

        prev_sum = float(w.sum())
        sector_rooms: dict[str, float] = {}
        total_room = 0.0
        for sector, idxs in sectors_map.items():
            idx = np.array(idxs, dtype=int)
            sector_sum = float(w[idx].sum())
            sector_room = max(0.0, SECTOR_CAP - sector_sum)
            asset_room = float(np.clip(upper[idx] - w[idx], 0.0, None).sum())
            room = min(sector_room, asset_room)
            if room > tol:
                sector_rooms[sector] = room
                total_room += room

        if total_room <= tol:
            break

        step = min(remaining, total_room)
        for sector, room in sector_rooms.items():
            idx = np.array(sectors_map[sector], dtype=int)
            sector_add = step * room / total_room
            headroom = np.clip(upper[idx] - w[idx], 0.0, None)
            headroom_total = float(headroom.sum())
            if headroom_total <= tol:
                continue
            w[idx] = w[idx] + sector_add * (headroom / headroom_total)

        w = np.clip(w, lower, upper)
        added = max(float(w.sum()) - prev_sum, 0.0)
        remaining = max(remaining - added, 0.0)

    return np.clip(w, lower, upper)


def _apply_common_constraints(
    w: np.ndarray,
    bounds: list[tuple[float, float]],
    tickers: list[str],
    results: list[dict],
) -> np.ndarray:
    """Project any weight vector onto the shared feasible set.

    Iteratively clips to per-ticker bounds, re-normalises to sum=1, and
    reduces over-concentrated sectors to the sector cap.  Converges quickly
    for a small number of holdings.
    """
    n = len(w)
    if n == 0:
        return w
    w = np.asarray(w, dtype=float).copy()
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)

    if lower.sum() > 1.0 + 1e-8 or upper.sum() < 1.0 - 1e-8:
        logger.warning(
            "Optimizer bounds are infeasible (sum lower=%.3f, sum upper=%.3f); falling back to midpoint projection",
            lower.sum(),
            upper.sum(),
        )
        mid = np.clip(0.5 * (lower + upper), lower, upper)
        total = float(mid.sum())
        return mid / total if total > 0 else np.full(n, 1.0 / n)

    # Replace any NaN with the mid of the bounds
    for i in range(n):
        if not np.isfinite(w[i]):
            w[i] = 0.5 * (lower[i] + upper[i])

    sectors_map = _sector_index(tickers, results)

    # Make non-negative before normalisation
    w = np.maximum(w, 0.0)
    if w.sum() <= 0:
        w = lower.copy()
    else:
        w = w / w.sum()
        w = np.clip(w, lower, upper)

    total_diff = 1.0 - float(w.sum())
    if abs(total_diff) > 1e-8:
        w = _redistribute_difference(w, total_diff, lower, upper, sectors_map)

    for _ in range(100):
        prev = w.copy()
        released = 0.0

        for sector, idxs in sectors_map.items():
            idx = np.array(idxs, dtype=int)
            sector_sum = float(w[idx].sum())
            if sector_sum <= SECTOR_CAP + 1e-8:
                continue

            excess = sector_sum - SECTOR_CAP
            reducible = np.clip(w[idx] - lower[idx], 0.0, None)
            reducible_total = float(reducible.sum())
            if reducible_total <= 1e-12:
                continue

            take = min(excess, reducible_total)
            w[idx] = w[idx] - take * (reducible / reducible_total)
            released += take

        if released > 1e-8:
            w = _redistribute_difference(w, released, lower, upper, sectors_map)

        total_diff = 1.0 - float(w.sum())
        if abs(total_diff) > 1e-8:
            w = _redistribute_difference(w, total_diff, lower, upper, sectors_map)

        if np.max(np.abs(w - prev)) < 1e-7:
            break

    w = np.clip(w, lower, upper)
    total_diff = 1.0 - float(w.sum())
    if abs(total_diff) > 1e-8:
        w = _redistribute_difference(w, total_diff, lower, upper, sectors_map)
    return np.clip(w, lower, upper)


def _sector_index(tickers: list[str], results: list[dict]) -> dict[str, list[int]]:
    """Map each sector to the list of ticker indices belonging to it."""
    sectors = _get_sectors(results, tickers)
    idx: dict[str, list[int]] = {}
    for i, t in enumerate(tickers):
        s = sectors.get(t, "Unknown")
        idx.setdefault(s, []).append(i)
    return idx


def _inverse_vol_weights(cov: np.ndarray) -> np.ndarray:
    """1/σ weights, normalised. Safe fallback for any method."""
    diag = np.diag(cov)
    vols = np.sqrt(np.clip(diag, 1e-12, None))
    inv = 1.0 / vols
    return inv / inv.sum()


# ---------------------------------------------------------------------------
# Method 1: Mean-Variance
# ---------------------------------------------------------------------------

def _mean_variance_optimize(
    mu: np.ndarray,
    cov: np.ndarray,
    current_w: np.ndarray,
    tickers: list[str],
    currencies: list[str],
    bounds: list[tuple[float, float]],
    sector_groups: dict[str, list[int]],
    per_ticker_cost: np.ndarray,
    risk_aversion: float = RISK_AVERSION,
) -> tuple[np.ndarray, list[str], dict]:
    """Markowitz MVO via SLSQP.

    max μ'w − (λ/2)·w'Σw − turnover_cost − fx_cost
    s.t. sum(w)=1, bounds, sector caps.
    """
    n = len(tickers)
    warnings: list[str] = []

    fx_cost = np.array([
        FX_COST_PER_LEG if c not in ("GBP", "GBX") else 0.0
        for c in currencies
    ])

    def objective(w):
        ret = mu @ w
        risk = w @ cov @ w
        delta_w = np.abs(w - current_w)
        turnover_cost = np.sum(per_ticker_cost * delta_w)
        fx_rebal_cost = np.sum(fx_cost * delta_w)
        return -(ret - (risk_aversion / 2) * risk - turnover_cost - fx_rebal_cost)

    constraints: list[dict] = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    ]
    for sector, idxs in sector_groups.items():
        if idxs:
            constraints.append({
                "type": "ineq",
                "fun": lambda w, idx=idxs: SECTOR_CAP - float(np.sum(w[idx])),
            })

    # Small ridge to avoid singular SLSQP Hessians
    cov_reg = cov + np.eye(n) * 1e-6

    # Warm start from current weights, clipped
    w0 = np.where(np.isnan(current_w), 1.0 / n, current_w)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    w0 = np.clip(w0, lo, hi)
    s = w0.sum()
    w0 = w0 / s if s > 0 else np.full(n, 1.0 / n)

    def obj_reg(w):
        ret = mu @ w
        risk = w @ cov_reg @ w
        delta_w = np.abs(w - current_w)
        turnover_cost = np.sum(per_ticker_cost * delta_w)
        fx_rebal_cost = np.sum(fx_cost * delta_w)
        return -(ret - (risk_aversion / 2) * risk - turnover_cost - fx_rebal_cost)

    result = minimize(
        obj_reg, w0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if result.success:
        w = np.maximum(result.x, 0.0)
        s = w.sum()
        w = w / s if s > 0 else np.full(n, 1.0 / n)
    else:
        warnings.append(f"MV did not converge: {result.message}")
        logger.warning("MV optimisation failed: %s — equal-weight fallback", result.message)
        w = np.full(n, 1.0 / n)

    return w, warnings, {"solver_success": bool(result.success)}


# ---------------------------------------------------------------------------
# Method 2: Minimum Variance (Jagannathan & Ma 2003)
# ---------------------------------------------------------------------------

def _min_variance_optimize(
    cov: np.ndarray,
    bounds: list[tuple[float, float]],
    sector_groups: dict[str, list[int]],
) -> tuple[np.ndarray, list[str], dict]:
    """Long-only minimum-variance portfolio."""
    n = cov.shape[0]
    warnings: list[str] = []

    cov_reg = cov + np.eye(n) * 1e-6
    w0 = _inverse_vol_weights(cov_reg)

    constraints: list[dict] = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    ]
    for sector, idxs in sector_groups.items():
        if idxs:
            constraints.append({
                "type": "ineq",
                "fun": lambda w, idx=idxs: SECTOR_CAP - float(np.sum(w[idx])),
            })

    result = minimize(
        lambda w: float(w @ cov_reg @ w),
        w0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if result.success:
        w = np.maximum(result.x, 0.0)
        s = w.sum()
        w = w / s if s > 0 else _inverse_vol_weights(cov_reg)
    else:
        warnings.append(f"MinVar did not converge: {result.message}")
        w = _inverse_vol_weights(cov_reg)
    return w, warnings, {"solver_success": bool(result.success)}


# ---------------------------------------------------------------------------
# Method 3: Equal Risk Contribution / Risk Parity
# ---------------------------------------------------------------------------

def _risk_parity_optimize(
    cov: np.ndarray,
    bounds: list[tuple[float, float]],
    sector_groups: dict[str, list[int]],
) -> tuple[np.ndarray, list[str], dict]:
    """Minimise squared deviations of per-asset risk contributions from 1/n."""
    n = cov.shape[0]
    warnings: list[str] = []
    cov_reg = cov + np.eye(n) * 1e-6
    w0 = _inverse_vol_weights(cov_reg)

    def objective(w):
        port_var = float(w @ cov_reg @ w)
        if port_var <= 0:
            return 0.0
        mrc = cov_reg @ w            # marginal risk contributions
        rc = w * mrc                 # per-asset contribution to variance
        target = port_var / n
        return float(np.sum((rc - target) ** 2))

    constraints: list[dict] = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    ]
    for sector, idxs in sector_groups.items():
        if idxs:
            constraints.append({
                "type": "ineq",
                "fun": lambda w, idx=idxs: SECTOR_CAP - float(np.sum(w[idx])),
            })

    result = minimize(
        objective, w0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if result.success:
        w = np.maximum(result.x, 0.0)
        s = w.sum()
        w = w / s if s > 0 else _inverse_vol_weights(cov_reg)
    else:
        warnings.append(f"RiskParity did not converge: {result.message}")
        w = _inverse_vol_weights(cov_reg)
    return w, warnings, {"solver_success": bool(result.success)}


# ---------------------------------------------------------------------------
# Method 4: Black-Litterman
# ---------------------------------------------------------------------------

def _black_litterman_optimize(
    mu_views: np.ndarray,
    cov: np.ndarray,
    current_w: np.ndarray,
    tickers: list[str],
    currencies: list[str],
    bounds: list[tuple[float, float]],
    sector_groups: dict[str, list[int]],
    per_ticker_cost: np.ndarray,
    results: list[dict],
    risk_aversion: float = RISK_AVERSION,
    tau: float = None,
) -> tuple[np.ndarray, list[str], dict]:
    """Posterior μ fusion, then MVO with the posterior.

    Prior: inverse-vol-based reverse-optimisation Π = λ·Σ·w_prior.
    Views: full-rank identity views (`P = I`, `Q = mu_views`).
    View uncertainty Ω widens when the holding's confidence_discount is low.
    """
    if tau is None:
        tau = OPTIMIZER_BL_TAU
    warnings: list[str] = []
    try:
        mu_bl, pi, omega_diag = _black_litterman_posterior(
            mu_views,
            cov,
            tickers,
            results,
            risk_aversion=risk_aversion,
            tau=tau,
        )
    except Exception as e:
        warnings.append(f"BL posterior failed ({e}); falling back to views μ")
        pi = risk_aversion * (cov @ _market_cap_prior_weights(tickers, results, cov))
        omega_diag = np.clip(tau * np.diag(cov), 1e-9, None)
        mu_bl = mu_views

    # MVO with the posterior
    w, sub_warn, sub_diag = _mean_variance_optimize(
        mu_bl, cov, current_w, tickers, currencies, bounds, sector_groups,
        per_ticker_cost, risk_aversion=risk_aversion,
    )
    warnings.extend(sub_warn)
    return w, warnings, {
        "bl_tau": tau,
        "mv_success": sub_diag.get("solver_success"),
        "prior_mean": float(np.mean(pi)),
        "view_uncertainty_mean": float(np.mean(omega_diag)),
    }


def _market_cap_prior_weights(
    tickers: list[str],
    results: list[dict],
    cov: np.ndarray,
) -> np.ndarray:
    """Market-cap-proportional prior weights (CAPM equilibrium).

    Black & Litterman (1992) requires a *neutral* prior — the
    market-clearing weights of the investable universe, which under CAPM
    correspond to market-cap weights.  When market-cap data is missing we
    fall back to 1/N (true neutral) before resorting to inverse-vol.

    The previous implementation used inverse-vol weights as the prior,
    which baked a risk-parity tilt into the Bayesian prior and then
    combined it with views and MVO — a triple-application of risk
    aversion that broke the BL posterior interpretation.
    """
    n = len(tickers)
    result_map = {r["ticker"]: r for r in results}
    caps = np.zeros(n, dtype=float)
    for i, t in enumerate(tickers):
        r = result_map.get(t, {})
        mc = r.get("market_cap") or r.get("_market_cap") or 0
        try:
            mc = float(mc) if mc else 0.0
        except (TypeError, ValueError):
            mc = 0.0
        caps[i] = max(mc, 0.0)
    total = float(caps.sum())
    if total > 0 and np.isfinite(total):
        return caps / total
    # Equal-weight fallback (true neutral prior)
    logger.debug("BL prior: market-cap data unavailable — falling back to 1/N neutral prior")
    return np.full(n, 1.0 / n)


def _black_litterman_posterior(
    mu_views: np.ndarray,
    cov: np.ndarray,
    tickers: list[str],
    results: list[dict],
    *,
    risk_aversion: float = RISK_AVERSION,
    tau: float | None = None,
    confidence_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Black-Litterman posterior mean, prior Π, and Ω diagonal."""
    if tau is None:
        tau = OPTIMIZER_BL_TAU

    n = len(tickers)
    result_map = {r["ticker"]: r for r in results}

    # CAPM-consistent equilibrium prior via market-cap weights.
    w_prior = _market_cap_prior_weights(tickers, results, cov)
    pi = risk_aversion * (cov @ w_prior)

    if confidence_override is None:
        confidence = np.array(
            [float(result_map.get(t, {}).get("confidence_discount") or 1.0) for t in tickers],
            dtype=float,
        )
        confidence = np.clip(confidence, 0.2, 1.0)
    else:
        confidence = np.asarray(confidence_override, dtype=float)
        if confidence.shape != (n,):
            raise ValueError("confidence_override must match tickers length")
        confidence = np.clip(confidence, 1e-9, None)

    omega_diag = np.clip(tau * np.diag(cov), 1e-9, None) / confidence
    Omega = np.diag(omega_diag)

    tau_sigma = tau * cov
    inv_tau_sigma = np.linalg.pinv(tau_sigma)
    inv_omega = np.linalg.pinv(Omega)
    A = inv_tau_sigma + inv_omega
    b = inv_tau_sigma @ pi + inv_omega @ mu_views
    mu_bl = np.linalg.pinv(A) @ b
    return mu_bl, pi, omega_diag


# ---------------------------------------------------------------------------
# Method 5: Hierarchical Risk Parity (López de Prado 2016)
# ---------------------------------------------------------------------------

def _hrp_optimize(
    cov: np.ndarray,
    tickers: list[str],
) -> tuple[np.ndarray, list[str], dict]:
    """Hierarchical Risk Parity — no matrix inversion, robust for any n."""
    warnings: list[str] = []
    n = cov.shape[0]
    if n < 3:
        return _inverse_vol_weights(cov), ["HRP: n<3 -> inverse-vol fallback"], {}

    try:
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform
    except Exception as e:
        warnings.append(f"HRP unavailable ({e}) -> inverse-vol fallback")
        return _inverse_vol_weights(cov), warnings, {}

    try:
        d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        corr = cov / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1.0, 1.0)
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
        # scipy expects a condensed distance matrix
        cond = squareform(dist, checks=False)
        Z = linkage(cond, method="single")
        order = _hrp_quasi_diag(Z)
        w = _hrp_recursive_bisect(cov, order)
    except Exception as e:
        warnings.append(f"HRP failed ({e}) -> inverse-vol fallback")
        return _inverse_vol_weights(cov), warnings, {}

    w = np.maximum(w, 0.0)
    s = w.sum()
    w = w / s if s > 0 else _inverse_vol_weights(cov)
    return w, warnings, {}


def _hrp_quasi_diag(link: np.ndarray) -> list[int]:
    """Return the leaf ordering that quasi-diagonalises the correlation matrix."""
    link = link.astype(int)
    n = link.shape[0] + 1
    # Start from the last merge
    sort_ix = [link[-1, 0], link[-1, 1]]
    num_items = n
    while max(sort_ix) >= num_items:
        new = []
        for i in sort_ix:
            if i >= num_items:
                idx = i - num_items
                new.extend([int(link[idx, 0]), int(link[idx, 1])])
            else:
                new.append(int(i))
        sort_ix = new
    return sort_ix


def _hrp_recursive_bisect(cov: np.ndarray, order: list[int]) -> np.ndarray:
    """Inverse-variance weighted recursive bisection along the leaf order."""
    n = cov.shape[0]
    w = np.ones(n)
    clusters: list[list[int]] = [list(order)]
    while clusters:
        new_clusters: list[list[int]] = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            mid = len(cl) // 2
            left = cl[:mid]
            right = cl[mid:]
            var_l = _cluster_var(cov, left)
            var_r = _cluster_var(cov, right)
            total = var_l + var_r
            if total <= 0:
                alpha = 0.5
            else:
                alpha = 1.0 - var_l / total
            for i in left:
                w[i] *= alpha
            for i in right:
                w[i] *= 1.0 - alpha
            new_clusters.append(left)
            new_clusters.append(right)
        clusters = new_clusters
    return w


def _cluster_var(cov: np.ndarray, idx: list[int]) -> float:
    """Inverse-variance weighted portfolio variance for a cluster."""
    sub = cov[np.ix_(idx, idx)]
    iv = 1.0 / np.clip(np.diag(sub), 1e-12, None)
    iv = iv / iv.sum()
    return float(iv @ sub @ iv)


# ---------------------------------------------------------------------------
# Ensemble combiner + history
# ---------------------------------------------------------------------------

def _load_optimizer_history() -> dict:
    """Load the ensemble method history file (empty skeleton on error)."""
    default = {"version": 1, "method_stats": {}, "runs": []}
    try:
        if not os.path.exists(OPTIMIZER_HISTORY_PATH):
            return default
        with open(OPTIMIZER_HISTORY_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return default
        data.setdefault("version", 1)
        data.setdefault("method_stats", {})
        data.setdefault("runs", [])
        return data
    except Exception as e:
        logger.warning("optimizer_history load failed (%s) — using empty", e)
        return default


def _save_optimizer_history(data: dict) -> None:
    """Persist the ensemble history atomically."""
    try:
        from utils.atomic_io import atomic_write_json
        atomic_write_json(OPTIMIZER_HISTORY_PATH, data, indent=2)
    except Exception as e:
        logger.warning("optimizer_history save failed: %s", e)


def _backfill_realised_sharpe(history: dict, *, min_age_days: int = 90) -> bool:
    """Score historical method allocations against realised returns.

    Bates & Granger (1969) combinations require *out-of-sample* loss signal —
    without it the softmax over ``method_stats`` degenerates to equal weights.
    For every run older than ``min_age_days`` with ``realised_90d is None``
    we fetch the next 90 trading days, compute each method's realised
    annualised Sharpe, and update ``method_stats`` in place.  Silent on any
    download failure (the caller falls back to equal weights).
    """
    runs = history.get("runs", []) or []
    if not runs:
        return False
    stats = history.setdefault("method_stats", {})
    updated = False
    cutoff = datetime.now() - timedelta(days=min_age_days)
    try:
        import yfinance as _yf
    except Exception:
        return False

    for rec in runs:
        if rec.get("realised_90d") is not None:
            continue
        try:
            run_dt = datetime.fromisoformat(str(rec.get("date"))[:19])
        except (TypeError, ValueError):
            continue
        if run_dt > cutoff:
            continue
        tickers = list(rec.get("tickers") or [])
        per_method = rec.get("per_method_weights") or {}
        if not tickers or not per_method:
            rec["realised_90d"] = {}
            continue
        try:
            download_symbols, symbol_to_originals = _yahoo_download_map(tickers)
            if not download_symbols:
                rec["realised_90d"] = {}
                continue
            px = _yf.download(
                download_symbols, start=run_dt.date(),
                end=(run_dt + timedelta(days=min_age_days + 5)).date(),
                progress=False, auto_adjust=True, group_by="ticker",
            )
        except Exception:
            continue
        if px is None or len(px) < 20:
            continue

        closes = {}
        close_frame = _close_frame_from_yahoo_download(px, download_symbols, symbol_to_originals)
        for t in tickers:
            if t not in close_frame:
                continue
            s = close_frame[t].dropna()
            if not s.empty:
                closes[t] = s
        if not closes:
            continue

        realised: dict[str, float] = {}
        for method, w_map in per_method.items():
            port_rets = []
            for t, w in w_map.items():
                s = closes.get(t)
                if s is None or len(s) < 2:
                    continue
                r = s.pct_change().dropna().values
                if r.size == 0:
                    continue
                port_rets.append((float(w), r))
            if not port_rets:
                continue
            min_len = min(len(r) for _, r in port_rets)
            if min_len < 20:
                continue
            pr = np.zeros(min_len)
            for w, r in port_rets:
                pr = pr + w * r[:min_len]
            mu = float(np.mean(pr)); sd = float(np.std(pr))
            sharpe = (mu / sd) * np.sqrt(252.0) if sd > 1e-8 else 0.0
            realised[method] = float(sharpe)
            s_entry = stats.setdefault(method, {"n": 0, "mean_sharpe": 0.0})
            n_prev = int(s_entry.get("n", 0))
            mean_prev = float(s_entry.get("mean_sharpe", 0.0))
            n_new = n_prev + 1
            s_entry["n"] = n_new
            s_entry["mean_sharpe"] = (mean_prev * n_prev + sharpe) / n_new
        if realised:
            rec["realised_90d"] = realised
            updated = True
    if updated:
        _save_optimizer_history(history)
    return updated


def _ensemble_method_weights(history: dict, methods: list[str]) -> dict[str, float]:
    """Softmax over realised mean Sharpe per method.

    Falls back to equal weights if we have fewer than 5 runs.  Floor and cap
    per-method weight at OPTIMIZER_METHOD_WEIGHT_MIN / _MAX, then renormalise.
    """
    # Update method_stats from any eligible (≥90d old) runs before weighting.
    try:
        _backfill_realised_sharpe(history)
    except Exception as e:
        logger.debug("optimizer realised-sharpe backfill failed: %s", e)
    runs = history.get("runs", [])
    stats = history.get("method_stats", {}) or {}
    if len(runs) < 5 or not stats:
        eq = 1.0 / max(len(methods), 1)
        return {m: eq for m in methods}

    tau = max(OPTIMIZER_ENSEMBLE_TEMPERATURE, 1e-6)
    weights: dict[str, float] = {}
    for m in methods:
        ms = stats.get(m) or {}
        if int(ms.get("n", 0)) < 3:
            weights[m] = 1.0
        else:
            sh = float(ms.get("mean_sharpe", 0.0) or 0.0)
            weights[m] = float(np.exp(sh / tau))

    total = sum(weights.values()) or 1.0
    weights = {m: v / total for m, v in weights.items()}
    # Floor + cap + renormalise
    lo, hi = OPTIMIZER_METHOD_WEIGHT_MIN, OPTIMIZER_METHOD_WEIGHT_MAX
    weights = {m: max(lo, min(hi, v)) for m, v in weights.items()}
    total = sum(weights.values()) or 1.0
    return {m: v / total for m, v in weights.items()}


def _ensemble_combine(
    per_method_w: dict[str, np.ndarray],
    method_weights: dict[str, float],
    bounds: list[tuple[float, float]],
    tickers: list[str],
    results: list[dict],
) -> np.ndarray:
    """Weighted-average the method vectors, then project onto constraints."""
    if not per_method_w:
        n = len(tickers)
        return np.full(n, 1.0 / n)

    n = len(tickers)
    w = np.zeros(n)
    total = 0.0
    for method, vec in per_method_w.items():
        mw = float(method_weights.get(method, 0.0))
        if mw <= 0 or vec is None:
            continue
        v = np.asarray(vec, dtype=float)
        if v.shape != (n,) or not np.isfinite(v).all():
            continue
        w = w + mw * v
        total += mw

    if total <= 0:
        return np.full(n, 1.0 / n)
    w = w / total
    return _apply_common_constraints(w, bounds, tickers, results)


def _record_optimizer_run(per_method_w: dict[str, np.ndarray], tickers: list[str]) -> None:
    """Append the current run to the history file."""
    try:
        hist = _load_optimizer_history()
        record = {
            "date": datetime.now().isoformat(),
            "tickers": list(tickers),
            "per_method_weights": {
                m: {t: float(v[i]) for i, t in enumerate(tickers)}
                for m, v in per_method_w.items()
                if v is not None and len(v) == len(tickers)
            },
            "realised_90d": None,
        }
        runs = hist.get("runs", [])
        runs.append(record)
        # Keep only the last 200 runs
        hist["runs"] = runs[-200:]
        _save_optimizer_history(hist)
    except Exception as e:
        logger.warning("Failed to record optimizer run: %s", e)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

OptimizerMethod = Callable[..., tuple[np.ndarray, list[str], dict]]


def _method_mean_variance(**kwargs) -> tuple[np.ndarray, list[str], dict]:
    return _mean_variance_optimize(
        kwargs["mu"],
        kwargs["cov"],
        kwargs["current_w"],
        kwargs["tickers"],
        kwargs["currencies"],
        kwargs["bounds"],
        kwargs["sector_groups"],
        kwargs["per_ticker_cost"],
        risk_aversion=kwargs["risk_aversion"],
    )


def _method_min_variance(**kwargs) -> tuple[np.ndarray, list[str], dict]:
    return _min_variance_optimize(
        kwargs["cov"],
        kwargs["bounds"],
        kwargs["sector_groups"],
    )


def _method_risk_parity(**kwargs) -> tuple[np.ndarray, list[str], dict]:
    return _risk_parity_optimize(
        kwargs["cov"],
        kwargs["bounds"],
        kwargs["sector_groups"],
    )


def _method_black_litterman(**kwargs) -> tuple[np.ndarray, list[str], dict]:
    return _black_litterman_optimize(
        kwargs["mu"],
        kwargs["cov"],
        kwargs["current_w"],
        kwargs["tickers"],
        kwargs["currencies"],
        kwargs["bounds"],
        kwargs["sector_groups"],
        kwargs["per_ticker_cost"],
        kwargs["results"],
        risk_aversion=kwargs["risk_aversion"],
    )


def _method_hrp(**kwargs) -> tuple[np.ndarray, list[str], dict]:
    return _hrp_optimize(
        kwargs["cov"],
        kwargs["tickers"],
    )


OPTIMIZER_METHODS: dict[str, OptimizerMethod] = {
    "mean_variance": _method_mean_variance,
    "min_variance": _method_min_variance,
    "risk_parity": _method_risk_parity,
    "black_litterman": _method_black_litterman,
    "hrp": _method_hrp,
}


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
    trades = []
    for i, ticker in enumerate(tickers):
        delta = optimal_w[i] - current_w[i]
        if abs(delta) < 0.02:
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
    trades.sort(key=lambda t: abs(t["delta_pct"]), reverse=True)
    return trades


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _run_single_method(
    method: str,
    *,
    mu: np.ndarray,
    cov: np.ndarray,
    current_w: np.ndarray,
    tickers: list[str],
    currencies: list[str],
    bounds: list[tuple[float, float]],
    sector_groups: dict[str, list[int]],
    per_ticker_cost: np.ndarray,
    results: list[dict],
    risk_aversion: float,
) -> tuple[Optional[np.ndarray], list[str], dict]:
    """Dispatch to a single method. Errors become warnings, weights become None."""
    try:
        runner = OPTIMIZER_METHODS.get(method)
        if runner is None:
            return None, [f"Unknown optimizer method: {method}"], {}

        w, warn, diag = runner(
            mu=mu,
            cov=cov,
            current_w=current_w,
            tickers=tickers,
            currencies=currencies,
            bounds=bounds,
            sector_groups=sector_groups,
            per_ticker_cost=per_ticker_cost,
            results=results,
            risk_aversion=risk_aversion,
        )

        w = _apply_common_constraints(w, bounds, tickers, results)
        return w, warn, diag
    except Exception as e:
        logger.warning("Optimizer method '%s' failed: %s", method, e)
        return None, [f"{method}: {type(e).__name__}: {e}"], {}


def optimize_portfolio(
    results: list[dict],
    holdings: list[dict],
    risk_data: dict | None = None,
    position_weights: list[dict] | None = None,
    regime: dict | None = None,
) -> PortfolioAllocation:
    """Run full multi-method ensemble portfolio optimisation."""
    tickers = [h["ticker"] for h in holdings]
    names = [h.get("name", h["ticker"]) for h in holdings]
    currencies = [h.get("currency", "GBP") for h in holdings]
    result_map = {r["ticker"]: r for r in results}
    n = len(tickers)

    all_warnings: list[str] = []

    # Risk-free rate
    risk_free = 0.04
    try:
        tnx = yf.download("^TNX", period="5d", progress=False, auto_adjust=True)
        if tnx is not None and not tnx.empty:
            risk_free = float(tnx["Close"].iloc[-1:].values[0]) / 100
    except Exception:
        pass

    # Current weights and portfolio value
    current_w = _current_weights(holdings, results)
    total_value = 0.0
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

    # μ and Σ (v2)
    mu = _estimate_expected_returns(results, tickers)

    # Action-aware SELL/STRONG SELL handling is enforced via _build_bounds
    # (tightens upper weight to MIN_WEIGHT / 2·MIN_WEIGHT) and via Ω
    # inflation in BL. Directly overriding μ stacks two penalties and
    # breaks BL's posterior interpretation — removed (Black & Litterman
    # 1992 require views to be encoded through Q/Ω, not through Π).

    regime_label = (regime or {}).get("regime_label", "NEUTRAL") if regime else "NEUTRAL"
    cov = _estimate_covariance(
        tickers,
        regime_label=regime_label,
        method=OPTIMIZER_COV_METHOD,
        currencies=currencies,
    )

    # Re-apply James-Stein shrinkage now that Σ is available so the target
    # is μ_mvp rather than the arithmetic cross-sectional mean (Jorion 1986).
    try:
        mu = _james_stein_shrink(mu, alpha_max=OPTIMIZER_MU_SHRINKAGE_MAX, cov=cov)
    except Exception as _js_e:
        logger.debug("mu_mvp shrinkage failed (%s) - keeping grand-mean target", _js_e)

    # Bounds / sector groups / per-ticker cost
    bounds = _build_bounds(tickers, results)
    sector_groups = _sector_index(tickers, results)

    # Transaction cost uses the *actual per-name trade value* (|Δw|·NAV) —
    # we don't know Δw yet so estimate via max of current weight vs a
    # first-pass rebalance size (MAX_WEIGHT); Almgren-Chriss is near-
    # monotone in trade value so a slight over-estimate is conservative
    # and self-correcting in the cost-adjusted optimiser.
    per_ticker_cost = np.array([
        estimate_transaction_cost(
            vol_ann=float(np.sqrt(cov[i, i])) if cov[i, i] > 0 else 0.20,
            avg_dollar_volume=(result_map.get(t, {}).get("avg_dollar_volume")
                               or result_map.get(t, {}).get("_avg_dollar_volume")
                               or 1e6),
            trade_value=max(
                total_value * max(MIN_WEIGHT, float(current_w[i])),
                total_value * 0.02,
            ),
            currency=(result_map.get(t, {}).get("currency")
                      or result_map.get(t, {}).get("_currency", "GBP")),
        )
        for i, t in enumerate(tickers)
    ])

    # Regime-adjusted risk aversion
    risk_aversion = RISK_AVERSION
    if regime_label == "BEAR":
        risk_aversion *= 1.5
    elif regime_label == "BULL":
        risk_aversion *= 0.8

    # --- Run each method in the ensemble ---
    requested_methods = list(OPTIMIZER_ENSEMBLE_METHODS)
    per_method_w_arr: dict[str, np.ndarray] = {}
    per_method_diag: dict[str, dict] = {}

    for method in requested_methods:
        w_m, warn_m, diag_m = _run_single_method(
            method,
            mu=mu, cov=cov, current_w=current_w,
            tickers=tickers, currencies=currencies,
            bounds=bounds, sector_groups=sector_groups,
            per_ticker_cost=per_ticker_cost,
            results=results, risk_aversion=risk_aversion,
        )
        all_warnings.extend(warn_m)
        if w_m is not None:
            per_method_w_arr[method] = w_m
            per_method_diag[method] = diag_m

    # --- Combine via history-weighted ensemble ---
    history_run_count = 0
    if per_method_w_arr:
        history = _load_optimizer_history()
        history_run_count = int(len(history.get("runs", [])))
        method_weights = _ensemble_method_weights(history, list(per_method_w_arr.keys()))
        optimal_w = _ensemble_combine(
            per_method_w_arr, method_weights, bounds, tickers, results,
        )
    else:
        all_warnings.append("All optimiser methods failed — equal-weight fallback")
        logger.warning("All optimiser methods failed — equal-weight fallback")
        method_weights = {}
        optimal_w = np.full(n, 1.0 / n)

    if np.allclose(optimal_w, current_w, atol=0.01):
        all_warnings.append("Portfolio is already near-optimal — no significant rebalancing needed")

    # --- Portfolio stats ---
    port_ret = float(mu @ optimal_w)
    port_vol = float(np.sqrt(max(optimal_w @ cov @ optimal_w, 0.0)))
    port_sharpe = (port_ret - risk_free) / port_vol if port_vol > 0 else 0.0
    vols = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    port_var = float(optimal_w @ cov @ optimal_w)
    if port_var > 0:
        mctr = (cov @ optimal_w) / np.sqrt(port_var)
    else:
        mctr = np.zeros(n)

    sharpe_contrib = np.zeros(n)
    for i in range(n):
        if vols[i] > 0:
            holding_sharpe = (mu[i] - risk_free) / vols[i]
            sharpe_contrib[i] = holding_sharpe * optimal_w[i]

    # --- Per-holding allocation records ---
    sectors_map = _get_sectors(results, tickers)
    holding_allocs: list[HoldingAllocation] = []
    for i, ticker in enumerate(tickers):
        r = result_map.get(ticker, {})
        delta = optimal_w[i] - current_w[i]
        fx_cost = abs(delta) * FX_COST_PER_LEG if currencies[i] not in ("GBP", "GBX") else 0.0
        holding_allocs.append(HoldingAllocation(
            ticker=ticker,
            name=names[i],
            current_weight=round(float(current_w[i]), 4),
            optimal_weight=round(float(optimal_w[i]), 4),
            rebalance_delta=round(float(delta), 4),
            expected_return=round(float(mu[i]), 4),
            volatility=round(float(vols[i]), 4),
            sharpe_contribution=round(float(sharpe_contrib[i]), 4),
            sector=sectors_map.get(ticker, "Unknown"),
            currency=currencies[i],
            action=r.get("action", "KEEP"),
            aggregate_score=r.get("aggregate_score", 0),
            fx_cost_if_rebalanced=round(fx_cost, 4),
        ))

    opt_sector_weights: dict[str, float] = {}
    for i, ticker in enumerate(tickers):
        s = sectors_map.get(ticker, "Unknown")
        opt_sector_weights[s] = opt_sector_weights.get(s, 0.0) + float(optimal_w[i])
    opt_sector_weights = {k: round(v, 4) for k, v in opt_sector_weights.items()}

    fx_exposure: dict[str, float] = {}
    for i, c in enumerate(currencies):
        norm_c = "GBP" if c == "GBX" else c
        fx_exposure[norm_c] = fx_exposure.get(norm_c, 0.0) + float(optimal_w[i])
    fx_exposure = {k: round(v, 4) for k, v in fx_exposure.items()}

    turnover = float(np.sum(np.abs(optimal_w - current_w))) / 2
    rebalance_trades = _build_rebalance_trades(
        tickers, names, current_w, optimal_w, total_value,
    )

    # --- Build additive per-method/method-weights views ---
    per_method_weights_view: dict[str, dict[str, float]] = {}
    for i, t in enumerate(tickers):
        per_method_weights_view[t] = {
            m: round(float(vec[i]), 4) for m, vec in per_method_w_arr.items()
        }
    method_weights_rounded = {m: round(float(v), 4) for m, v in method_weights.items()}

    # Summary method string (preserves string consumers)
    if method_weights_rounded:
        summary = "ensemble(" + ",".join(
            f"{_short(m)}={v:.2f}" for m, v in method_weights_rounded.items()
        ) + ")"
    else:
        summary = "equal_weight_fallback"

    # --- Record run to history (non-fatal) ---
    if per_method_w_arr:
        _record_optimizer_run(per_method_w_arr, tickers)

    return PortfolioAllocation(
        holdings=holding_allocs,
        portfolio_expected_return=round(port_ret, 4),
        portfolio_volatility=round(port_vol, 4),
        portfolio_sharpe=round(port_sharpe, 3),
        risk_free_rate=round(risk_free, 4),
        method=summary,
        sector_weights=opt_sector_weights,
        fx_exposure=fx_exposure,
        turnover=round(turnover, 4),
        rebalance_trades=rebalance_trades,
        warnings=all_warnings,
        per_method_weights=per_method_weights_view,
        method_weights=method_weights_rounded,
        mu_version="v2" if OPTIMIZER_MU_V2 else "legacy",
        cov_method=OPTIMIZER_COV_METHOD,
        history_run_count=history_run_count,
    )


def _short(method: str) -> str:
    """Short label for the method string."""
    return {
        "mean_variance": "mv",
        "min_variance": "mvar",
        "risk_parity": "rp",
        "black_litterman": "bl",
        "hrp": "hrp",
    }.get(method, method[:4])


def optimize_with_candidate(
    results: list[dict],
    holdings: list[dict],
    candidate_ticker: str,
    replace_ticker: str,
    risk_data: dict | None = None,
    regime: dict | None = None,
) -> tuple[PortfolioAllocation, PortfolioAllocation]:
    """Compare current portfolio vs portfolio with a swap."""
    current_alloc = optimize_portfolio(results, holdings, risk_data, regime=regime)

    new_holdings = [h for h in holdings if h["ticker"] != replace_ticker]
    replaced = next((h for h in holdings if h["ticker"] == replace_ticker), None)
    if replaced:
        new_holdings.append({
            "ticker": candidate_ticker,
            "name": candidate_ticker,
            "quantity": replaced["quantity"],
            "avg_buy_price": 0,
            "currency": replaced.get("currency", "GBP"),
        })

    new_results = [r for r in results if r["ticker"] != replace_ticker]
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
