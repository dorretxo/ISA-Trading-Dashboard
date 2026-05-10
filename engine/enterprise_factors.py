"""Enterprise-grade factor definitions.

This module centralises the hedge-fund-grade factor math that replaces the
heuristic value lens and oscillator-heavy scoring.  Each function cites the
paper it implements; unit tests live in `tests/test_enterprise_factors.py`.

Factors:
    - EV/EBIT and EV/EBITDA earnings yields (Loughran & Wellman 2011; Greenblatt 2006)
    - Piotroski F-score (Piotroski 2000; Walkshäusl 2020)
    - Gross Profitability / Assets (Novy-Marx 2013)
    - Residual momentum 12-1 with 1-month skip
      (Jegadeesh & Titman 1993; Blitz, Huij & Martens 2011)
    - Barroso-Santa-Clara risk-managed momentum (2015)

Utilities:
    - Sector-neutral cross-sectional z-score with winsorisation and
      James-Stein shrinkage toward the global mean (Stein 1956; Jorion 1986;
      MSCI Enhanced Value methodology)
    - Cross-sectional orthogonalisation (OLS residuals — Fama-MacBeth 1973)
    - Regime-conditioned factor tilts (Cooper, Gutierrez & Hameed 2004;
      Asness, Moskowitz & Pedersen 2013)

All public helpers are defensive: `None` on missing input, never raise.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, Mapping

import numpy as np

from engine.fscore_utils import f_score_min_coverage
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _finite(value) -> float | None:
    """Return float(value) if finite, else None."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


# ---------------------------------------------------------------------------
# #1 EV/EBIT and EV/EBITDA (Loughran & Wellman 2011; Greenblatt 2006)
# ---------------------------------------------------------------------------

def enterprise_value(info: Mapping) -> float | None:
    """Compute Enterprise Value = MarketCap + TotalDebt − Cash.

    Prefers yfinance's pre-computed `enterpriseValue` when present, otherwise
    assembles it from the balance-sheet components in `info`.  Returns None
    if any essential input is missing or the result is non-positive.
    """
    ev = _finite(info.get("enterpriseValue"))
    if ev is not None and ev > 0:
        return ev

    mcap = _finite(info.get("marketCap"))
    debt = _finite(info.get("totalDebt"))
    cash = _finite(info.get("totalCash")) or _finite(
        info.get("cashAndCashEquivalents")
    ) or 0.0
    if mcap is None or mcap <= 0 or debt is None:
        return None
    ev_calc = mcap + debt - cash
    return ev_calc if ev_calc > 0 else None


def compute_ev_ebit(info: Mapping) -> dict[str, float | None]:
    """Return EV/EBIT, EV/EBITDA, and their earnings-yield inversions.

    Earnings yield = EBIT / EV (Greenblatt "Magic Formula"): higher = cheaper.
    Loughran & Wellman (2011) showed EV/EBITDA dominates P/E on 10y Sharpe.

    EV/EBIT is undefined for financials (EBIT isn't meaningful for banks);
    callers should screen financials separately.
    """
    ev = enterprise_value(info)
    ebit = _finite(info.get("ebit")) or _finite(info.get("operatingIncome"))
    ebitda = _finite(info.get("ebitda"))

    out: dict[str, float | None] = {
        "enterprise_value": ev,
        "ev_ebit": None,
        "ev_ebitda": None,
        "ebit_yield": None,    # EBIT / EV — higher = cheaper
        "ebitda_yield": None,  # EBITDA / EV
        "ev_ebit_score": None,
    }
    if ev is None:
        return out

    if ebit is not None and ebit > 0:
        out["ev_ebit"] = ev / ebit
        out["ebit_yield"] = ebit / ev
    if ebitda is not None and ebitda > 0:
        out["ev_ebitda"] = ev / ebitda
        out["ebitda_yield"] = ebitda / ev

    # Heuristic anchor score for ensembles that still expect [-1, 1]:
    # EBIT yield of 10% is roughly median; 20%+ is deep value.
    if out["ebit_yield"] is not None:
        y = out["ebit_yield"]
        out["ev_ebit_score"] = _clip((y - 0.10) / 0.10)
    return out


# ---------------------------------------------------------------------------
# #2 Piotroski F-score (Piotroski 2000; Walkshäusl 2020)
# ---------------------------------------------------------------------------
#
# Nine binary accounting checks, each 1 point if met:
#   Profitability (4):
#     1. ROA > 0
#     2. CFO > 0
#     3. Δ ROA > 0 (YoY)
#     4. CFO > Net Income (accrual quality)
#   Leverage / Liquidity / Source of Funds (3):
#     5. Δ long-term debt / assets < 0
#     6. Δ current ratio > 0
#     7. No new shares issued (Δ shares ≤ 0)
#   Operating Efficiency (2):
#     8. Δ gross margin > 0
#     9. Δ asset turnover > 0
#
# Each `PeriodFinancials` is a dict with keys:
#   net_income, operating_cashflow, total_assets, long_term_debt,
#   current_assets, current_liabilities, shares_outstanding,
#   gross_profit, revenue

def _piotroski_test(
    latest: Mapping,
    prior: Mapping | None,
) -> tuple[int, dict[str, int | None]]:
    """Run the nine F-score tests. Each test is 1 (pass), 0 (fail), or None (missing)."""
    ni = _finite(latest.get("net_income"))
    cfo = _finite(latest.get("operating_cashflow"))
    ta = _finite(latest.get("total_assets"))
    ta_prior = _finite(prior.get("total_assets")) if prior else None

    ni_prior = _finite(prior.get("net_income")) if prior else None
    ltd = _finite(latest.get("long_term_debt"))
    ltd_prior = _finite(prior.get("long_term_debt")) if prior else None
    ca = _finite(latest.get("current_assets"))
    cl = _finite(latest.get("current_liabilities"))
    ca_prior = _finite(prior.get("current_assets")) if prior else None
    cl_prior = _finite(prior.get("current_liabilities")) if prior else None
    shares = _finite(latest.get("shares_outstanding"))
    shares_prior = _finite(prior.get("shares_outstanding")) if prior else None
    gp = _finite(latest.get("gross_profit"))
    gp_prior = _finite(prior.get("gross_profit")) if prior else None
    rev = _finite(latest.get("revenue"))
    rev_prior = _finite(prior.get("revenue")) if prior else None

    checks: dict[str, int | None] = {}

    # 1. ROA > 0
    roa = (ni / ta) if (ni is not None and ta and ta > 0) else None
    checks["roa_positive"] = 1 if (roa is not None and roa > 0) else (0 if roa is not None else None)

    # 2. CFO > 0
    checks["cfo_positive"] = 1 if (cfo is not None and cfo > 0) else (0 if cfo is not None else None)

    # 3. Δ ROA > 0
    roa_prior = (ni_prior / ta_prior) if (ni_prior is not None and ta_prior and ta_prior > 0) else None
    if roa is not None and roa_prior is not None:
        checks["delta_roa_positive"] = 1 if roa > roa_prior else 0
    else:
        checks["delta_roa_positive"] = None

    # 4. CFO > NI  (accrual quality)
    if cfo is not None and ni is not None:
        checks["accrual_quality"] = 1 if cfo > ni else 0
    else:
        checks["accrual_quality"] = None

    # 5. Δ LTD / assets < 0
    lev = (ltd / ta) if (ltd is not None and ta and ta > 0) else None
    lev_prior = (ltd_prior / ta_prior) if (ltd_prior is not None and ta_prior and ta_prior > 0) else None
    if lev is not None and lev_prior is not None:
        checks["delta_leverage_down"] = 1 if lev < lev_prior else 0
    else:
        checks["delta_leverage_down"] = None

    # 6. Δ current ratio > 0
    curr_ratio = (ca / cl) if (ca is not None and cl and cl > 0) else None
    curr_prior = (ca_prior / cl_prior) if (ca_prior is not None and cl_prior and cl_prior > 0) else None
    if curr_ratio is not None and curr_prior is not None:
        checks["delta_current_ratio_up"] = 1 if curr_ratio > curr_prior else 0
    else:
        checks["delta_current_ratio_up"] = None

    # 7. No new shares
    if shares is not None and shares_prior is not None:
        checks["no_new_shares"] = 1 if shares <= shares_prior * 1.01 else 0  # 1% tolerance
    else:
        checks["no_new_shares"] = None

    # 8. Δ gross margin > 0
    gm = (gp / rev) if (gp is not None and rev and rev > 0) else None
    gm_prior = (gp_prior / rev_prior) if (gp_prior is not None and rev_prior and rev_prior > 0) else None
    if gm is not None and gm_prior is not None:
        checks["delta_gross_margin_up"] = 1 if gm > gm_prior else 0
    else:
        checks["delta_gross_margin_up"] = None

    # 9. Δ asset turnover > 0
    at_ratio = (rev / ta) if (rev is not None and ta and ta > 0) else None
    at_prior = (rev_prior / ta_prior) if (rev_prior is not None and ta_prior and ta_prior > 0) else None
    if at_ratio is not None and at_prior is not None:
        checks["delta_asset_turnover_up"] = 1 if at_ratio > at_prior else 0
    else:
        checks["delta_asset_turnover_up"] = None

    passed = sum(1 for v in checks.values() if v == 1)
    return passed, checks


def compute_piotroski_f_score(
    latest: Mapping | None,
    prior: Mapping | None = None,
) -> dict[str, float | int | dict | None]:
    """Compute the F-score and return the checklist.

    Returns dict with keys:
      - `f_score` (0..9 int) or None when latest is missing
      - `f_score_coverage` (0..1 float) — fraction of checks with data
      - `f_score_gate` (bool) — passes the ≥6 gate (Piotroski 2000 original cut)
      - `checks` (dict of 9 named booleans or None)
      - `f_score_score` ([-1,1]) — scaled for ensemble blenders
    """
    if not latest:
        return {
            "f_score": None,
            "f_score_coverage": None,
            "f_score_gate": False,
            "checks": {},
            "f_score_score": None,
        }

    passed, checks = _piotroski_test(latest, prior)
    answered = sum(1 for v in checks.values() if v is not None)
    coverage = answered / 9.0 if answered else 0.0

    # Scale score: 0..9 → −1..+1 centred at 4.5.  Below 4 = bearish, ≥6 = bullish.
    scaled = _clip((passed - 4.5) / 3.0)

    # Gate requires decent coverage (≥6 answered checks) and ≥6 passes
    gate = coverage >= f_score_min_coverage() and passed >= 6

    return {
        "f_score": int(passed),
        "f_score_coverage": float(coverage),
        "f_score_gate": bool(gate),
        "checks": checks,
        "f_score_score": float(scaled),
    }


# ---------------------------------------------------------------------------
# Gross Profitability / Assets (Novy-Marx 2013) — first-class version
# ---------------------------------------------------------------------------

def compute_gpa(info: Mapping) -> dict[str, float | None]:
    """Gross profits / total assets — Novy-Marx (2013).

    This is the same number used (at 35% weight) inside the existing quality
    composite, exposed here as a first-class field with its own anchor
    score.  Median GPA across US equities is ~0.30; >0.40 is strong quality.
    """
    gp = _finite(info.get("grossProfits")) or _finite(info.get("grossProfit"))
    ta = _finite(info.get("totalAssets"))
    if gp is None or ta is None or ta <= 0:
        return {"gpa": None, "gpa_score": None}
    gpa = gp / ta
    return {"gpa": gpa, "gpa_score": _clip((gpa - 0.30) / 0.20)}


# ---------------------------------------------------------------------------
# #4 Residual momentum 12-1 + Barroso-Santa-Clara vol scaling
# ---------------------------------------------------------------------------

def residual_momentum_12_1(
    close: pd.Series,
    market_close: pd.Series,
    trading_days_per_month: int = 21,
    skip_months: int = 1,
    lookback_months: int = 12,
) -> float | None:
    """Beta-adjusted residual momentum over a 12-1 window.

    Jegadeesh & Titman (1993): the standard momentum anomaly is the
    cumulative return from t-12 to t-2 months (skipping the most recent
    month to avoid short-term reversal).

    Blitz, Huij & Martens (2011): residualise the returns against the
    market via time-series OLS regression.  The residuals — stripped of
    CAPM β exposure — have ~30-40% higher information ratio than raw
    momentum and much lower crash risk.

    Implementation:
        1. Compute daily returns for stock and market, aligned on index.
        2. Run rolling 12-month (t-12m → t-1m) OLS: r_i = α + β · r_m + ε.
        3. Return the sum of ε over the window (the "residual return").
    """
    if close is None or market_close is None:
        return None
    if len(close) < (lookback_months + skip_months) * trading_days_per_month + 10:
        return None
    if len(market_close) < (lookback_months + skip_months) * trading_days_per_month + 10:
        return None

    ret_i = close.pct_change(fill_method=None).dropna()
    ret_m = market_close.pct_change(fill_method=None).dropna()
    joined = pd.concat([ret_i, ret_m], axis=1, join="inner").dropna()
    if joined.shape[1] != 2 or len(joined) < lookback_months * trading_days_per_month:
        return None

    total_days = lookback_months * trading_days_per_month
    skip_days = skip_months * trading_days_per_month
    window = joined.iloc[-(total_days + skip_days): -skip_days] if skip_days > 0 else joined.iloc[-total_days:]
    if len(window) < total_days // 2:
        return None

    y = window.iloc[:, 0].values.astype(float)
    x = window.iloc[:, 1].values.astype(float)
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = float((x_c * x_c).sum())
    if denom <= 1e-12:
        beta = 0.0
    else:
        beta = float((x_c * y_c).sum() / denom)
    alpha = float(y.mean() - beta * x.mean())
    resid = y - (alpha + beta * x)
    residual_return = float(np.sum(resid))
    return residual_return


def barroso_santa_clara_scale(
    signal: float,
    realised_vol: float | None,
    target_vol: float = 0.12,
    cap: float = 3.0,
) -> float:
    """Risk-managed momentum scaling (Barroso & Santa-Clara 2015).

    Scales a momentum signal by (target_vol / realised_vol) so that the
    position carries a constant ex-ante volatility budget.  When realised
    vol spikes (e.g. March 2020, October 2008) the signal is de-weighted,
    avoiding the well-known momentum crashes that follow volatility shocks.

    target_vol defaults to 12% annualised — a common choice in the literature.
    """
    if not math.isfinite(signal):
        return 0.0
    if realised_vol is None or not math.isfinite(realised_vol) or realised_vol <= 0:
        return float(signal)
    scale = target_vol / realised_vol
    scale = max(1.0 / cap, min(cap, scale))
    return float(signal) * scale


# ---------------------------------------------------------------------------
# #6 Sector-neutral cross-sectional z-score
# ---------------------------------------------------------------------------

def sector_neutral_zscore(
    values: Iterable[float | None],
    sectors: Iterable[str | None],
    *,
    winsor: float = 3.0,
    min_peers: int = 5,
    shrinkage_max: float = 0.40,
) -> np.ndarray:
    """Sector-demeaned cross-sectional z-score with winsorisation.

    Steps:
      1. Group values by sector.
      2. Within each group with >= `min_peers` observations, demean and
         scale by sector std, winsorise at ±winsor σ.
      3. For under-populated sectors (< min_peers), shrink toward the
         global mean via James-Stein α (Stein 1956; Jorion 1986).  α is
         higher when the sector cross-section is narrow.
      4. Re-z-score globally after demeaning so the output has zero mean
         and unit variance, and clip to ±winsor.

    This is the construction used by MSCI Enhanced Value / Quality indices
    and the AQR QMJ factor.

    Returns an ndarray aligned to the input order.  Missing values
    (None/NaN) come back as 0.0 (neutral).
    """
    vals = np.array(
        [float(v) if (v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))) else np.nan
         for v in values], dtype=float
    )
    secs = [str(s) if s else "Unknown" for s in sectors]
    n = len(vals)
    if n == 0:
        return vals

    # Global mean/std (robust to NaN)
    if np.all(np.isnan(vals)):
        return np.zeros(n)
    global_mu = float(np.nanmean(vals))
    global_sd = float(np.nanstd(vals))
    if global_sd < 1e-12:
        return np.zeros(n)

    # Sector-demeaned
    demeaned = vals.copy()
    for sector in set(secs):
        idx = [i for i, s in enumerate(secs) if s == sector]
        if len(idx) < min_peers:
            # shrink toward global mean
            alpha = min(shrinkage_max, 1.0 / (1 + len(idx)))
            for i in idx:
                if not np.isnan(vals[i]):
                    sector_mu = float(np.nanmean([vals[j] for j in idx]))
                    shrunk = (1 - alpha) * sector_mu + alpha * global_mu
                    demeaned[i] = vals[i] - shrunk
        else:
            sector_vals = np.array([vals[i] for i in idx], dtype=float)
            mu = float(np.nanmean(sector_vals))
            for i in idx:
                if not np.isnan(vals[i]):
                    demeaned[i] = vals[i] - mu

    # Rescale by global std of the demeaned values
    scale = float(np.nanstd(demeaned))
    if scale < 1e-12:
        scale = global_sd
    z = np.where(np.isnan(demeaned), 0.0, demeaned / scale)
    return np.clip(z, -winsor, winsor)


# ---------------------------------------------------------------------------
# #7 Cross-sectional orthogonalisation (OLS residual)
# ---------------------------------------------------------------------------

def orthogonalise(
    target: Iterable[float],
    *regressors: Iterable[float],
    add_intercept: bool = True,
) -> np.ndarray:
    """Return the target stripped of linear dependence on the regressors.

    Equivalent to a one-shot Fama-MacBeth (1973) cross-sectional regression:
    y = α + β₁·X₁ + β₂·X₂ + … + ε; return ε.  Used to "residualise" the
    sentiment and forecast signals against the factor core so they add
    incremental IC without dominating (Tetlock 2007; Asness et al. 2019).

    Rows with missing values are dropped pairwise via pandas.  The returned
    array is aligned to the input length, with 0.0 for rows that had to be
    dropped.
    """
    y = np.array([float(v) if (v is not None and math.isfinite(float(v))) else np.nan for v in target], dtype=float)
    Xs = [np.array([float(v) if (v is not None and math.isfinite(float(v))) else np.nan for v in r], dtype=float)
          for r in regressors]
    n = len(y)
    if not Xs or n == 0:
        return y

    X = np.column_stack(Xs)
    if add_intercept:
        X = np.column_stack([np.ones(n), X])

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if mask.sum() < X.shape[1] + 2:
        # Not enough data — return 0.0 (neutral)
        out = np.zeros(n)
        return out

    y_fit = y[mask]
    X_fit = X[mask]
    try:
        beta, *_ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
    except Exception as e:
        logger.warning("Orthogonalisation OLS failed: %s", e)
        return np.where(np.isfinite(y), y, 0.0)

    pred = X @ beta
    resid = y - pred
    return np.where(np.isfinite(resid), resid, 0.0)


# ---------------------------------------------------------------------------
# #8 Regime-conditioned factor tilts
# ---------------------------------------------------------------------------

# Evidence base:
#   Cooper, Gutierrez & Hameed (2004): momentum concentrates in UP markets.
#   Asness, Moskowitz & Pedersen (2013): value and momentum work globally,
#     with correlation that flips in crises — diversification across the pair
#     is itself regime-dependent.
#   Barroso & Santa-Clara (2015): momentum crashes follow volatility shocks.
#   MSCI / S&P Dynamic Factor methodologies: factor weight schedules by regime.

REGIME_FACTOR_TILTS: dict[str, dict[str, float]] = {
    "BULL":    {"value": 0.85, "quality": 0.95, "momentum": 1.25, "low_vol": 0.70, "f_score": 0.90},
    "NEUTRAL": {"value": 1.00, "quality": 1.00, "momentum": 1.00, "low_vol": 1.00, "f_score": 1.00},
    "BEAR":    {"value": 1.15, "quality": 1.30, "momentum": 0.60, "low_vol": 1.40, "f_score": 1.20},
}


def regime_factor_tilt(
    base_weights: Mapping[str, float],
    regime_label: str | None,
) -> dict[str, float]:
    """Multiply `base_weights` by the regime-specific tilt and re-normalise."""
    reg = (regime_label or "NEUTRAL").upper()
    tilts = REGIME_FACTOR_TILTS.get(reg, REGIME_FACTOR_TILTS["NEUTRAL"])
    tilted: dict[str, float] = {}
    for k, w in base_weights.items():
        t = tilts.get(k, 1.0)
        tilted[k] = float(w) * float(t)
    total = sum(tilted.values())
    if total <= 0:
        return dict(base_weights)
    return {k: v / total for k, v in tilted.items()}


# ---------------------------------------------------------------------------
# Piotroski data extractor (wraps yfinance objects for convenience)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TTM-quarterly Piotroski and GPA (fixes annual-staleness for cyclicals)
# ---------------------------------------------------------------------------
#
# The annual-frequency Piotroski score goes stale mid-cycle: a commodity
# producer at trough prints a broken F-score for 9-12 months after
# fundamentals have already inflected.  Rolling four quarters of flows
# against the matched B/S snapshots cuts that lag to ~90 days and lets the
# score respond to inflection points (Beneish 1997; Piotroski & So 2012).
#
# Quarterly limitations:
#   - Operating cash flow is not in FMP income statements; we still use
#     yfinance `operatingCashflow` for the latest TTM CFO check when
#     available.  Prior-period CFO is unavailable → check #3 (ΔCFO) remains
#     unanswered, not failed.  Coverage handles this.

def _fmp_row_get(row: Mapping, *keys) -> float | None:
    """Pull the first finite numeric field among `keys` from an FMP row."""
    for k in keys:
        v = row.get(k)
        f = _finite(v)
        if f is not None:
            return f
    return None


def _sum_quarterly(rows: list[Mapping], key_aliases: tuple[str, ...]) -> float | None:
    """Sum a flow-type field across rows. Returns None if any row is missing it."""
    total = 0.0
    any_found = False
    for r in rows:
        v = _fmp_row_get(r, *key_aliases)
        if v is None:
            return None
        total += v
        any_found = True
    return total if any_found else None


def _ttm_period_financials(
    income_window: list[Mapping],
    balance_snapshot: Mapping,
    *,
    include_cfo: float | None = None,
    include_shares: bool = True,
) -> dict | None:
    """Build a PeriodFinancials dict from 4 quarterly income statements + one B/S.

    `include_cfo` is supplied externally for the latest TTM only (yfinance
    provides trailing CFO; we don't have a quarterly cashflow FMP call).
    """
    if not income_window or len(income_window) < 4:
        return None

    ni = _sum_quarterly(income_window, ("netIncome", "bottomLineNetIncome"))
    gp = _sum_quarterly(income_window, ("grossProfit",))
    rev = _sum_quarterly(income_window, ("revenue",))
    if ni is None and gp is None and rev is None:
        return None

    ta = _fmp_row_get(balance_snapshot, "totalAssets")
    ltd = _fmp_row_get(balance_snapshot, "longTermDebt")
    ca = _fmp_row_get(balance_snapshot, "totalCurrentAssets")
    cl = _fmp_row_get(balance_snapshot, "totalCurrentLiabilities")

    shares = None
    if include_shares:
        # Use the *latest* quarter's share count (not an average)
        shares = _fmp_row_get(income_window[0], "weightedAverageShsOut")

    return {
        "net_income": ni,
        "operating_cashflow": _finite(include_cfo) if include_cfo is not None else None,
        "total_assets": ta,
        "long_term_debt": ltd,
        "current_assets": ca,
        "current_liabilities": cl,
        "shares_outstanding": shares,
        "gross_profit": gp,
        "revenue": rev,
    }


def compute_ttm_piotroski_f_score(
    quarterly_incomes: list[Mapping] | None,
    quarterly_balances: list[Mapping] | None,
    *,
    info: Mapping | None = None,
) -> dict:
    """TTM-rolling Piotroski F-score using 8 quarters of statements.

    quarterly_incomes: list of FMP quarterly income rows, newest first.
    quarterly_balances: list of FMP quarterly balance rows, newest first.

    Needs at least 8 quarters of income and 5 quarters of balance sheet
    (latest + the Q-4 snapshot).  Returns the same shape as
    `compute_piotroski_f_score`.
    """
    if (
        not quarterly_incomes
        or not quarterly_balances
        or len(quarterly_incomes) < 8
        or len(quarterly_balances) < 5
    ):
        return {
            "f_score": None,
            "f_score_coverage": None,
            "f_score_gate": False,
            "checks": {},
            "f_score_score": None,
            "f_score_source": "ttm_unavailable",
        }

    latest_window = quarterly_incomes[0:4]   # most recent TTM
    prior_window = quarterly_incomes[4:8]    # prior TTM (for Δ tests)
    latest_bs = quarterly_balances[0]
    prior_bs = quarterly_balances[4] if len(quarterly_balances) >= 5 else None

    cfo_ttm = None
    if info:
        # yfinance `operatingCashflow` is TTM trailing — apply to latest only
        cfo_ttm = _finite(info.get("operatingCashflow"))

    latest = _ttm_period_financials(latest_window, latest_bs, include_cfo=cfo_ttm)
    prior = _ttm_period_financials(prior_window, prior_bs) if prior_bs is not None else None
    if latest is None:
        return {
            "f_score": None,
            "f_score_coverage": None,
            "f_score_gate": False,
            "checks": {},
            "f_score_score": None,
            "f_score_source": "ttm_extraction_failed",
        }

    result = compute_piotroski_f_score(latest, prior)
    result["f_score_source"] = "ttm_quarterly"
    return result


def compute_ttm_gpa(
    quarterly_incomes: list[Mapping] | None,
    quarterly_balances: list[Mapping] | None,
) -> dict[str, float | None]:
    """GPA using trailing 4Q gross profit / latest quarter total assets.

    Same construction as annual GPA (Novy-Marx 2013), but updated quarterly
    so cyclical recoveries / declines show up ~9 months sooner.
    """
    if not quarterly_incomes or not quarterly_balances or len(quarterly_incomes) < 4:
        return {"gpa": None, "gpa_score": None, "gpa_source": "ttm_unavailable"}

    gp_ttm = _sum_quarterly(quarterly_incomes[0:4], ("grossProfit",))
    ta = _fmp_row_get(quarterly_balances[0], "totalAssets")
    if gp_ttm is None or ta is None or ta <= 0:
        return {"gpa": None, "gpa_score": None, "gpa_source": "ttm_extraction_failed"}
    gpa = gp_ttm / ta
    return {"gpa": gpa, "gpa_score": _clip((gpa - 0.30) / 0.20), "gpa_source": "ttm_quarterly"}


# ---------------------------------------------------------------------------
# Annual extractor (kept for fallback path when quarterly data unavailable)
# ---------------------------------------------------------------------------

def extract_piotroski_period(
    *,
    income_row: Mapping | None,
    balance_row: Mapping | None,
    cashflow_row: Mapping | None,
    info: Mapping | None = None,
) -> dict:
    """Assemble a `PeriodFinancials` record from yfinance row dicts.

    yfinance returns `Ticker.financials`, `.balance_sheet`, `.cashflow` as
    pandas DataFrames with line items as rows and report dates as columns.
    The caller should pick the column (report period) and pass each line's
    dict-like slice here.  Missing rows → None.
    """
    inc = income_row or {}
    bal = balance_row or {}
    cf = cashflow_row or {}
    inf = info or {}

    def g(d: Mapping, *keys):
        for k in keys:
            v = d.get(k)
            f = _finite(v)
            if f is not None:
                return f
        return None

    return {
        "net_income": g(inc, "Net Income", "Net Income Common Stockholders", "NetIncome")
                      or g(inf, "netIncomeToCommon"),
        "operating_cashflow": g(cf, "Operating Cash Flow", "Total Cash From Operating Activities",
                                "Cash Flow From Continuing Operating Activities")
                              or g(inf, "operatingCashflow"),
        "total_assets": g(bal, "Total Assets", "TotalAssets") or g(inf, "totalAssets"),
        "long_term_debt": g(bal, "Long Term Debt", "LongTermDebt"),
        "current_assets": g(bal, "Current Assets", "Total Current Assets"),
        "current_liabilities": g(bal, "Current Liabilities", "Total Current Liabilities"),
        "shares_outstanding": g(inc, "Diluted Average Shares", "Basic Average Shares")
                              or g(inf, "sharesOutstanding"),
        "gross_profit": g(inc, "Gross Profit", "GrossProfit") or g(inf, "grossProfits"),
        "revenue": g(inc, "Total Revenue", "Revenue") or g(inf, "totalRevenue"),
    }
