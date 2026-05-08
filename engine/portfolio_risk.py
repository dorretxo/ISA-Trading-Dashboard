"""Portfolio-level risk analysis: correlations, sector concentration, risk warnings.

Adds institutional-grade risk metrics (2026-04 upgrade):

* Historical Value-at-Risk and Expected Shortfall (Artzner et al. 1999 —
  coherent risk measures); ES is preferred because VaR is not subadditive.
* Cornish-Fisher VaR as a distribution-aware adjustment for skew/kurtosis
  (Cornish & Fisher 1937; Favre & Galeano 2002).
* Portfolio β vs. SPY from simple OLS on daily returns.
* Historical scenario replay — applies actual realised daily returns from
  documented crisis windows (GFC 2008-10-06, COVID 2020-03-12, 2022 selloff,
  August 2015 flash-crash).  Richer than parametric shocks because it
  preserves the observed cross-asset correlation in each stress episode.
"""

import logging
from contextlib import contextmanager
from datetime import date

import numpy as np
import pandas as pd

import config
from utils.data_fetch import get_price_history, get_ticker_info

logger = logging.getLogger(__name__)


@contextmanager
def _quiet_yfinance_errors():
    """Suppress expected missing-history noise during historical stress replay."""
    yf_logger = logging.getLogger("yfinance")
    previous_level = yf_logger.level
    yf_logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        yf_logger.setLevel(previous_level)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
CORRELATION_HIGH = 0.70       # Flag pairs with |corr| above this
SECTOR_CONCENTRATION_PCT = getattr(config, "DISCOVERY_SECTOR_PCT_CAP", 0.30)
VAR_ALPHA = 0.05              # 95% VaR / ES by default
BENCHMARK_TICKER = getattr(config, "PORTFOLIO_BENCHMARK", "SPY")

# Canonical historical stress windows (trade dates inclusive).  Chosen to
# cover distinct regimes: GFC banking crisis, COVID liquidity shock, 2022
# rates-driven tech selloff, August 2015 RMB devaluation flash-crash.
STRESS_SCENARIOS: list[dict] = [
    {"name": "GFC 2008",        "start": "2008-09-15", "end": "2008-10-10"},
    {"name": "COVID 2020",      "start": "2020-02-20", "end": "2020-03-23"},
    {"name": "2022 Rates",      "start": "2022-01-03", "end": "2022-06-16"},
    {"name": "Aug 2015 FX",     "start": "2015-08-17", "end": "2015-08-25"},
]


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def calculate_correlation_matrix(tickers: list[str]) -> pd.DataFrame:
    """Compute pairwise return correlation matrix using 90-day daily returns.

    Returns a DataFrame of shape (n_tickers, n_tickers) with Pearson correlations.
    """
    returns_dict = {}
    for ticker in tickers:
        df = get_price_history(ticker)
        if df is not None and not df.empty and len(df) > 20:
            closes = df["Close"].tail(90)
            daily_returns = closes.pct_change().dropna()
            if len(daily_returns) >= 15:
                returns_dict[ticker] = daily_returns.values[-min(len(daily_returns), 60):]

    if len(returns_dict) < 2:
        return pd.DataFrame()

    # Align lengths — use shortest common length
    min_len = min(len(v) for v in returns_dict.values())
    aligned = {t: r[-min_len:] for t, r in returns_dict.items()}

    returns_df = pd.DataFrame(aligned)
    corr_matrix = returns_df.corr(method="pearson")
    return corr_matrix


def find_high_correlations(
    corr_matrix: pd.DataFrame,
    threshold: float = CORRELATION_HIGH,
) -> list[tuple[str, str, float]]:
    """Find ticker pairs with |correlation| above threshold."""
    if corr_matrix.empty:
        return []

    pairs = []
    tickers = corr_matrix.columns.tolist()
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            corr = corr_matrix.iloc[i, j]
            if not np.isnan(corr) and abs(corr) >= threshold:
                pairs.append((tickers[i], tickers[j], round(corr, 3)))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# Sector concentration
# ---------------------------------------------------------------------------

def detect_sector_concentration(
    results: list[dict],
    holdings: list[dict],
) -> tuple[dict[str, float], list[str]]:
    """Analyse sector weights and flag concentration risks.

    Returns (sector_weights, warnings) where sector_weights maps
    sector name → fraction of total portfolio value, and warnings
    is a list of human-readable risk messages.
    """
    sector_values = {}
    total_value = 0.0

    for result, holding in zip(results, holdings):
        ticker = result["ticker"]
        price = result.get("current_price") or 0
        qty = holding.get("quantity", 0)
        value = price * qty

        # Apply GBX → GBP conversion for display consistency
        currency = holding.get("currency", "GBP")
        if currency == "GBX":
            value *= 0.01

        # Determine sector from yfinance info
        info = get_ticker_info(ticker)
        sector = info.get("sector", "Unknown")
        if not sector:
            sector = "Unknown"

        sector_values[sector] = sector_values.get(sector, 0) + value
        total_value += value

    if total_value <= 0:
        return {}, []

    sector_weights = {s: round(v / total_value, 4) for s, v in sector_values.items()}

    warnings = []
    for sector, weight in sorted(sector_weights.items(), key=lambda x: x[1], reverse=True):
        if weight >= SECTOR_CONCENTRATION_PCT:
            warnings.append(
                f"{sector} sector is {weight:.0%} of portfolio "
                f"(>{SECTOR_CONCENTRATION_PCT:.0%} threshold)"
            )

    return sector_weights, warnings


# ---------------------------------------------------------------------------
# Holdings → aligned daily return matrix
# ---------------------------------------------------------------------------

def _portfolio_weights(results: list[dict], holdings: list[dict]) -> dict[str, float]:
    """Current GBP-normalised weights for the live portfolio."""
    weights: dict[str, float] = {}
    total = 0.0
    for r, h in zip(results, holdings):
        t = r.get("ticker")
        if not t:
            continue
        price = float(r.get("current_price") or 0)
        qty = float(h.get("quantity") or 0)
        factor = 0.01 if (h.get("currency") == "GBX") else 1.0
        val = price * qty * factor
        weights[t] = weights.get(t, 0.0) + val
        total += val
    if total <= 0:
        return {}
    return {t: v / total for t, v in weights.items()}


def _aligned_returns(tickers: list[str], lookback: int = 250) -> pd.DataFrame:
    """Fetch daily returns for `tickers`, aligned on a common index."""
    frames: dict[str, pd.Series] = {}
    for t in tickers:
        df = get_price_history(t)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        closes = df["Close"].tail(lookback + 5).dropna()
        if len(closes) < 30:
            continue
        frames[t] = closes.pct_change().dropna()
    if not frames:
        return pd.DataFrame()
    rets = pd.DataFrame(frames).dropna(how="all").fillna(0.0)
    return rets.tail(lookback)


# ---------------------------------------------------------------------------
# Value-at-Risk / Expected Shortfall / β / stress
# ---------------------------------------------------------------------------

def compute_portfolio_var_es(
    returns: pd.DataFrame,
    weights: dict[str, float],
    alpha: float = VAR_ALPHA,
) -> dict:
    """Historical VaR / ES at confidence ``1 - alpha`` on daily return series.

    Returns VaR / ES both as fraction of portfolio and 1-day Cornish-Fisher
    VaR adjusted for sample skew/kurtosis (useful when the empirical left
    tail is thin).  Caller scales to NAV externally.
    """
    empty = {
        "var_1d": None, "es_1d": None, "var_cf_1d": None,
        "vol_annual": None, "n_obs": 0,
    }
    if returns is None or returns.empty or not weights:
        return empty
    cols = [t for t in returns.columns if t in weights]
    if not cols:
        return empty
    w_vec = np.array([weights[t] for t in cols], dtype=float)
    w_sum = float(w_vec.sum())
    if w_sum <= 0:
        return empty
    w_vec = w_vec / w_sum
    port_rets = returns[cols].values @ w_vec
    port_rets = port_rets[np.isfinite(port_rets)]
    if port_rets.size < 30:
        return empty

    q = float(np.quantile(port_rets, alpha))
    tail = port_rets[port_rets <= q]
    es = float(tail.mean()) if tail.size else q
    vol_annual = float(np.std(port_rets)) * np.sqrt(252.0)

    # Cornish-Fisher adjustment to the z-quantile
    from scipy.stats import norm, skew, kurtosis
    z = float(norm.ppf(alpha))
    s = float(skew(port_rets, bias=False))
    k = float(kurtosis(port_rets, bias=False))  # excess kurtosis
    z_cf = (
        z
        + (z**2 - 1) * s / 6.0
        + (z**3 - 3 * z) * k / 24.0
        - (2 * z**3 - 5 * z) * (s**2) / 36.0
    )
    mu = float(np.mean(port_rets)); sd = float(np.std(port_rets))
    var_cf = mu + sd * z_cf

    return {
        "var_1d": round(q, 5),             # e.g. -0.018 = lose 1.8% or worse on 5% of days
        "es_1d": round(es, 5),             # conditional loss when VaR is breached
        "var_cf_1d": round(float(var_cf), 5),
        "vol_annual": round(vol_annual, 4),
        "n_obs": int(port_rets.size),
    }


def compute_portfolio_beta(
    returns: pd.DataFrame,
    weights: dict[str, float],
    benchmark: str = BENCHMARK_TICKER,
) -> dict:
    """OLS β of portfolio daily returns against the benchmark."""
    empty = {"beta": None, "alpha_daily": None, "r_squared": None,
             "benchmark": benchmark}
    if returns is None or returns.empty or not weights:
        return empty
    bench_df = get_price_history(benchmark)
    if bench_df is None or bench_df.empty or "Close" not in bench_df.columns:
        return empty
    bench_rets = bench_df["Close"].tail(len(returns) + 10).pct_change().dropna()
    if bench_rets.empty:
        return empty

    cols = [t for t in returns.columns if t in weights]
    w = np.array([weights[t] for t in cols], dtype=float)
    if w.sum() <= 0:
        return empty
    w = w / w.sum()
    port = pd.Series(returns[cols].values @ w, index=returns.index)

    aligned = pd.concat([port, bench_rets], axis=1, join="inner").dropna()
    aligned.columns = ["port", "bench"]
    if len(aligned) < 30:
        return empty
    x = aligned["bench"].values
    y = aligned["port"].values
    vx = float(np.var(x))
    if vx < 1e-12:
        return empty
    beta = float(np.cov(y, x, ddof=1)[0, 1] / vx)
    alpha_d = float(np.mean(y) - beta * np.mean(x))
    ss_res = float(np.sum((y - (alpha_d + beta * x)) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else None
    return {
        "beta": round(beta, 3),
        "alpha_daily": round(alpha_d, 6),
        "r_squared": round(r2, 3) if r2 is not None else None,
        "benchmark": benchmark,
    }


def run_stress_scenarios(
    tickers: list[str],
    weights: dict[str, float],
) -> list[dict]:
    """Replay each :data:`STRESS_SCENARIOS` window on the current weights.

    For each scenario we fetch per-ticker daily returns over the window,
    compound them (∏(1+r) − 1), and weight by current portfolio weights to
    get a scenario loss.  Missing tickers are silently dropped (their weight
    is excluded from the event; loss scales only with covered names).
    """
    out: list[dict] = []
    if not tickers or not weights:
        return out

    try:
        import yfinance as _yf
    except Exception:
        return out

    for scen in STRESS_SCENARIOS:
        try:
            start = scen["start"]; end = scen["end"]
            with _quiet_yfinance_errors():
                px = _yf.download(
                    tickers, start=start, end=end,
                    progress=False, auto_adjust=True, group_by="ticker",
                )
        except Exception as e:
            logger.debug("stress scenario fetch failed for %s: %s", scen["name"], e)
            continue
        if px is None or len(px) < 2:
            continue

        scenario_ret = 0.0
        covered_weight = 0.0
        worst_name: tuple[str, float] | None = None
        per_ticker = {}
        for t in tickers:
            try:
                s = px[t]["Close"] if (t in px.columns.get_level_values(0)) else None
            except Exception:
                s = None
            if s is None:
                continue
            s = s.dropna()
            if len(s) < 2:
                continue
            r = float(s.iloc[-1] / s.iloc[0] - 1.0)
            w = float(weights.get(t, 0.0))
            if w <= 0:
                continue
            scenario_ret += w * r
            covered_weight += w
            per_ticker[t] = round(r, 4)
            if worst_name is None or r < worst_name[1]:
                worst_name = (t, r)
        if covered_weight <= 0:
            continue
        # Rescale to full portfolio assuming uncovered names match covered mean.
        rescaled = scenario_ret / covered_weight
        out.append({
            "name": scen["name"],
            "window": f"{scen['start']} → {scen['end']}",
            "portfolio_return": round(rescaled, 4),
            "coverage": round(covered_weight, 3),
            "worst_name": worst_name[0] if worst_name else None,
            "worst_return": round(worst_name[1], 4) if worst_name else None,
            "per_ticker": per_ticker,
        })
    return out


# ---------------------------------------------------------------------------
# Composite risk assessment
# ---------------------------------------------------------------------------

def assess_portfolio_risk(
    results: list[dict],
    holdings: list[dict],
) -> dict:
    """Master portfolio risk function.

    Returns dict with:
        correlation_matrix: pd.DataFrame
        high_correlations: list of (ticker1, ticker2, corr) tuples
        sector_weights: dict of sector → fraction
        concentration_warnings: list of warning strings
        risk_score: float 0-1 (higher = more risky)
    """
    tickers = [r["ticker"] for r in results]

    # Correlation analysis
    corr_matrix = calculate_correlation_matrix(tickers)
    high_corrs = find_high_correlations(corr_matrix)

    # Sector analysis
    sector_weights, concentration_warnings = detect_sector_concentration(results, holdings)

    # Composite risk score (0 = low risk, 1 = high risk)
    risk_components = []

    # Correlation risk: fraction of pairs that are highly correlated
    n_pairs = len(tickers) * (len(tickers) - 1) / 2 if len(tickers) > 1 else 1
    corr_risk = min(1.0, len(high_corrs) / max(n_pairs * 0.3, 1))
    risk_components.append(corr_risk * 0.4)

    # Concentration risk: max sector weight above threshold
    if sector_weights:
        max_sector = max(sector_weights.values())
        conc_risk = max(
            0.0,
            (max_sector - SECTOR_CONCENTRATION_PCT) / max(1.0 - SECTOR_CONCENTRATION_PCT, 1e-6),
        )
        risk_components.append(conc_risk * 0.4)
    else:
        risk_components.append(0.0)

    # Diversification penalty: too few holdings
    n_holdings = len(results)
    div_risk = max(0.0, 1.0 - n_holdings / 15)  # 15+ holdings = no penalty
    risk_components.append(div_risk * 0.2)

    risk_score = min(1.0, sum(risk_components))

    # Build additional warnings
    all_warnings = list(concentration_warnings)
    if len(high_corrs) >= 3:
        all_warnings.append(
            f"{len(high_corrs)} highly correlated pairs detected (ρ > {CORRELATION_HIGH})"
        )
    if n_holdings < 8:
        all_warnings.append(
            f"Low diversification: only {n_holdings} holdings"
        )

    # --- Institutional-grade risk metrics (VaR/ES/β/stress) ---
    var_es: dict = {}
    beta_info: dict = {}
    scenarios: list[dict] = []
    try:
        weights_map = _portfolio_weights(results, holdings)
        if weights_map:
            ret_df = _aligned_returns(list(weights_map.keys()), lookback=250)
            if not ret_df.empty:
                var_es = compute_portfolio_var_es(ret_df, weights_map)
                beta_info = compute_portfolio_beta(ret_df, weights_map)
            scenarios = run_stress_scenarios(list(weights_map.keys()), weights_map)
    except Exception as e:
        logger.warning("portfolio risk metrics (VaR/β/stress) failed: %s", e)

    # Promote severe tail to the warning list (1-day 95% ES worse than -4%)
    es_1d = var_es.get("es_1d") if var_es else None
    if es_1d is not None and es_1d < -0.04:
        all_warnings.append(
            f"95% 1-day Expected Shortfall {es_1d*100:+.1f}% — heavy-tailed portfolio"
        )
    worst_scen = None
    if scenarios:
        worst_scen = min(scenarios, key=lambda s: s.get("portfolio_return", 0.0))
        if worst_scen and worst_scen.get("portfolio_return", 0.0) < -0.20:
            all_warnings.append(
                f"Stress replay '{worst_scen['name']}': "
                f"{worst_scen['portfolio_return']*100:+.1f}%"
            )

    return {
        "correlation_matrix": corr_matrix,
        "high_correlations": high_corrs,
        "sector_weights": sector_weights,
        "concentration_warnings": all_warnings,
        "risk_score": round(risk_score, 3),
        "var_es": var_es,
        "beta": beta_info,
        "stress_scenarios": scenarios,
        "worst_stress": worst_scen,
    }
