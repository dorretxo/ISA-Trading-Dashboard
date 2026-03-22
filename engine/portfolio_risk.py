"""Portfolio-level risk analysis: correlations, sector concentration, risk warnings."""

import logging

import numpy as np
import pandas as pd

import config
from utils.data_fetch import get_price_history, get_ticker_info

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
CORRELATION_HIGH = 0.70       # Flag pairs with |corr| above this
SECTOR_CONCENTRATION_PCT = 0.40  # Warn if any sector > 40% of portfolio value


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
        conc_risk = max(0.0, (max_sector - 0.25) / 0.75)  # 25% = no risk, 100% = max risk
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

    return {
        "correlation_matrix": corr_matrix,
        "high_correlations": high_corrs,
        "sector_weights": sector_weights,
        "concentration_warnings": all_warnings,
        "risk_score": round(risk_score, 3),
    }
