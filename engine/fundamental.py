"""Fundamental analysis: P/E, EPS, D/E, Short Interest, Inst. Ownership,
Insider Transactions, Analyst Targets, Revenue Growth, Margins, ROE, FCF.

FMP is the PRIMARY data source (Starter plan: 300 calls/min). Provides
earnings surprises, quarterly trends, analyst revisions, PEG ratio,
sector-relative P/E. yfinance is the FALLBACK for core metrics.

Sector-relative P/E now works for ALL tickers (not just FMP-covered US stocks).
A local sector P/E cache accumulates P/E ratios by sector across analysis runs,
providing median sector P/E as a fallback when FMP sector_pe is unavailable.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import config
from utils.data_fetch import get_insider_transactions, get_ticker_info

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sector P/E cache — accumulates P/E by sector across runs
# ---------------------------------------------------------------------------

_SECTOR_PE_CACHE_PATH = Path(__file__).parent.parent / "feature_cache" / "sector_pe_cache.json"

# In-memory: {sector: [pe1, pe2, ...]}  — populated by analyse() calls
_sector_pe_data: dict[str, list[float]] = {}
_sector_pe_loaded = False


def _load_sector_pe_cache() -> None:
    """Load persisted sector P/E data from disk (once per process)."""
    global _sector_pe_data, _sector_pe_loaded
    if _sector_pe_loaded:
        return
    _sector_pe_loaded = True

    if _SECTOR_PE_CACHE_PATH.exists():
        try:
            with open(_SECTOR_PE_CACHE_PATH, "r") as f:
                raw = json.load(f)
            # Validate structure
            if isinstance(raw, dict):
                for sector, pe_list in raw.items():
                    if isinstance(pe_list, list):
                        valid = [float(p) for p in pe_list
                                 if isinstance(p, (int, float)) and 0 < p < 500]
                        if valid:
                            _sector_pe_data[sector] = valid
            logger.info("Sector PE cache loaded: %d sectors, %d total entries",
                        len(_sector_pe_data),
                        sum(len(v) for v in _sector_pe_data.values()))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Sector PE cache load failed: %s", e)
            _sector_pe_data = {}


def _save_sector_pe_cache() -> None:
    """Persist sector P/E cache to disk."""
    try:
        _SECTOR_PE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Keep only the most recent 50 P/E values per sector (rolling window)
        trimmed = {s: pes[-50:] for s, pes in _sector_pe_data.items() if pes}
        tmp = _SECTOR_PE_CACHE_PATH.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(trimmed, f, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(_SECTOR_PE_CACHE_PATH)
    except OSError as e:
        logger.warning("Sector PE cache save failed: %s", e)


def _register_sector_pe(sector: str, pe: float) -> None:
    """Register a ticker's P/E ratio for its sector (accumulates over time)."""
    if not sector or pe is None or pe <= 0 or pe > 500:
        return
    _sector_pe_data.setdefault(sector, []).append(pe)


def _get_sector_median_pe(sector: str) -> float | None:
    """Get median P/E for a sector from accumulated data. Needs ≥3 samples."""
    pes = _sector_pe_data.get(sector)
    if not pes or len(pes) < 3:
        return None
    return float(np.median(pes))


# ---------------------------------------------------------------------------
# FMP data fetching (returns None when FMP unavailable)
# ---------------------------------------------------------------------------

def _get_fmp_fundamentals(ticker: str) -> dict | None:
    """Fetch FMP fundamental data. Returns None if FMP unavailable."""
    try:
        from utils.fmp_client import (
            get_earnings_surprises, get_analyst_estimates,
            get_key_metrics, get_income_statement,
            get_upgrades_downgrades, get_company_profile,
            get_sector_pe, get_earnings_calendar, is_available,
        )
        if not is_available():
            return None

        profile = get_company_profile(ticker)
        sector = profile.get("sector") if profile else None

        return {
            "earnings_surprises": get_earnings_surprises(ticker, limit=8),
            "analyst_estimates": get_analyst_estimates(ticker, period="quarter", limit=4),
            "key_metrics": get_key_metrics(ticker, period="quarter", limit=8),
            "income_statement": get_income_statement(ticker, period="quarter", limit=8),
            "upgrades_downgrades": get_upgrades_downgrades(ticker),
            "earnings_calendar": get_earnings_calendar(ticker),
            "sector": sector,
            "sector_pe": get_sector_pe(sector) if sector else None,
            "profile": profile,
        }
    except Exception as e:
        logger.debug("FMP fundamentals fetch failed for %s: %s", ticker, e)
        return None


# ---------------------------------------------------------------------------
# FMP scoring helpers
# ---------------------------------------------------------------------------

def _score_earnings_surprises(fmp: dict) -> tuple[float, list[str], str | None]:
    """Score earnings beat/miss track record. Returns (score_delta, reasons, beat_rate_str)."""
    surprises = fmp.get("earnings_surprises")
    if not surprises or not isinstance(surprises, list) or len(surprises) < 4:
        return 0.0, [], None

    recent = surprises[:4]  # newest first
    beats = 0
    misses = 0
    for s in recent:
        actual = s.get("actualEarningResult")
        estimated = s.get("estimatedEarning")
        if actual is not None and estimated is not None:
            if actual > estimated:
                beats += 1
            elif actual < estimated:
                misses += 1

    beat_rate = f"{beats}/4"
    score = 0.0
    reasons = []

    if beats >= 4:
        score = 0.15
        reasons.append(f"beat EPS estimates {beat_rate} quarters")
    elif beats >= 3:
        score = 0.10
        reasons.append(f"beat EPS estimates {beat_rate} quarters")
    elif beats <= 1:
        score = -0.10
        reasons.append(f"missed EPS estimates {misses}/4 quarters")

    return score, reasons, beat_rate


def _score_quarterly_trends(fmp: dict) -> tuple[float, list[str], str | None]:
    """Score quarterly trends in EPS, revenue, margins. Returns (score_delta, reasons, trend_str)."""
    statements = fmp.get("income_statement")
    if not statements or not isinstance(statements, list) or len(statements) < 4:
        return 0.0, [], None

    # Extract series (newest first)
    eps_series = [s.get("eps") for s in statements[:4] if s.get("eps") is not None]
    rev_series = [s.get("revenue") for s in statements[:4] if s.get("revenue") is not None]
    margin_series = [s.get("netIncomeRatio") for s in statements[:4] if s.get("netIncomeRatio") is not None]

    score = 0.0
    reasons = []
    trend_parts = []

    # Count consecutive improvements (newest to oldest → reversed for comparison)
    def _count_consecutive_improvements(series):
        if len(series) < 2:
            return 0, 0
        improving = 0
        declining = 0
        for i in range(len(series) - 1):
            if series[i] > series[i + 1]:  # newest > older = improving
                improving += 1
            elif series[i] < series[i + 1]:
                declining += 1
        return improving, declining

    # EPS trend
    if len(eps_series) >= 4:
        imp, dec = _count_consecutive_improvements(eps_series)
        if imp >= 3:
            score += 0.10
            reasons.append("EPS improving 3+ consecutive quarters")
            trend_parts.append("EPS↑")
        elif dec >= 3:
            score -= 0.10
            reasons.append("EPS declining 3+ consecutive quarters")
            trend_parts.append("EPS↓")

    # Revenue trend
    if len(rev_series) >= 4:
        imp, dec = _count_consecutive_improvements(rev_series)
        if imp >= 3:
            score += 0.05
            trend_parts.append("Rev↑")
        elif dec >= 3:
            score -= 0.05
            trend_parts.append("Rev↓")

    # Margin trend
    if len(margin_series) >= 4:
        imp, dec = _count_consecutive_improvements(margin_series)
        if imp >= 3:
            score += 0.05
            trend_parts.append("Margin↑")
        elif dec >= 3:
            score -= 0.05
            trend_parts.append("Margin↓")

    # Cap at ±0.15
    score = max(-0.15, min(0.15, score))
    trend_str = ", ".join(trend_parts) if trend_parts else None

    return score, reasons, trend_str


def _score_analyst_revisions(fmp: dict) -> tuple[float, list[str], str | None]:
    """Score analyst estimate revisions and upgrade/downgrade momentum."""
    score = 0.0
    reasons = []
    revision_str = None

    # Analyst estimates — compare recent forward estimate changes
    estimates = fmp.get("analyst_estimates")
    if estimates and isinstance(estimates, list) and len(estimates) >= 2:
        try:
            current_eps = estimates[0].get("estimatedEpsAvg")
            older_eps = estimates[-1].get("estimatedEpsAvg")
            if current_eps and older_eps and older_eps != 0:
                revision_pct = ((current_eps - older_eps) / abs(older_eps)) * 100
                if revision_pct > 5:
                    score += 0.10
                    reasons.append(f"analyst EPS estimates revised up +{revision_pct:.0f}%")
                    revision_str = f"+{revision_pct:.0f}%"
                elif revision_pct < -5:
                    score -= 0.10
                    reasons.append(f"analyst EPS estimates revised down {revision_pct:.0f}%")
                    revision_str = f"{revision_pct:.0f}%"
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    # Analyst consensus (grades-consensus: strongBuy/buy/hold/sell/strongSell)
    grades = fmp.get("upgrades_downgrades")
    upgrades = 0
    downgrades = 0
    if grades and isinstance(grades, list) and grades:
        g = grades[0]
        if "strongBuy" in g or "buy" in g:
            # New format: aggregated consensus counts from /grades-consensus
            buy_count = (g.get("strongBuy", 0) or 0) + (g.get("buy", 0) or 0)
            sell_count = (g.get("sell", 0) or 0) + (g.get("strongSell", 0) or 0)
            upgrades = buy_count
            downgrades = sell_count
            # Strong consensus signal
            total = buy_count + (g.get("hold", 0) or 0) + sell_count
            if total > 0:
                buy_ratio = buy_count / total
                if buy_ratio >= 0.70:
                    score += 0.05
                    reasons.append(f"strong analyst consensus: {buy_count} buy vs {sell_count} sell")
                elif buy_ratio <= 0.30:
                    score -= 0.05
                    reasons.append(f"weak analyst consensus: {buy_count} buy vs {sell_count} sell")
        else:
            # Legacy format: individual upgrade/downgrade actions
            cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
            for entry in grades:
                pub_date = entry.get("publishedDate", "")[:10]
                if pub_date >= cutoff:
                    action = (entry.get("action") or "").lower()
                    if "upgrade" in action:
                        upgrades += 1
                    elif "downgrade" in action:
                        downgrades += 1

            if upgrades - downgrades >= 2:
                score += 0.05
            elif downgrades - upgrades >= 2:
                score -= 0.05

    # Cap at ±0.10
    score = max(-0.10, min(0.10, score))

    return score, reasons, revision_str, upgrades, downgrades


def _score_peg_ratio(fmp: dict | None, pe_ratio: float | None = None,
                     eps_growth: float | None = None) -> tuple[float, list[str], float | None]:
    """Score PEG ratio. Tries FMP first, then computes locally from yfinance P/E and EPS growth."""
    peg = None

    # Try FMP key_metrics first
    if fmp:
        metrics = fmp.get("key_metrics")
        if metrics and isinstance(metrics, list) and metrics:
            fmp_peg = metrics[0].get("pegRatio")
            if fmp_peg is not None:
                try:
                    peg = float(fmp_peg)
                except (TypeError, ValueError):
                    pass

    # Fallback: compute PEG locally from yfinance P/E and EPS growth
    if peg is None and pe_ratio is not None and eps_growth is not None:
        if pe_ratio > 0 and eps_growth > 0.01:  # Only meaningful with positive P/E and growth
            eps_growth_pct = eps_growth * 100  # PEG = P/E ÷ EPS growth %
            if eps_growth_pct > 0:
                peg = pe_ratio / eps_growth_pct

    if peg is None:
        return 0.0, [], None

    score = 0.0
    reasons = []

    if peg < 0:
        score = -0.10
        reasons.append(f"negative PEG ({peg:.1f}, declining growth)")
    elif peg < 1.0:
        score = 0.10
        reasons.append(f"attractive PEG ratio ({peg:.1f})")
    elif peg < 1.5:
        score = 0.05
    elif peg > 2.0:
        score = -0.05
        reasons.append(f"expensive PEG ratio ({peg:.1f})")

    return score, reasons, peg


def _score_sector_relative_pe(pe_ratio, fmp: dict,
                               fallback_sector: str | None = None) -> tuple[float, list[str], float | None, str | None]:
    """Score P/E relative to sector median.

    Uses FMP sector_pe when available (US stocks). For non-US stocks, falls back
    to the locally accumulated sector P/E cache (median of observed P/Es per sector).
    """
    sector_pe = fmp.get("sector_pe") if fmp else None

    # Fallback: use locally accumulated sector median P/E
    if sector_pe is None and fallback_sector:
        sector_pe = _get_sector_median_pe(fallback_sector)

    if pe_ratio is None or sector_pe is None or sector_pe <= 0:
        return 0.0, [], sector_pe, None

    try:
        premium_pct = ((pe_ratio - sector_pe) / sector_pe) * 100
    except (TypeError, ZeroDivisionError):
        return 0.0, [], sector_pe, None

    score = 0.0
    reasons = []
    pe_vs_str = f"{premium_pct:+.0f}% vs sector"

    if premium_pct > 50:
        score = -0.10
        reasons.append(f"P/E {premium_pct:.0f}% above sector median")
    elif premium_pct > 20:
        score = -0.05
    elif premium_pct < -50:
        score = 0.10
        reasons.append(f"P/E {abs(premium_pct):.0f}% below sector median")
    elif premium_pct < -20:
        score = 0.05

    return score, reasons, sector_pe, pe_vs_str


def _get_next_earnings_date(fmp: dict) -> tuple[str | None, int | None]:
    """Find next earnings date and days until it."""
    calendar = fmp.get("earnings_calendar")
    if not calendar or not isinstance(calendar, list):
        return None, None

    today = datetime.now().date()
    for entry in calendar:
        try:
            ed = datetime.strptime(entry.get("date", "")[:10], "%Y-%m-%d").date()
            if ed >= today:
                days_until = (ed - today).days
                return entry["date"][:10], days_until
        except (ValueError, TypeError):
            continue
    return None, None


# ---------------------------------------------------------------------------
# Dividend safety scoring
# ---------------------------------------------------------------------------

def _score_dividend_safety(info: dict) -> tuple[float, list[str], dict]:
    """Score dividend yield attractiveness, payout sustainability, ex-div proximity.

    Returns (score_delta, reasons, dividend_data_dict).
    Data comes from yfinance info dict (already fetched, no extra API calls).
    """
    score = 0.0
    reasons = []
    data: dict = {
        "dividend_yield": None,
        "payout_ratio": None,
        "ex_dividend_date": None,
        "ex_dividend_days": None,
        "five_year_avg_yield": None,
    }

    div_yield = info.get("dividendYield")  # ratio, e.g. 0.047 = 4.7%
    payout = info.get("payoutRatio")
    five_yr = info.get("fiveYearAvgDividendYield")  # percentage, e.g. 3.2
    ex_div_ts = info.get("exDividendDate")  # Unix timestamp or None

    if div_yield is not None and div_yield > 0:
        data["dividend_yield"] = div_yield

        # Yield attractiveness vs 5yr average
        if five_yr is not None and five_yr > 0:
            five_yr_ratio = five_yr / 100.0  # Convert percentage to ratio
            data["five_year_avg_yield"] = five_yr_ratio
            if div_yield > five_yr_ratio * 1.2:
                score += 0.10
                reasons.append(f"yield {div_yield:.1%} above 5yr avg {five_yr_ratio:.1%}")
            elif div_yield < five_yr_ratio * 0.8:
                score -= 0.05

        # Absolute yield tiers
        if div_yield > config.DIVIDEND_YIELD_TRAP_THRESHOLD:
            score -= 0.05
            reasons.append(f"yield {div_yield:.1%} may be a trap (>{config.DIVIDEND_YIELD_TRAP_THRESHOLD:.0%})")
        elif div_yield > 0.04:
            score += 0.05
            reasons.append(f"attractive yield {div_yield:.1%}")

    if payout is not None and payout >= 0:
        data["payout_ratio"] = payout
        if payout < config.DIVIDEND_PAYOUT_HEALTHY:
            score += 0.05
        elif payout > config.DIVIDEND_PAYOUT_UNSUSTAINABLE:
            score -= 0.10
            reasons.append(f"unsustainable payout ratio ({payout:.0%})")
        elif payout > config.DIVIDEND_PAYOUT_STRETCHED:
            score -= 0.05
            reasons.append(f"stretched payout ratio ({payout:.0%})")

    # Ex-dividend proximity (metadata only, no score impact)
    if ex_div_ts is not None:
        try:
            ex_date = datetime.fromtimestamp(int(ex_div_ts))
            today = datetime.now()
            days_until = (ex_date - today).days
            if days_until >= 0:
                data["ex_dividend_date"] = ex_date.strftime("%Y-%m-%d")
                data["ex_dividend_days"] = days_until
        except (ValueError, TypeError, OSError):
            pass

    score = max(-0.15, min(0.15, score))
    return score, reasons, data


# ---------------------------------------------------------------------------
# Balance sheet strength scoring
# ---------------------------------------------------------------------------

def _score_balance_sheet_strength(info: dict) -> tuple[float, list[str], dict]:
    """Score balance sheet health: net debt/EBITDA, current ratio, cash-to-debt.

    Returns (score_delta, reasons, balance_sheet_data_dict).
    Data comes from yfinance info dict (already fetched).
    """
    score = 0.0
    reasons = []
    data: dict = {
        "current_ratio": None,
        "net_debt_ebitda": None,
        "cash_to_debt": None,
        "balance_sheet_grade": None,
    }

    total_debt = info.get("totalDebt")
    total_cash = info.get("totalCash")
    ebitda = info.get("ebitda")
    current_ratio = info.get("currentRatio")

    # Net debt / EBITDA
    if total_debt is not None and total_cash is not None and ebitda is not None and ebitda > 0:
        net_debt = total_debt - total_cash
        nd_ebitda = net_debt / ebitda
        data["net_debt_ebitda"] = round(nd_ebitda, 2)

        if net_debt < 0:
            score += 0.15
            reasons.append(f"fortress balance sheet (net cash, ND/EBITDA {nd_ebitda:.1f}x)")
        elif nd_ebitda > config.NET_DEBT_EBITDA_DANGER:
            score -= 0.20
            reasons.append(f"dangerously leveraged (ND/EBITDA {nd_ebitda:.1f}x)")
        elif nd_ebitda > config.NET_DEBT_EBITDA_HIGH:
            score -= 0.10
            reasons.append(f"high leverage (ND/EBITDA {nd_ebitda:.1f}x)")

    # Current ratio
    if current_ratio is not None:
        data["current_ratio"] = round(current_ratio, 2)
        if current_ratio < config.CURRENT_RATIO_MIN:
            score -= 0.10
            reasons.append(f"liquidity risk (current ratio {current_ratio:.1f})")
        elif current_ratio > 1.5:
            score += 0.05

    # Cash-to-debt ratio
    if total_debt is not None and total_debt > 0 and total_cash is not None:
        c2d = total_cash / total_debt
        data["cash_to_debt"] = round(c2d, 2)
        if c2d > 1.0:
            score += 0.05
        elif c2d < 0.2:
            score -= 0.05

    # Letter grade for UI display
    score_clamped = max(-0.20, min(0.20, score))
    if score_clamped >= 0.12:
        data["balance_sheet_grade"] = "A"
    elif score_clamped >= 0.03:
        data["balance_sheet_grade"] = "B"
    elif score_clamped >= -0.05:
        data["balance_sheet_grade"] = "C"
    else:
        data["balance_sheet_grade"] = "D"

    return score_clamped, reasons, data


# ---------------------------------------------------------------------------
# Governance red flag composite (proxy from existing signals)
# ---------------------------------------------------------------------------

def _compute_governance_flag(
    insider_buys: int,
    insider_sells: int,
    short_pct: float | None,
    analyst_upside: float | None,
    quarterly_trend: str | None,
    earnings_beat_rate: str | None,
) -> tuple[bool, list[str]]:
    """Detect governance concern from co-occurring warning signals.

    No score impact — constituent signals already score individually.
    This flag surfaces a composite UI warning when multiple red flags align
    (the GCT pattern from professional analysis: extreme insider selling +
    high short interest + price above consensus + declining margins).
    """
    warnings = []

    # 1. Extreme insider selling with zero offsetting purchases
    if insider_sells >= 5 and insider_buys == 0:
        warnings.append(f"extreme insider selling ({insider_sells} sells, 0 buys)")

    # 2. Elevated short interest (>10%)
    if short_pct is not None and short_pct > 0.10:
        warnings.append(f"elevated short interest ({short_pct:.1%})")

    # 3. Price significantly above analyst target
    if analyst_upside is not None and analyst_upside < -10:
        warnings.append(f"trading {abs(analyst_upside):.0f}% above analyst target")

    # 4. Declining margins
    if quarterly_trend and "Margin" in quarterly_trend and "\u2193" in quarterly_trend:
        warnings.append("declining profit margins")

    # 5. Poor earnings execution
    if earnings_beat_rate and earnings_beat_rate in ("0/4", "1/4"):
        warnings.append(f"poor earnings track record ({earnings_beat_rate})")

    threshold = getattr(config, "GOVERNANCE_FLAG_THRESHOLD", 3)
    return len(warnings) >= threshold, warnings


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse(ticker: str) -> dict:
    """Run fundamental analysis. Returns dict of metrics + a score from -1 to 1.

    FMP is tried FIRST as primary data source; yfinance fills any gaps.
    Sector-relative P/E works for ALL tickers via local sector PE cache fallback.
    """
    # Ensure sector PE cache is loaded (once per process)
    _load_sector_pe_cache()

    info = get_ticker_info(ticker)
    if not info:
        info = {}  # FMP can still provide data even if yfinance fails

    # Fetch FMP data first (primary source — Starter plan: 300 calls/min)
    fmp = _get_fmp_fundamentals(ticker)

    # Use FMP profile to supplement/override yfinance when available
    fmp_profile = fmp.get("profile") if fmp else None
    fmp_metrics = None
    if fmp and fmp.get("key_metrics") and isinstance(fmp["key_metrics"], list):
        fmp_metrics = fmp["key_metrics"][0]  # Most recent quarter

    # Core metrics: prefer FMP quarterly data, fall back to yfinance
    pe_ratio = info.get("trailingPE") or info.get("forwardPE")
    if pe_ratio is None and fmp_metrics:
        pe_ratio = fmp_metrics.get("peRatio")

    # Determine sector: FMP profile > yfinance info
    ticker_sector = None
    if fmp and fmp.get("sector"):
        ticker_sector = fmp["sector"]
    elif info.get("sector"):
        ticker_sector = info["sector"]

    # Register P/E into sector cache for cross-sectional comparison
    if ticker_sector and pe_ratio is not None and pe_ratio > 0:
        _register_sector_pe(ticker_sector, pe_ratio)

    eps_growth = info.get("earningsGrowth")
    debt_to_equity = info.get("debtToEquity")
    if debt_to_equity is None and fmp_metrics:
        de = fmp_metrics.get("debtToEquity")
        if de is not None:
            debt_to_equity = de * 100  # FMP returns ratio, yfinance uses percentage

    short_pct = info.get("shortPercentOfFloat")
    short_ratio = info.get("shortRatio")
    inst_ownership = info.get("heldPercentInstitutions")
    insider_ownership = info.get("heldPercentInsiders")

    analyst_target = info.get("targetMeanPrice")
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    if current_price is None and fmp_profile:
        current_price = fmp_profile.get("price")
    analyst_rec = info.get("recommendationKey")
    num_analysts = info.get("numberOfAnalystOpinions")

    revenue_growth = info.get("revenueGrowth")
    profit_margin = info.get("profitMargins")
    if profit_margin is None and fmp_metrics:
        profit_margin = fmp_metrics.get("netIncomePerRevenue")

    roe = info.get("returnOnEquity")
    if roe is None and fmp_metrics:
        roe = fmp_metrics.get("roe")

    fcf = info.get("freeCashflow")
    market_cap = info.get("marketCap")
    if market_cap is None and fmp_profile:
        market_cap = fmp_profile.get("mktCap")

    # If both sources failed, return empty
    if not info and fmp is None:
        return _empty_result("No fundamental data available")

    fcf_yield = None
    if fcf is not None and market_cap and market_cap > 0:
        fcf_yield = fcf / market_cap

    reasons = []

    # -----------------------------------------------------------------------
    # Structured into 4 orthogonal sub-factors to prevent double-counting.
    # Each sub-factor is scored -1 to +1 and capped independently.
    # Academic basis: Fama-French (value), Novy-Marx 2013 (quality),
    # Chan/Jegadeesh/Lakonishok 1996 (institutional/revisions).
    # -----------------------------------------------------------------------

    # --- Sub-factor 1: VALUE (30%) ---
    # P/E, PEG, sector-relative P/E, FCF yield
    value_score = 0.0
    if pe_ratio is not None:
        if pe_ratio < 0:
            value_score -= 0.6
            reasons.append(f"negative P/E ({pe_ratio:.1f})")
        elif pe_ratio > 35:
            value_score -= 0.5
            reasons.append(f"high P/E ({pe_ratio:.1f})")
        elif pe_ratio > 25:
            value_score -= 0.2
            reasons.append(f"elevated P/E ({pe_ratio:.1f})")
        elif pe_ratio < 10:
            value_score += 0.5
            reasons.append(f"low P/E ({pe_ratio:.1f})")
        elif pe_ratio < 15:
            value_score += 0.3
            reasons.append(f"attractive P/E ({pe_ratio:.1f})")
        else:
            value_score += 0.1
    else:
        reasons.append("P/E unavailable")

    if fcf_yield is not None:
        if fcf_yield > 0.08:
            value_score += 0.3
            reasons.append(f"strong FCF yield ({fcf_yield:.1%})")
        elif fcf_yield > 0.04:
            value_score += 0.1
        elif fcf_yield < -0.02:
            value_score -= 0.2
            reasons.append(f"negative FCF yield ({fcf_yield:.1%})")

    value_score = max(-1.0, min(1.0, value_score))

    # --- Sub-factor 2: QUALITY (30%) ---
    # ROE, profit margin, D/E, short interest
    quality_score = 0.0
    if roe is not None:
        if roe > 0.25:
            quality_score += 0.4
            reasons.append(f"high ROE ({roe:.0%})")
        elif roe > 0.15:
            quality_score += 0.2
        elif roe < 0:
            quality_score -= 0.3
    if profit_margin is not None:
        if profit_margin > 0.20:
            quality_score += 0.3
        elif profit_margin < 0:
            quality_score -= 0.4
            reasons.append(f"unprofitable (margin {profit_margin:.0%})")
        elif profit_margin < 0.05:
            quality_score -= 0.1
    if debt_to_equity is not None:
        if debt_to_equity > 200:
            quality_score -= 0.3
            reasons.append(f"high leverage (D/E {debt_to_equity:.0f}%)")
        elif debt_to_equity > 100:
            quality_score -= 0.1
        elif debt_to_equity < 50:
            quality_score += 0.2
    else:
        reasons.append("D/E unavailable")
    if short_pct is not None:
        if short_pct > config.SHORT_INTEREST_HIGH:
            quality_score -= 0.2
            reasons.append(f"high short interest ({short_pct:.1%})")
        elif short_pct < config.SHORT_INTEREST_LOW:
            quality_score += 0.1

    quality_score = max(-1.0, min(1.0, quality_score))

    # --- Sub-factor 3: GROWTH (25%) ---
    # EPS growth, revenue growth, quarterly trends
    growth_score = 0.0
    if eps_growth is not None:
        if eps_growth > 0.20:
            growth_score += 0.4
            reasons.append(f"strong EPS growth ({eps_growth:.0%})")
        elif eps_growth > 0.05:
            growth_score += 0.15
        elif eps_growth < -0.10:
            growth_score -= 0.5
            reasons.append(f"declining earnings ({eps_growth:.0%})")
        elif eps_growth < 0:
            growth_score -= 0.25
            reasons.append(f"negative EPS growth ({eps_growth:.0%})")
    else:
        reasons.append("EPS growth unavailable")

    if revenue_growth is not None:
        if revenue_growth > 0.15:
            growth_score += 0.3
            reasons.append(f"strong revenue growth ({revenue_growth:.0%})")
        elif revenue_growth > 0.05:
            growth_score += 0.1
        elif revenue_growth < -0.10:
            growth_score -= 0.3
            reasons.append(f"declining revenue ({revenue_growth:.0%})")
        elif revenue_growth < 0:
            growth_score -= 0.1

    growth_score = max(-1.0, min(1.0, growth_score))

    # --- Sub-factor 4: INSTITUTIONAL (15%) ---
    # Analyst consensus, insider activity, inst. ownership, analyst target
    inst_score = 0.0

    insider_txns = get_insider_transactions(ticker)
    insider_buys = insider_txns.get("buys", 0)
    insider_sells = insider_txns.get("sells", 0)
    insider_net = insider_txns.get("net_label", "")

    if insider_buys > 0 and insider_buys > insider_sells:
        inst_score += 0.3
        reasons.append(f"insider buying ({insider_buys} buys vs {insider_sells} sells)")
    elif insider_sells > 0 and insider_sells > insider_buys:
        inst_score -= 0.2
        reasons.append(f"insider selling ({insider_sells} sells vs {insider_buys} buys)")

    if inst_ownership is not None:
        if inst_ownership > config.INST_OWNERSHIP_HIGH:
            inst_score += 0.2
            reasons.append(f"strong institutional backing ({inst_ownership:.0%})")
        elif inst_ownership < 0.10:
            inst_score -= 0.2
            reasons.append(f"low institutional interest ({inst_ownership:.0%})")

    analyst_upside = None
    if analyst_target and current_price and current_price > 0 and num_analysts and num_analysts >= 3:
        analyst_upside = ((analyst_target - current_price) / current_price) * 100
        if analyst_upside > 20:
            inst_score += 0.3
            reasons.append(f"analysts target +{analyst_upside:.0f}% upside")
        elif analyst_upside > 10:
            inst_score += 0.15
        elif analyst_upside < -15:
            inst_score -= 0.3
            reasons.append(f"analysts target {analyst_upside:.0f}% downside")
        elif analyst_upside < -5:
            inst_score -= 0.15

    if analyst_rec and num_analysts and num_analysts >= 3:
        if analyst_rec in ("strong_buy", "buy"):
            inst_score += 0.15
        elif analyst_rec in ("sell", "strong_sell"):
            inst_score -= 0.2
            reasons.append(f"analyst consensus: {analyst_rec.replace('_', ' ')}")

    inst_score = max(-1.0, min(1.0, inst_score))

    # --- Blend sub-factors ---
    score = (
        0.30 * value_score
        + 0.30 * quality_score
        + 0.25 * growth_score
        + 0.15 * inst_score
    )

    # --- FMP-enhanced scoring (supplements sub-factors) ---
    fmp_available = fmp is not None
    earnings_beat_rate = None
    quarterly_trend = None
    estimate_revision = None
    peg_ratio = None
    sector_pe = None
    pe_vs_sector = None
    recent_upgrades = 0
    recent_downgrades = 0
    next_earnings_date = None
    earnings_proximity_days = None

    if fmp:
        # Earnings surprise track record (boosts quality/growth confidence)
        s_delta, s_reasons, earnings_beat_rate = _score_earnings_surprises(fmp)
        score += s_delta * 0.5  # Scale down — already captured in sub-factors
        reasons.extend(s_reasons)

        # Quarterly trends (growth confirmation)
        t_delta, t_reasons, quarterly_trend = _score_quarterly_trends(fmp)
        score += t_delta * 0.5
        reasons.extend(t_reasons)

        # Analyst revisions (institutional signal)
        r_delta, r_reasons, estimate_revision, recent_upgrades, recent_downgrades = (
            _score_analyst_revisions(fmp)
        )
        score += r_delta * 0.5
        reasons.extend(r_reasons)

        # Earnings calendar
        next_earnings_date, earnings_proximity_days = _get_next_earnings_date(fmp)

    # Sector-relative P/E — supplements value sub-factor
    sr_delta, sr_reasons, sector_pe, pe_vs_sector = _score_sector_relative_pe(
        pe_ratio, fmp or {}, fallback_sector=ticker_sector,
    )
    score += sr_delta * 0.5
    reasons.extend(sr_reasons)

    # PEG ratio — supplements value/growth
    p_delta, p_reasons, peg_ratio = _score_peg_ratio(fmp, pe_ratio, eps_growth)
    score += p_delta * 0.5
    reasons.extend(p_reasons)

    # Analyst consensus fallback
    if recent_upgrades == 0 and recent_downgrades == 0 and not fmp:
        yf_rec = info.get("recommendationKey")
        num_a = info.get("numberOfAnalystOpinions")
        if yf_rec and num_a and num_a >= 3:
            if yf_rec in ("strong_buy",):
                recent_upgrades = num_a
                score += 0.03
                reasons.append(f"yfinance consensus: strong buy ({num_a} analysts)")
            elif yf_rec in ("sell", "strong_sell"):
                recent_downgrades = num_a
                score -= 0.03
                reasons.append(f"yfinance consensus: {yf_rec.replace('_', ' ')} ({num_a} analysts)")

    # --- Dividend safety (supplements quality sub-factor) ---
    div_delta, div_reasons, div_data = _score_dividend_safety(info)
    score += div_delta * 0.5  # Scale down — quality sub-factor already scores some overlap
    reasons.extend(div_reasons)

    # --- Balance sheet strength (supplements quality sub-factor) ---
    bs_delta, bs_reasons, bs_data = _score_balance_sheet_strength(info)
    score += bs_delta * 0.5
    reasons.extend(bs_reasons)

    score = max(-1.0, min(1.0, score))

    # --- Governance red flag (metadata only — no score impact) ---
    governance_flag, governance_reasons = _compute_governance_flag(
        insider_buys=insider_buys,
        insider_sells=insider_sells,
        short_pct=short_pct,
        analyst_upside=analyst_upside,
        quarterly_trend=quarterly_trend,
        earnings_beat_rate=earnings_beat_rate,
    )

    # Persist sector PE cache periodically (lightweight — only writes if data changed)
    _save_sector_pe_cache()

    return {
        "score": score,
        "reasons": reasons,
        "sector": ticker_sector,
        # yfinance metrics
        "pe_ratio": pe_ratio,
        "eps_growth": eps_growth,
        "debt_to_equity": debt_to_equity,
        "short_pct": short_pct,
        "short_ratio": short_ratio,
        "inst_ownership": inst_ownership,
        "insider_ownership": insider_ownership,
        "insider_buys": insider_buys,
        "insider_sells": insider_sells,
        "insider_net": insider_net,
        "insider_transactions": insider_txns.get("recent", []),
        "analyst_target": analyst_target,
        "analyst_upside": analyst_upside,
        "analyst_rec": analyst_rec,
        "num_analysts": num_analysts,
        "revenue_growth": revenue_growth,
        "profit_margin": profit_margin,
        "roe": roe,
        "fcf_yield": fcf_yield,
        "market_cap": market_cap,
        # FMP metrics
        "fmp_available": fmp_available,
        "earnings_beat_rate": earnings_beat_rate,
        "quarterly_trend": quarterly_trend,
        "estimate_revision": estimate_revision,
        "peg_ratio": peg_ratio,
        "sector_pe": sector_pe,
        "pe_vs_sector": pe_vs_sector,
        "recent_upgrades": recent_upgrades,
        "recent_downgrades": recent_downgrades,
        "next_earnings_date": next_earnings_date,
        "earnings_proximity_days": earnings_proximity_days,
        # Dividend safety
        "dividend_yield": div_data.get("dividend_yield"),
        "payout_ratio": div_data.get("payout_ratio"),
        "ex_dividend_date": div_data.get("ex_dividend_date"),
        "ex_dividend_days": div_data.get("ex_dividend_days"),
        "five_year_avg_yield": div_data.get("five_year_avg_yield"),
        # Balance sheet strength
        "current_ratio": bs_data.get("current_ratio"),
        "net_debt_ebitda": bs_data.get("net_debt_ebitda"),
        "cash_to_debt": bs_data.get("cash_to_debt"),
        "balance_sheet_grade": bs_data.get("balance_sheet_grade"),
        # Governance red flag
        "governance_flag": governance_flag,
        "governance_reasons": governance_reasons,
    }


def _empty_result(reason: str) -> dict:
    return {
        "score": 0.0,
        "reasons": [reason],
        "sector": None,
        "pe_ratio": None, "eps_growth": None, "debt_to_equity": None,
        "short_pct": None, "short_ratio": None,
        "inst_ownership": None, "insider_ownership": None,
        "insider_buys": 0, "insider_sells": 0, "insider_net": "",
        "insider_transactions": [],
        "analyst_target": None, "analyst_upside": None,
        "analyst_rec": None, "num_analysts": None,
        "revenue_growth": None, "profit_margin": None,
        "roe": None, "fcf_yield": None, "market_cap": None,
        "fmp_available": False,
        "earnings_beat_rate": None, "quarterly_trend": None,
        "estimate_revision": None, "peg_ratio": None,
        "sector_pe": None, "pe_vs_sector": None,
        "recent_upgrades": 0, "recent_downgrades": 0,
        "next_earnings_date": None, "earnings_proximity_days": None,
        # Dividend safety
        "dividend_yield": None, "payout_ratio": None,
        "ex_dividend_date": None, "ex_dividend_days": None,
        "five_year_avg_yield": None,
        # Balance sheet strength
        "current_ratio": None, "net_debt_ebitda": None,
        "cash_to_debt": None, "balance_sheet_grade": None,
        # Governance red flag
        "governance_flag": False, "governance_reasons": [],
    }
