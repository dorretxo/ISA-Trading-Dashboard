"""Global Discovery Engine v4 — Multi-Lens + Wider Funnel.

Architecture:
- Feature Store caches batch price factors daily (cheap, runs on full universe)
- Multi-lens prescreen: momentum + value entry paths before fundamental quality is available
- Medium-cost tier: lightweight fundamentals on 250 before deep analysis on 60
- Soft thresholds: graduated penalties replace hard cuts
- Cross-sectional normalization: scores ranked within sector/cap peers
- Fundamental quality lens introduced once gross profitability / ROE / FCF-assets are available
- Diversified final selector: sector caps without rigid geographic minimums

Funnel stages:
1. Universe Assembly → ~2000-3000 candidates (FMP US + yfinance global)
2. Multi-Lens Screen → ~250 (momentum + value + composite)
3. Quick Filter      → ~220 (soft penalties for beta, penny stock, volume)
4. Correlation Filter → soft penalty (no hard rejection)
5. Quick Rank        → top 60 (medium-cost tier: momentum + value + fundamental quality)
6. Full Scoring      → 60 scored (analyse_holding pipeline)
7. FX + Fit + Diversification → final ranked list with sector balancing
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

import config
from engine.factor_momentum import (
    calculate_network_momentum,
    compute_factor_returns,
    get_factor_tilt_features,
    load_cached_factor_returns,
    load_optional_supply_chain_mapping,
)
from engine.factors import (
    adjust_alpha_for_confidence,
    compute_all_extended_factors,
    compute_factor_scores_from_result,
    compute_fundamental_quality_metrics,
    factor_tilt_adjustment,
)
from engine.enterprise_factors import compute_piotroski_f_score
from engine.fscore_utils import (
    f_score_min_coverage,
    is_f_score_actionable,
    normalize_f_score_coverage,
)
from engine.institutional_prior import neutral_prior, score_universe
from utils.atomic_io import atomic_write_json
from utils import pit_store
from utils.fmp_client import (
    screen_stocks, get_key_metrics, get_financial_ratios,
    get_company_profile, is_available,
)
from utils.feature_store import FeatureStore, compute_batch_factors
from utils.safe_numeric import safe_float

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cross_sectional_zscore(values: np.ndarray) -> np.ndarray:
    """Z-score values across batch, clamp [-3,3], rescale to [-1,1]."""
    if len(values) < 3:
        return values
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if std < 1e-8:
        return np.zeros_like(values)
    z = (values - mean) / std
    z = np.clip(z, -3.0, 3.0)
    return z / 3.0


def _sector_neutral_zscore(
    values: np.ndarray,
    sectors: list[str],
    *,
    min_bucket_size: int = 4,
) -> np.ndarray:
    """Z-score within sector (cross-sector buckets only), fallback to
    cross-sectional z for sectors with too few names.  Matches how
    institutional desks neutralise sector effects before ranking on
    residual stock-specific alpha (Grinold & Kahn 2000 §4).
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return values
    n = len(values)
    out = np.zeros(n, dtype=float)
    by_sector: dict[str, list[int]] = {}
    for i, s in enumerate(sectors or ["Unknown"] * n):
        by_sector.setdefault(s or "Unknown", []).append(i)

    # Global z as fallback for small-bucket sectors
    global_z = _cross_sectional_zscore(values)

    for sec, idx_list in by_sector.items():
        if len(idx_list) < min_bucket_size:
            for i in idx_list:
                out[i] = global_z[i]
            continue
        bucket = values[idx_list]
        mean = np.nanmean(bucket)
        std = np.nanstd(bucket)
        if std < 1e-8:
            for i in idx_list:
                out[i] = 0.0
            continue
        z = (bucket - mean) / std
        z = np.clip(z, -3.0, 3.0) / 3.0
        for k, i in enumerate(idx_list):
            out[i] = float(z[k])
    return out


def _residualise_against(
    target: np.ndarray,
    regressor: np.ndarray,
) -> np.ndarray:
    """OLS-residualise ``target`` against ``regressor`` (cross-sectional).

    Returns the target minus its best linear fit on the regressor, so the
    residual is orthogonal to the regressor.  Used to decorrelate e.g. a
    quality or forecast pillar from raw price momentum before combining.
    """
    target = np.asarray(target, dtype=float)
    regressor = np.asarray(regressor, dtype=float)
    if target.size < 3 or regressor.size != target.size:
        return target
    mask = np.isfinite(target) & np.isfinite(regressor)
    if mask.sum() < 3:
        return target
    xm = regressor[mask]
    ym = target[mask]
    x_std = float(np.std(xm))
    if x_std < 1e-8:
        return target
    slope = float(np.cov(xm, ym, ddof=0)[0, 1] / np.var(xm))
    intercept = float(np.mean(ym) - slope * np.mean(xm))
    fitted = intercept + slope * regressor
    resid = target - fitted
    # Re-centre to preserve the original mean
    return resid - float(np.mean(resid[mask])) + float(np.mean(ym))


def _parse_revision_pct(value) -> float | None:
    """Parse a revision string like '+7%' into a numeric percentage."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip().replace("%", "")
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _parse_beat_rate(value) -> float | None:
    """Parse strings like '3/4' into a 0..1 beat ratio."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip()
        if "/" in text:
            num, den = text.split("/", 1)
            den_f = float(den)
            if den_f > 0:
                return float(num) / den_f
        return None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


_COUNTRY_TO_REGION = {
    "US": "US",
    "USA": "US",
    "GB": "UK",
    "DE": "Germany",
    "FR": "France",
    "ES": "Spain",
    "NL": "Netherlands",
    "IT": "Italy",
    "CH": "Switzerland",
    "SE": "Nordics",
    "DK": "Nordics",
    "FI": "Nordics",
    "NO": "Nordics",
    "CA": "Canada",
    "AU": "Australia",
    "JP": "Japan",
    "HK": "Hong Kong",
    "SG": "Singapore",
}


def _candidate_symbol(candidate: dict) -> str:
    return str(candidate.get("symbol", "")).upper()


def _candidate_region(candidate: dict) -> str:
    """Best-effort region label used for cohort-aware promotion."""
    region = candidate.get("_region")
    if region:
        return str(region)

    country = str(candidate.get("country") or candidate.get("_country") or "").upper()
    if country:
        return _COUNTRY_TO_REGION.get(country, country)

    source = str(candidate.get("_source", "")).lower()
    exchange = str(candidate.get("_exchange_query", "") or candidate.get("_exchange", "")).upper()
    if source == "fmp" or exchange in {"NYSE", "NASDAQ", "AMEX", "US"}:
        return "US"
    return "Unknown"


def _candidate_sector_label(candidate: dict) -> str:
    sector = (
        candidate.get("_yf_sector")
        or candidate.get("sector")
        or candidate.get("_sector")
        or "Unknown"
    )
    return _normalize_sector(sector.title() if isinstance(sector, str) else "Unknown")


def _candidate_liquidity_bucket(candidate: dict) -> str:
    """Group candidates by rough liquidity / capacity cohort."""
    avg_dollar_volume = safe_float(
        candidate.get("_avg_dollar_volume"),
        default=None,
    )
    if avg_dollar_volume is not None:
        if avg_dollar_volume < 1_500_000:
            return "thin"
        if avg_dollar_volume < 8_000_000:
            return "mid"
        return "liquid"

    market_cap = safe_float(
        candidate.get("_market_cap", candidate.get("marketCap")),
        default=None,
    )
    if market_cap is not None:
        if market_cap < 300_000_000:
            return "micro"
        if market_cap < 2_000_000_000:
            return "small"
        if market_cap < 10_000_000_000:
            return "mid"
        return "large"

    tier = candidate.get("_universe_tier")
    if tier == 2:
        return "mid"
    if tier == 1:
        return "large"
    return "unknown"


def _excluded_country_tokens() -> set[str]:
    return {
        str(value).strip().upper()
        for value in getattr(config, "DISCOVERY_EXCLUDED_COUNTRIES", set())
        if str(value).strip()
    }


def _candidate_exclusion_reason(candidate: dict) -> str | None:
    """Return the configured exclusion reason for a candidate, if any."""
    symbol = _candidate_symbol(candidate)
    if not symbol:
        return None

    excluded_tickers = {
        str(value).upper().strip()
        for value in getattr(config, "DISCOVERY_EXCLUDED_TICKERS", set())
        if str(value).strip()
    }
    if symbol in excluded_tickers:
        return "excluded ticker"

    excluded_suffixes = tuple(
        str(value).upper().strip()
        for value in getattr(config, "DISCOVERY_EXCLUDED_TICKER_SUFFIXES", ())
        if str(value).strip()
    )
    if excluded_suffixes and symbol.endswith(excluded_suffixes):
        return "excluded ticker suffix"

    country_tokens = _excluded_country_tokens()
    for key in ("country", "_country", "_region", "region"):
        token = str(candidate.get(key, "") or "").strip().upper()
        if token and token in country_tokens:
            return "excluded country"
    return None


def _filter_excluded_candidates(
    candidates: list[dict],
    rejections=None,
    *,
    stage: str = "universe_exclusion",
) -> list[dict]:
    """Apply configured hard universe exclusions and optionally record rejects."""
    if not candidates:
        return candidates
    kept: list[dict] = []
    removed = 0
    for candidate in candidates:
        reason = _candidate_exclusion_reason(candidate)
        if reason:
            removed += 1
            if rejections is not None:
                rejections.append(CandidateRejection(
                    _candidate_symbol(candidate),
                    candidate.get("companyName", candidate.get("name", "")),
                    candidate.get("_exchange_query", candidate.get("exchange", "")),
                    stage,
                    reason,
                ))
            continue
        kept.append(candidate)
    if removed:
        logger.info("Universe exclusions: removed %d candidates at %s", removed, stage)
    return kept


def _challenge_ticker_order() -> dict[str, int]:
    """Configured manual challenge lane, excluding anything outside universe."""
    if not bool(getattr(config, "DISCOVERY_CHALLENGE_RESERVE_ENABLED", False)):
        return {}
    order: dict[str, int] = {}
    for raw in _challenge_entries():
        symbol = raw.get("symbol") if isinstance(raw, dict) else raw
        symbol = str(symbol or "").upper().strip()
        if not symbol or symbol in order:
            continue
        candidate = {"symbol": symbol}
        if isinstance(raw, dict):
            candidate.update(raw)
        if _candidate_exclusion_reason(candidate):
            continue
        order[symbol] = len(order)
    return order


def _challenge_entries() -> list:
    """Return generated/manual challenge entries with a legacy fallback."""
    if bool(getattr(config, "DISCOVERY_CHALLENGE_AUTO_ENABLED", True)):
        try:
            from engine.challenge_candidates import get_challenge_candidates
            generated = get_challenge_candidates()
            if generated:
                return generated
        except Exception as exc:
            logger.warning("Challenge candidate generator failed; using manual fallback: %s", exc)
    return (
        list(getattr(config, "DISCOVERY_CHALLENGE_MANUAL_OVERRIDES", []) or [])
        + list(getattr(config, "DISCOVERY_CHALLENGE_TICKERS", []) or [])
    )


def _is_challenge_candidate(candidate: dict) -> bool:
    return _candidate_symbol(candidate) in _challenge_ticker_order()


def _candidate_country_hint(symbol: str) -> tuple[str, str, str]:
    """Return (country, region, exchange) hints for challenge/forced stubs."""
    symbol = str(symbol or "").upper().strip()
    if symbol.endswith(".L"):
        return ("GB", "UK", "LSE")
    if symbol.endswith(".T"):
        return ("JP", "Japan", "TSE")
    if symbol.endswith(".TO"):
        return ("CA", "Canada", "TSX")
    if symbol.endswith(".HK"):
        return ("HK", "Hong Kong", "HKEX")
    if symbol.endswith(".KS") or symbol.endswith(".KQ"):
        return ("KR", "South Korea", "KRX")
    if symbol in {"TECK"}:
        return ("CA", "Canada", "NYSE")
    return ("US", "US", "")


def _make_challenge_candidate(symbol: str, raw_entry=None) -> dict:
    country, region, exchange = _candidate_country_hint(symbol)
    if isinstance(raw_entry, dict):
        country = raw_entry.get("country", country)
        region = raw_entry.get("region", raw_entry.get("_region", region))
        exchange = raw_entry.get("exchange", exchange)
    candidate = {
        "symbol": symbol,
        "companyName": (
            raw_entry.get("companyName") or raw_entry.get("name") or symbol
            if isinstance(raw_entry, dict)
            else symbol
        ),
        "country": country,
        "sector": raw_entry.get("sector", "") if isinstance(raw_entry, dict) else "",
        "industry": raw_entry.get("industry", "") if isinstance(raw_entry, dict) else "",
        "_exchange_query": exchange,
        "_source": "challenge_coverage",
        "_region": region,
        "_universe_tier": 1,
        "_index_source": "CHALLENGE",
        "_challenge_candidate": True,
    }
    if isinstance(raw_entry, dict):
        for key in (
            "challenge_source",
            "challenge_reason",
            "challenge_score",
            "required_checks",
            "expires_at",
            "factor_groups",
            "last_seen",
        ):
            if key in raw_entry:
                candidate[key] = raw_entry[key]
        candidate["_challenge_source"] = raw_entry.get("challenge_source", raw_entry.get("source", "challenge"))
        candidate["_challenge_reason"] = raw_entry.get("challenge_reason", raw_entry.get("reason", ""))
        candidate["_challenge_score"] = raw_entry.get("challenge_score", raw_entry.get("score"))
    return candidate


_PIT_FUNDAMENTALS_CACHE: dict | None = None


def _pit_store_cache() -> dict:
    """Load the PIT fundamental store once per process for cheap-screen use."""
    global _PIT_FUNDAMENTALS_CACHE
    if _PIT_FUNDAMENTALS_CACHE is None:
        try:
            _PIT_FUNDAMENTALS_CACHE = pit_store._load_store()  # type: ignore[attr-defined]
        except Exception:
            _PIT_FUNDAMENTALS_CACHE = {"version": 1, "tickers": {}}
    return _PIT_FUNDAMENTALS_CACHE


def _parse_iso_date(value) -> date | None:
    if isinstance(value, date):
        return value
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _pit_latest_and_prior(ticker: str, as_of: date | None = None) -> tuple[dict | None, dict | None, str | None]:
    """Fast PIT lookup: latest knowable snapshot plus closest YoY prior."""
    store = _pit_store_cache()
    entries = store.get("tickers", {}).get(str(ticker).upper())
    if not entries:
        return None, None, None

    cutoff = as_of or date.today()
    available: list[tuple[date, str, dict]] = []
    for rd_str, payload in entries.items():
        rd = _parse_iso_date(rd_str)
        if rd is None or not isinstance(payload, dict):
            continue
        accepted = _parse_iso_date(payload.get("_accepted_date"))
        available_date = accepted or (rd + timedelta(days=45))
        if available_date <= cutoff:
            available.append((rd, rd_str, payload))
    if not available:
        return None, None, None
    available.sort(key=lambda item: item[0], reverse=True)
    latest_rd, latest_key, latest = available[0]
    prior = None
    prior_candidates = [item for item in available[1:] if item[0] < latest_rd]
    if prior_candidates:
        target_prior = latest_rd - timedelta(days=365)
        prior = min(prior_candidates, key=lambda item: abs((item[0] - target_prior).days))[2]
    return dict(latest), (dict(prior) if prior else None), latest_key


def _free_cashflow_from_pit(snapshot: dict | None) -> float | None:
    if not snapshot:
        return None
    ocf = safe_float(snapshot.get("operating_cashflow"), default=None)
    capex = safe_float(snapshot.get("capital_expenditure"), default=None)
    if ocf is None:
        return None
    if capex is None:
        return ocf
    return ocf + capex if capex < 0 else ocf - capex


def _candidate_market_cap(candidate: dict, snapshot: dict | None = None) -> float | None:
    return safe_float(
        candidate.get("_market_cap", candidate.get("marketCap", (snapshot or {}).get("market_cap"))),
        default=None,
    )


def _fill_missing_candidate_field(candidate: dict, key: str, value) -> None:
    """Fill Stage 5b metadata without clobbering PIT-safe Stage 2 fields."""
    if candidate.get(key) in (None, "") and value not in (None, ""):
        candidate[key] = value


_PIT_RESULT_OVERRIDE_MAP = {
    "f_score": "_f_score",
    "f_score_coverage": "_f_score_coverage",
    "f_score_score": "_f_score_score",
    "gpa": "_gpa",
    "gpa_score": "_gpa_score",
    "gross_profitability": "_gross_profitability",
    "fcf_to_assets": "_fcf_to_assets",
    "fcf_yield": "_fcf_yield",
    "ev_ebit": "_ev_ebit",
    "ebit_yield": "_pit_ebit_yield",
    "ev_ebit_score": "_ev_ebit_score",
    "net_debt_ebitda": "_net_debt_ebitda",
    "cash_to_debt": "_cash_to_debt",
    "revenue_growth": "_revenue_growth",
    "pe_ratio": "_pe_ratio",
    "market_cap": "_market_cap",
    "quality_score_fundamental": "_pit_quality_score",
}


def _current_replay_like_fundamental_features(ticker: str, candidate: dict) -> dict:
    """Return today's replay-style PIT feature bundle for live discovery."""
    price = safe_float(
        candidate.get("price", candidate.get("_last_price", candidate.get("current_price"))),
        default=None,
    )
    if price is None or price <= 0:
        return {}
    try:
        from engine.historical_replay import _fundamental_features as replay_fundamental_features

        features = replay_fundamental_features(ticker, pd.Timestamp(date.today()), signal_price=price)
        return features if isinstance(features, dict) else {}
    except Exception as exc:
        logger.debug("Replay-style PIT fundamentals unavailable for %s: %s", ticker, exc)
        return {}


def _apply_replay_like_stage2_features(out: dict, features: dict) -> None:
    """Merge replay-style PIT features into Stage 2 private fields."""
    if not features:
        return
    mapping = {
        "f_score": "_f_score",
        "f_score_coverage": "_f_score_coverage",
        "f_score_score": "_f_score_score",
        "gpa": "_gpa",
        "gpa_score": "_gpa_score",
        "gross_profitability": "_gross_profitability",
        "fcf_to_assets": "_fcf_to_assets",
        "earnings_stability": "_earnings_stability",
        "eps_growth_variance_5y": "_eps_growth_variance_5y",
        "fcf_yield": "_fcf_yield",
        "ev_ebit": "_ev_ebit",
        "ebit_yield": "_pit_ebit_yield",
        "ev_ebit_score": "_ev_ebit_score",
        "net_debt_ebitda": "_net_debt_ebitda",
        "cash_to_debt": "_cash_to_debt",
        "revenue_growth": "_revenue_growth",
        "pe_ratio": "_pe_ratio",
        "market_cap": "_market_cap",
        "quality_score_fundamental": "_pit_quality_score",
        "value_factor_score": "_pit_value_score",
    }
    for public_key, private_key in mapping.items():
        value = features.get(public_key)
        if value not in (None, ""):
            out[private_key] = value
    if features.get("f_score") is not None:
        try:
            out["_f_score"] = int(features["f_score"])
        except (TypeError, ValueError):
            pass


def _apply_pit_factor_overrides(result: dict, candidate: dict) -> dict:
    """Prefer PIT-safe Stage 2 factor inputs over live Stage 6 metadata."""
    for public_key, private_key in _PIT_RESULT_OVERRIDE_MAP.items():
        value = candidate.get(private_key)
        if value not in (None, ""):
            result[public_key] = value

    pit_earnings_yield = safe_float(candidate.get("_pit_earnings_yield"), default=None)
    pit_pe_ratio = safe_float(candidate.get("_pe_ratio"), default=None)
    if pit_pe_ratio is None and pit_earnings_yield is not None and pit_earnings_yield > 0:
        result["pe_ratio"] = 1.0 / pit_earnings_yield

    f_score = safe_float(result.get("f_score"), default=None)
    if f_score is not None:
        result["f_score_score"] = float(np.clip((f_score - 4.5) / 3.0, -1.0, 1.0))
        result["f_score_gate"] = bool(f_score >= 6)

    gpa = safe_float(result.get("gpa"), default=None)
    if gpa is not None:
        result["gross_profitability"] = gpa
        result["_gross_profitability"] = gpa
        result["gpa_score"] = float(np.clip((gpa - 0.30) / 0.20, -1.0, 1.0))

    ebit_yield = safe_float(result.get("ebit_yield"), default=None)
    if ebit_yield is not None:
        result["ev_ebit_score"] = float(np.clip((ebit_yield - 0.10) / 0.10, -1.0, 1.0))
    return result


def _pit_snapshot_to_info(
    ticker: str,
    candidate: dict,
    snapshot: dict | None,
    prior: dict | None = None,
) -> dict:
    """Map PIT/FMP fields to the yfinance-like keys Stage 5b already consumes."""
    if not snapshot:
        return {}

    mcap = _candidate_market_cap(candidate, snapshot)
    total_debt = safe_float(snapshot.get("total_debt"), default=None)
    if total_debt is None:
        total_debt = (
            safe_float(snapshot.get("long_term_debt"), default=0.0)
            + safe_float(snapshot.get("short_term_debt"), default=0.0)
        )
    cash = safe_float(snapshot.get("cash"), default=0.0)
    net_income = safe_float(snapshot.get("net_income"), default=None)
    revenue = safe_float(snapshot.get("revenue"), default=None)
    revenue_prior = safe_float((prior or {}).get("revenue"), default=None)
    ni_prior = safe_float((prior or {}).get("net_income"), default=None)
    total_assets = safe_float(snapshot.get("total_assets"), default=None)
    fcf = _free_cashflow_from_pit(snapshot)
    equity_proxy = (total_assets - total_debt) if (total_assets is not None and total_debt is not None) else None

    info: dict = {
        "symbol": ticker,
        "sector": candidate.get("sector") or candidate.get("_sector") or "",
        "industry": candidate.get("industry") or candidate.get("_industry") or "",
        "marketCap": mcap,
        "totalDebt": total_debt,
        "totalCash": cash,
        "grossProfits": safe_float(snapshot.get("gross_profit"), default=None),
        "grossProfit": safe_float(snapshot.get("gross_profit"), default=None),
        "totalAssets": total_assets,
        "freeCashflow": fcf,
        "operatingCashflow": safe_float(snapshot.get("operating_cashflow"), default=None),
        "netIncomeToCommon": net_income,
        "netIncome": net_income,
        "totalRevenue": revenue,
        "ebit": safe_float(snapshot.get("ebit"), default=None),
        "ebitda": safe_float(snapshot.get("ebitda"), default=None),
        "sharesOutstanding": safe_float(snapshot.get("shares_outstanding"), default=None),
    }
    if mcap and total_debt is not None:
        ev = mcap + max(total_debt, 0.0) - max(cash or 0.0, 0.0)
        if ev > 0:
            info["enterpriseValue"] = ev
    if mcap and net_income and net_income > 0:
        info["trailingPE"] = mcap / net_income
    if equity_proxy and equity_proxy > 0 and net_income is not None:
        info["returnOnEquity"] = net_income / equity_proxy
    if revenue and revenue_prior and revenue_prior > 0:
        info["revenueGrowth"] = (revenue / revenue_prior) - 1.0
    if net_income is not None and ni_prior and abs(ni_prior) > 1e-9:
        info["earningsGrowth"] = (net_income - ni_prior) / abs(ni_prior)
    return {k: v for k, v in info.items() if v is not None}


def _compute_stage2_factor_metrics(ticker: str, candidate: dict, momentum_metrics: dict) -> dict:
    """Cached, PIT-safe factor sleeves used before expensive Stage 6 analysis."""
    latest, prior, report_date = _pit_latest_and_prior(ticker)
    replay_like_features = _current_replay_like_fundamental_features(ticker, candidate)
    coverage = 0.0
    out: dict = {
        "_pit_report_date": report_date,
        "_pit_factor_coverage": 0.0,
        "_pit_quality_score": None,
        "_pit_value_score": None,
        "_pit_low_risk_score": None,
        "_pit_pead_score": None,
    }

    quality_components: list[float] = []
    value_components: list[float] = []
    if latest:
        pit_source = str(latest.get("_source") or "fmp")
        out["_pit_source"] = pit_source
        f = compute_piotroski_f_score(latest, prior)
        f_score = safe_float(f.get("f_score"), default=None)
        f_cov = normalize_f_score_coverage(f.get("f_score_coverage"))
        if is_f_score_actionable(f_score, f_cov, config_module=config):
            quality_components.append(float(np.clip((f_score - 4.5) / 3.0, -1.0, 1.0)))
            coverage += 0.25 * (f_cov or 0.0)
            out["_f_score"] = int(f_score)
            out["_f_score_coverage"] = f_cov

        gp = safe_float(latest.get("gross_profit"), default=None)
        assets = safe_float(latest.get("total_assets"), default=None)
        if gp is not None and assets and assets > 0:
            gpa = gp / assets
            quality_components.append(float(np.clip((gpa - 0.25) / 0.20, -1.0, 1.0)))
            coverage += 0.20
            out["_gpa"] = gpa
            out["_gross_profitability"] = gpa

        fcf = _free_cashflow_from_pit(latest)
        if fcf is not None and assets and assets > 0:
            fcf_to_assets = fcf / assets
            quality_components.append(float(np.clip((fcf_to_assets - 0.03) / 0.08, -1.0, 1.0)))
            coverage += 0.20
            out["_fcf_to_assets"] = fcf_to_assets

        ni = safe_float(latest.get("net_income"), default=None)
        cfo = safe_float(latest.get("operating_cashflow"), default=None)
        if ni is not None and cfo is not None and assets and assets > 0:
            accrual_quality = (cfo - ni) / assets
            quality_components.append(float(np.clip(accrual_quality / 0.08, -1.0, 1.0)))
            coverage += 0.15

        debt = safe_float(latest.get("total_debt"), default=None)
        if debt is None:
            debt = safe_float(latest.get("long_term_debt"), default=0.0) + safe_float(latest.get("short_term_debt"), default=0.0)
        if debt is not None and assets and assets > 0:
            debt_assets = debt / assets
            quality_components.append(float(np.clip((0.45 - debt_assets) / 0.35, -1.0, 1.0)))
            coverage += 0.10
        cash = safe_float(latest.get("cash"), default=0.0)
        ebitda_latest = safe_float(latest.get("ebitda"), default=None)
        if ebitda_latest is not None and ebitda_latest > 0:
            out["_net_debt_ebitda"] = ((debt or 0.0) - (cash or 0.0)) / ebitda_latest
        if debt is not None and debt > 0:
            out["_cash_to_debt"] = (cash or 0.0) / debt

        mcap = _candidate_market_cap(candidate, latest)
        if mcap and mcap > 0:
            if ni is not None and ni > 0:
                ey = ni / mcap
                value_components.append(float(np.clip((ey - 0.04) / 0.08, -1.0, 1.0)))
                out["_pit_earnings_yield"] = ey
            if fcf is not None and fcf > 0:
                fcf_yield = fcf / mcap
                value_components.append(float(np.clip((fcf_yield - 0.04) / 0.08, -1.0, 1.0)))
                out["_fcf_yield"] = fcf_yield
            ebit = safe_float(latest.get("ebit"), default=None)
            debt_for_ev = safe_float(latest.get("total_debt"), default=debt or 0.0)
            ev = mcap + max(debt_for_ev or 0.0, 0.0) - max(cash or 0.0, 0.0)
            if ebit is not None and ebit > 0 and ev > 0:
                ebit_yield = ebit / ev
                value_components.append(float(np.clip((ebit_yield - 0.06) / 0.08, -1.0, 1.0)))
                out["_pit_ebit_yield"] = ebit_yield
                out["_ev_ebit"] = ev / ebit

    if quality_components:
        out["_pit_quality_score"] = float(np.clip(np.mean(quality_components), -1.0, 1.0))
    if value_components:
        out["_pit_value_score"] = float(np.clip(np.mean(value_components), -1.0, 1.0))
    _apply_replay_like_stage2_features(out, replay_like_features)

    beta = safe_float(momentum_metrics.get("_beta"), default=None)
    vol = safe_float(momentum_metrics.get("_vol_20d"), default=None)
    dv = safe_float(momentum_metrics.get("avg_dollar_volume"), default=None)
    rel = safe_float(candidate.get("_relative_strength"), default=None)
    low_risk_components: list[float] = []
    if beta is not None:
        low_risk_components.append(float(np.clip((1.15 - beta) / 1.15, 0.0, 1.0)))
    if vol is not None:
        low_risk_components.append(float(np.clip((0.50 - vol) / 0.45, 0.0, 1.0)))
    if dv is not None and dv > 0:
        low_risk_components.append(float(np.clip(np.log10(max(dv, 1.0) / 1_000_000.0) / 3.0, 0.0, 1.0)))
    if momentum_metrics.get("above_sma50"):
        low_risk_components.append(0.65)
    if rel is not None:
        low_risk_components.append(float(np.clip(0.50 + rel * 5.0, 0.0, 1.0)))
    if low_risk_components:
        out["_pit_low_risk_score"] = float(np.clip(np.mean(low_risk_components), 0.0, 1.0))

    if latest and str(latest.get("_source") or "").lower() == "yfinance_quarterly":
        # yfinance quarterly statements improve breadth, but lack accepted
        # filing timestamps and can be less standardized than FMP/SEC data.
        coverage *= 0.75
        for key in ("_pit_quality_score", "_pit_value_score"):
            if out.get(key) is not None:
                out[key] = float(np.clip(float(out[key]) * 0.85, -1.0, 1.0))

    out["_pit_factor_coverage"] = float(min(1.0, coverage))
    return out


# Accepts both ISO-3166-1-alpha-2 codes (e.g. from the dynamic supplement
# and yfinance .info) and the free-text region labels produced by
# utils.global_universe._TICKER_TO_REGION ("UK", "Germany", "Japan", ...).
# Using both conventions avoids a silent drop to the OTHER floor for the
# majority of ex-US developed-market names whose `_region` is set by the
# static universe path.
_DEVELOPED_REGIONS = {
    # ISO codes
    "GB", "UK", "DE", "FR", "IT", "ES", "NL", "CH", "SE", "DK",
    "NO", "FI", "BE", "IE", "AT", "PT", "JP", "AU", "CA", "NZ",
    "HK", "SG", "IL", "LU",
    # Labels from utils.global_universe._ALL_REGIONS
    "GERMANY", "FRANCE", "SPAIN", "NETHERLANDS", "ITALY", "SWITZERLAND",
    "NORDICS", "CANADA", "AUSTRALIA", "JAPAN", "HONG KONG", "SINGAPORE",
}


def _dollar_volume_floor(candidate: dict, floors: dict) -> float:
    """Region-aware dollar-volume floor for the prescreen gate.

    Prefers an explicit `_region` tag on the candidate, then falls back to
    the country code, then to ticker-suffix heuristics. Returns 0 if no
    floors are configured so callers can treat the gate as disabled.
    """
    if not floors:
        return 0.0

    region_raw = (
        candidate.get("_region")
        or candidate.get("country")
        or ""
    )
    region = str(region_raw).strip().upper()

    if not region:
        sym = (candidate.get("symbol") or "").upper()
        if "." not in sym:
            region = "US"

    if region == "US":
        return float(floors.get("US", 0) or 0)
    if region in _DEVELOPED_REGIONS:
        return float(floors.get("DEVELOPED", 0) or 0)
    return float(floors.get("OTHER", 0) or 0)


def _effective_group_floor(target_n: int, configured_floor: int, groups: set[str]) -> int:
    """Cap group floors so coverage targets cannot overwhelm the shortlist."""
    valid_groups = {g for g in groups if g and g != "Unknown" and g != "unknown"}
    if target_n <= 0 or configured_floor <= 0 or not valid_groups:
        return 0
    return max(1, min(configured_floor, target_n // len(valid_groups)))


def _scaled_lens_min_counts(lens_min_counts: dict[str, int] | None, target_n: int) -> dict[str, int]:
    """Scale full-run lens quotas down for bounded discovery runs.

    The production config expresses minimum counts for the normal 250-name
    Stage 5b panel. A smaller verification run (for example top 20 or 60)
    should preserve the same lens mix instead of filling every slot with the
    first configured lens.
    """
    if not lens_min_counts or target_n <= 0:
        return {}
    raw = {
        str(k): max(0, int(v or 0))
        for k, v in lens_min_counts.items()
        if max(0, int(v or 0)) > 0
    }
    raw_total = sum(raw.values())
    if raw_total <= 0:
        return {}
    if raw_total <= target_n:
        return raw

    scaled_float = {lens: count * target_n / raw_total for lens, count in raw.items()}
    scaled = {lens: int(np.floor(value)) for lens, value in scaled_float.items()}
    remainder = target_n - sum(scaled.values())
    for lens, _frac in sorted(
        ((lens, value - scaled[lens]) for lens, value in scaled_float.items()),
        key=lambda item: item[1],
        reverse=True,
    ):
        if remainder <= 0:
            break
        scaled[lens] += 1
        remainder -= 1
    return {lens: count for lens, count in scaled.items() if count > 0}


def _adaptive_promote_candidates(
    candidates: list[dict],
    *,
    target_n: int,
    score_key: str,
    preselect_pct: float,
    region_floor: int = 0,
    sector_floor: int = 0,
    liquidity_floor: int = 0,
    lens_floor: int = 0,
    lens_min_counts: dict[str, int] | None = None,
    sector_max: int | None = None,
) -> list[dict]:
    """Promote candidates with a global core plus cohort-aware backfills.

    The lens_floor guarantees minimum representation per entry lens
    (momentum, value, quality, composite) so that non-momentum pathways
    aren't crushed by the global score ranking.
    """
    if not candidates or target_n <= 0:
        return []

    ordered = sorted(
        candidates,
        key=lambda cand: safe_float(cand.get(score_key), default=0.0),
        reverse=True,
    )
    if len(ordered) <= target_n:
        return ordered

    lens_min_counts = _scaled_lens_min_counts(lens_min_counts, target_n)

    selected: list[dict] = []
    selected_symbols: set[str] = set()
    selected_sector_counts: dict[str, int] = {}

    def _add(candidate: dict) -> bool:
        symbol = _candidate_symbol(candidate)
        if not symbol or symbol in selected_symbols or len(selected) >= target_n:
            return False
        sector = _candidate_sector_label(candidate)
        if sector_max is not None and sector_max > 0:
            if selected_sector_counts.get(sector, 0) >= sector_max:
                return False
        selected.append(candidate)
        selected_symbols.add(symbol)
        selected_sector_counts[sector] = selected_sector_counts.get(sector, 0) + 1
        return True

    preselect_n = min(
        target_n,
        max(1, int(round(target_n * max(0.0, min(preselect_pct, 1.0))))),
    )
    if lens_min_counts:
        reserved_min_total = sum(max(0, int(v or 0)) for v in lens_min_counts.values())
        preselect_n = min(preselect_n, max(0, target_n - min(target_n, reserved_min_total)))
    for candidate in ordered[:preselect_n]:
        _add(candidate)

    def _top_up(group_fn, configured_floor: int) -> None:
        active_groups = {
            group_fn(candidate)
            for candidate in ordered
            if group_fn(candidate)
        }
        floor = _effective_group_floor(target_n, configured_floor, active_groups)
        if floor <= 0 or len(selected) >= target_n:
            return

        per_group_counts: dict[str, int] = {}
        for candidate in selected:
            group = group_fn(candidate)
            per_group_counts[group] = per_group_counts.get(group, 0) + 1

        for candidate in ordered:
            if len(selected) >= target_n:
                break
            group = group_fn(candidate)
            if not group:
                continue
            if per_group_counts.get(group, 0) >= floor:
                continue
            if _add(candidate):
                per_group_counts[group] = per_group_counts.get(group, 0) + 1

    # Lens floor first — ensures non-momentum pathways survive before
    # region/sector/liquidity backfill consumes remaining slots.
    if lens_min_counts:
        per_lens_counts: dict[str, int] = {}
        for candidate in selected:
            lens = str(candidate.get("_entry_lens", "") or "")
            per_lens_counts[lens] = per_lens_counts.get(lens, 0) + 1

        for lens, min_count in lens_min_counts.items():
            lens = str(lens)
            target_count = max(0, int(min_count or 0))
            if target_count <= 0:
                continue
            for candidate in ordered:
                if len(selected) >= target_n or per_lens_counts.get(lens, 0) >= target_count:
                    break
                if str(candidate.get("_entry_lens", "") or "") != lens:
                    continue
                if _add(candidate):
                    per_lens_counts[lens] = per_lens_counts.get(lens, 0) + 1

    if lens_floor > 0:
        _top_up(lambda c: c.get("_entry_lens", ""), lens_floor)

    _top_up(_candidate_region, region_floor)
    _top_up(_candidate_sector_label, sector_floor)
    _top_up(_candidate_liquidity_bucket, liquidity_floor)

    for candidate in ordered:
        if len(selected) >= target_n:
            break
        _add(candidate)

    return selected


def _promote_ready_reserve(
    pool: list[dict],
    selected: list[dict],
    *,
    target_n: int,
    reserve_n: int,
    min_ready_score: float,
    score_key: str,
) -> list[dict]:
    """Ensure some Stage 6 slots go to candidates with actionable entry setup.

    The reserve only replaces weak, less-ready names with higher-ready names;
    it does not expand the shortlist or bypass the main score ordering.
    """
    if reserve_n <= 0 or not pool or not selected:
        return selected

    selected_symbols = {_candidate_symbol(c) for c in selected}

    def _ready(c: dict) -> float:
        return safe_float(c.get("_cheap_ready_score"), default=0.0)

    target_ready = min(max(0, reserve_n), target_n)
    current_ready = sum(1 for c in selected if _ready(c) >= min_ready_score)
    if current_ready >= target_ready:
        return selected

    ready_pool = sorted(
        [
            c for c in pool
            if _candidate_symbol(c) not in selected_symbols
            and _ready(c) >= min_ready_score
        ],
        key=lambda c: (_ready(c), safe_float(c.get(score_key), default=0.0)),
        reverse=True,
    )
    if not ready_pool:
        return selected

    protected_symbols: set[str] = set()
    lens_min_counts = _scaled_lens_min_counts(
        getattr(config, "DISCOVERY_FULL_SCORE_LENS_MIN_COUNTS", {}) or {},
        target_n,
    )
    lens_counts: dict[str, int] = {}
    for candidate in selected:
        lens = str(candidate.get("_entry_lens", "") or "")
        lens_counts[lens] = lens_counts.get(lens, 0) + 1

    for candidate in ready_pool:
        if current_ready >= target_ready:
            break
        replacement_idx = None
        replacement_score = float("inf")
        for idx, existing in enumerate(selected):
            symbol = _candidate_symbol(existing)
            if symbol in protected_symbols:
                continue
            if _ready(existing) >= min_ready_score:
                continue
            existing_lens = str(existing.get("_entry_lens", "") or "")
            min_for_lens = int(lens_min_counts.get(existing_lens, 0) or 0)
            if min_for_lens > 0 and lens_counts.get(existing_lens, 0) <= min_for_lens:
                continue
            score = safe_float(existing.get(score_key), default=-999.0)
            if replacement_idx is None or score < replacement_score:
                replacement_idx = idx
                replacement_score = score
        if replacement_idx is None:
            break
        old_lens = str(selected[replacement_idx].get("_entry_lens", "") or "")
        new_lens = str(candidate.get("_entry_lens", "") or "")
        selected[replacement_idx] = candidate
        lens_counts[old_lens] = max(0, lens_counts.get(old_lens, 0) - 1)
        lens_counts[new_lens] = lens_counts.get(new_lens, 0) + 1
        symbol = _candidate_symbol(candidate)
        selected_symbols.add(symbol)
        protected_symbols.add(symbol)
        current_ready += 1

    selected = selected[:target_n]
    selected.sort(key=lambda c: safe_float(c.get(score_key), default=0.0), reverse=True)
    return selected


def _cheap_quality_score(candidate: dict) -> float:
    """Cheap-quality reserve score for MLI-style mid-cap candidates.

    The score intentionally uses only Stage-5 cached/PIT-safe fields and does
    not change the final action label.  It exists to make sure cheap, clean,
    liquid quality names reach the expensive scoring stage.
    """
    if _candidate_exclusion_reason(candidate):
        return 0.0

    score = 0.0
    ev_ebit = safe_float(candidate.get("_ev_ebit"), default=None)
    if ev_ebit is not None and ev_ebit > 0:
        if ev_ebit <= 18:
            score += 0.24
        elif ev_ebit <= 25:
            score += 0.16
        elif ev_ebit <= 32:
            score += 0.06

    f_score = safe_float(candidate.get("_f_score"), default=None)
    f_cov = normalize_f_score_coverage(candidate.get("_f_score_coverage"))
    if is_f_score_actionable(f_score, f_cov, config_module=config):
        if f_score >= 7:
            score += 0.20
        elif f_score >= 6:
            score += 0.16
        elif f_score >= 5:
            score += 0.08

    gpa = safe_float(candidate.get("_gpa"), default=None)
    pit_quality = safe_float(candidate.get("_pit_quality_score"), default=None)
    if gpa is not None:
        if gpa >= 0.20:
            score += 0.18
        elif gpa >= 0.15:
            score += 0.14
        elif gpa >= 0.10:
            score += 0.07
    if pit_quality is not None and pit_quality > 0:
        score += min(0.12, 0.12 * pit_quality)

    net_debt = safe_float(candidate.get("_net_debt_ebitda"), default=None)
    cash_to_debt = safe_float(candidate.get("_cash_to_debt"), default=None)
    if net_debt is not None:
        if net_debt <= 0:
            score += 0.12
        elif net_debt <= 1.0:
            score += 0.08
    elif cash_to_debt is not None and cash_to_debt >= 1.0:
        score += 0.08

    ret30 = safe_float(candidate.get("_ret_30d"), default=None)
    ret90 = safe_float(candidate.get("_ret_90d"), default=None)
    if ret30 is not None:
        if -0.05 <= ret30 <= 0.18:
            score += 0.08
        elif 0.18 < ret30 <= 0.28:
            score += 0.02
        elif ret30 > 0.35:
            score -= 0.12
    if ret90 is not None and -0.10 <= ret90 <= 0.35:
        score += 0.04
    if candidate.get("_above_sma50") and candidate.get("_above_sma200"):
        score += 0.08
    if safe_float(candidate.get("_avg_dollar_volume"), default=0.0) >= 5_000_000:
        score += 0.04
    return float(np.clip(score, 0.0, 1.0))


def _promote_cheap_quality_reserve(
    pool: list[dict],
    selected: list[dict],
    *,
    target_n: int,
    reserve_n: int,
    min_score: float,
    score_key: str,
) -> list[dict]:
    """Reserve Stage-6 seats for cheap-quality names before final gates decide."""
    if reserve_n <= 0 or not pool or not selected:
        return selected
    selected = list(selected or [])[:target_n]
    selected_symbols = {_candidate_symbol(c) for c in selected}
    target_count = min(max(0, reserve_n), target_n)
    current_count = sum(
        1 for c in selected
        if safe_float(c.get("_cheap_quality_score"), default=0.0) >= min_score
    )
    if current_count >= target_count:
        return selected

    pool_ranked = sorted(
        [
            c for c in pool
            if _candidate_symbol(c)
            and _candidate_symbol(c) not in selected_symbols
            and safe_float(c.get("_cheap_quality_score"), default=0.0) >= min_score
        ],
        key=lambda c: (
            safe_float(c.get("_cheap_quality_score"), default=0.0),
            safe_float(c.get(score_key), default=0.0),
        ),
        reverse=True,
    )
    if not pool_ranked:
        return selected

    protected_symbols: set[str] = set()
    promoted = 0
    for candidate in pool_ranked:
        if current_count >= target_count:
            break
        replacement_idx = None
        replacement_key: tuple[float, float] | None = None
        for idx, existing in enumerate(selected):
            symbol = _candidate_symbol(existing)
            if symbol in protected_symbols:
                continue
            existing_cq = safe_float(existing.get("_cheap_quality_score"), default=0.0)
            if existing_cq >= min_score:
                continue
            key = (existing_cq, safe_float(existing.get(score_key), default=-999.0))
            if replacement_idx is None or key < replacement_key:
                replacement_idx = idx
                replacement_key = key
        if replacement_idx is None:
            break
        selected[replacement_idx] = candidate
        symbol = _candidate_symbol(candidate)
        protected_symbols.add(symbol)
        selected_symbols.add(symbol)
        candidate["_cheap_quality_reserved"] = True
        current_count += 1
        promoted += 1

    if promoted:
        logger.info("Cheap-quality reserve promoted %d candidates into %s shortlist", promoted, score_key)
    selected.sort(key=lambda c: safe_float(c.get(score_key), default=0.0), reverse=True)
    return selected[:target_n]


def _promote_ready_lane(
    pool: list[dict],
    selected: list[dict],
    *,
    target_n: int,
    reserve_n: int,
    min_ready_score: float,
    score_key: str,
) -> list[dict]:
    """Reserve Stage 6 seats for names likely to be buyable today.

    The lane does not bypass final scoring or action gates. It only ensures
    that deep analysis sees enough candidates with clean cheap setup and
    cached/PIT evidence instead of spending every slot on stretched watchlist
    ideas.
    """
    if reserve_n <= 0 or not pool or not selected:
        return selected

    target_ready = min(max(0, reserve_n), target_n)
    selected = list(selected or [])[:target_n]
    selected_symbols = {_candidate_symbol(c) for c in selected}

    def _ready(c: dict) -> float:
        return safe_float(c.get("_ready_lane_score"), default=0.0)

    current_ready = sum(1 for c in selected if _ready(c) >= min_ready_score)
    if current_ready >= target_ready:
        return selected

    ready_pool = sorted(
        [
            c for c in pool
            if _candidate_symbol(c)
            and _candidate_symbol(c) not in selected_symbols
            and _ready(c) >= min_ready_score
        ],
        key=lambda c: (_ready(c), safe_float(c.get(score_key), default=0.0)),
        reverse=True,
    )
    if not ready_pool:
        return selected

    lens_min_counts = _scaled_lens_min_counts(
        getattr(config, "DISCOVERY_FULL_SCORE_LENS_MIN_COUNTS", {}) or {},
        target_n,
    )
    lens_counts: dict[str, int] = {}
    for candidate in selected:
        lens = str(candidate.get("_entry_lens", "") or "")
        lens_counts[lens] = lens_counts.get(lens, 0) + 1

    max_replace = max(
        1,
        int(round(target_n * float(getattr(config, "DISCOVERY_FULL_SCORE_READY_LANE_MAX_REPLACE_PCT", 0.45)))),
    )
    replaced = 0
    protected_symbols: set[str] = set()
    for candidate in ready_pool:
        if current_ready >= target_ready or replaced >= max_replace:
            break
        replacement_idx = None
        replacement_key: tuple[float, float] | None = None
        for idx, existing in enumerate(selected):
            symbol = _candidate_symbol(existing)
            if symbol in protected_symbols:
                continue
            if _ready(existing) >= min_ready_score:
                continue
            existing_lens = str(existing.get("_entry_lens", "") or "")
            min_for_lens = int(lens_min_counts.get(existing_lens, 0) or 0)
            if min_for_lens > 0 and lens_counts.get(existing_lens, 0) <= min_for_lens:
                continue
            key = (
                _ready(existing),
                safe_float(existing.get(score_key), default=-999.0),
            )
            if replacement_key is None or key < replacement_key:
                replacement_idx = idx
                replacement_key = key
        if replacement_idx is None:
            break
        old_lens = str(selected[replacement_idx].get("_entry_lens", "") or "")
        new_lens = str(candidate.get("_entry_lens", "") or "")
        selected[replacement_idx] = candidate
        lens_counts[old_lens] = max(0, lens_counts.get(old_lens, 0) - 1)
        lens_counts[new_lens] = lens_counts.get(new_lens, 0) + 1
        selected_symbols.add(_candidate_symbol(candidate))
        protected_symbols.add(_candidate_symbol(candidate))
        current_ready += 1
        replaced += 1

    selected.sort(key=lambda c: safe_float(c.get(score_key), default=0.0), reverse=True)
    return selected[:target_n]


def _promote_challenge_reserve(
    pool: list[dict],
    selected: list[dict],
    *,
    target_n: int,
    reserve_n: int,
    score_key: str,
) -> list[dict]:
    """Reserve deep-scoring slots for audited challenge names.

    This is a funnel acceleration lane, not an action override: promoted names
    still need to pass full scoring, action gates, and the ready contract.
    """
    if (
        reserve_n <= 0
        or not bool(getattr(config, "DISCOVERY_CHALLENGE_RESERVE_ENABLED", False))
        or not pool
    ):
        return selected

    challenge_order = _challenge_ticker_order()
    if not challenge_order:
        return selected

    selected = list(selected or [])
    selected_symbols = {_candidate_symbol(c) for c in selected}
    current_challenge = sum(1 for c in selected if _candidate_symbol(c) in challenge_order)
    target_challenge = min(max(0, reserve_n), len(challenge_order), target_n)
    if current_challenge >= target_challenge:
        return selected[:target_n]

    challenge_pool = sorted(
        [
            c for c in pool
            if _candidate_symbol(c) in challenge_order
            and _candidate_symbol(c) not in selected_symbols
            and _candidate_exclusion_reason(c) is None
        ],
        key=lambda c: (
            challenge_order.get(_candidate_symbol(c), 9999),
            -safe_float(c.get(score_key), default=0.0),
        ),
    )
    if not challenge_pool:
        return selected[:target_n]

    protected_symbols = {
        _candidate_symbol(c)
        for c in selected
        if _candidate_symbol(c) in challenge_order
    }
    promoted = 0
    for candidate in challenge_pool:
        if current_challenge >= target_challenge:
            break
        symbol = _candidate_symbol(candidate)
        if len(selected) < target_n:
            selected.append(candidate)
        else:
            replacement_idx = None
            replacement_score = float("inf")
            for idx, existing in enumerate(selected):
                existing_symbol = _candidate_symbol(existing)
                if existing_symbol in protected_symbols:
                    continue
                score = safe_float(existing.get(score_key), default=-999.0)
                if replacement_idx is None or score < replacement_score:
                    replacement_idx = idx
                    replacement_score = score
            if replacement_idx is None:
                break
            selected[replacement_idx] = candidate
        candidate["_challenge_reserved"] = True
        selected_symbols.add(symbol)
        protected_symbols.add(symbol)
        current_challenge += 1
        promoted += 1

    if promoted:
        logger.info("Challenge reserve promoted %d candidates into %s shortlist", promoted, score_key)
    selected = selected[:target_n]
    selected.sort(key=lambda c: safe_float(c.get(score_key), default=0.0), reverse=True)
    return selected


def _build_partial_score_result(item: dict, failure_reason: str) -> dict:
    """Construct a degraded-but-usable result when deep scoring fails."""
    candidate = item["candidate"]
    ticker = item["holding"]["ticker"]
    price = safe_float(candidate.get("price") or candidate.get("_last_price"), default=0.0)
    momentum_score = safe_float(candidate.get("_momentum_score"), default=0.5)
    technical_score = float(np.clip((momentum_score - 0.5) * 2.0, -1.0, 1.0))
    quality_score = safe_float(
        candidate.get("_quality_score_fundamental", candidate.get("_quality_score")),
        default=0.0,
    )
    forecast_score = float(np.clip((safe_float(candidate.get("_quick_score"), default=0.5) - 0.5) * 1.5, -1.0, 1.0))
    aggregate_score = (
        technical_score * config.WEIGHTS.get("technical", 0.30)
        + quality_score * config.WEIGHTS.get("fundamental", 0.40)
        + forecast_score * config.WEIGHTS.get("forecast", 0.22)
    )

    return {
        "ticker": ticker,
        "name": candidate.get("companyName", ticker),
        "current_price": price,
        "technical_score": round(technical_score, 4),
        "fundamental_score": round(quality_score, 4),
        "sentiment_score": 0.0,
        "forecast_score": round(forecast_score, 4),
        "aggregate_score": round(aggregate_score, 4),
        "why": f"Partial discovery analysis used: {failure_reason}",
        "analysis_degraded": True,
        "analysis_degraded_reason": failure_reason,
        "quality_score_fundamental": candidate.get("_quality_score_fundamental", quality_score),
        "gross_profitability": candidate.get("_gross_profitability"),
        "fcf_to_assets": candidate.get("_fcf_to_assets"),
        "earnings_stability": candidate.get("_earnings_stability"),
        "eps_growth_variance_5y": candidate.get("_eps_growth_variance_5y"),
        "expected_return_90d": 0.0,
    }


def _quality_overlay_score(result: dict) -> float:
    """Quality overlay from the live fundamental quality factor."""
    quality_score = result.get("quality_score_fundamental")
    if quality_score is None:
        quality_score = result.get("_quality_score_fundamental")
    if quality_score is None:
        return 0.0
    return max(-0.15, min(0.15, quality_score * 0.15))


def _compute_qmj_quality(info: dict) -> tuple[float, dict]:
    """Compatibility wrapper around the shared QMJ-style quality module."""
    metrics = compute_fundamental_quality_metrics(info)
    details = {
        "gross_profitability": metrics.get("gross_profitability"),
        "roe": metrics.get("roe"),
        "fcf_to_assets": metrics.get("fcf_to_assets"),
        "earnings_stability": metrics.get("earnings_stability"),
        "eps_growth_variance_5y": metrics.get("eps_growth_variance_5y"),
    }
    return round(metrics.get("quality_score", 0.0), 4), details


def _pead_overlay_score(result: dict, overlay) -> float:
    """Post-earnings drift / revision overlay using already-fetched fields."""
    revision_pct = _parse_revision_pct(result.get("estimate_revision"))
    beat_rate = _parse_beat_rate(result.get("earnings_beat_rate"))

    if not getattr(overlay, "post_earnings_recent", False) and revision_pct is None:
        return 0.0

    surprise_component = 0.0
    if getattr(overlay, "post_earnings_recent", False):
        if getattr(overlay, "earnings_miss", False):
            miss_pct = safe_float(getattr(overlay, "earnings_miss_pct", None), default=0.0)
            surprise_component = max(-1.0, min(0.0, miss_pct / 15.0))
        elif beat_rate is not None:
            surprise_component = max(-1.0, min(1.0, (beat_rate - 0.5) / 0.5))

    revision_component = 0.0
    if revision_pct is not None:
        revision_component = max(-1.0, min(1.0, revision_pct / 10.0))

    beat_component = 0.0
    if beat_rate is not None:
        beat_component = max(-1.0, min(1.0, (beat_rate - 0.5) / 0.25))

    raw = 0.5 * surprise_component + 0.3 * revision_component + 0.2 * beat_component
    scale = getattr(config, "PEAD_MAX_OVERLAY", 0.10)
    return max(-scale, min(scale, raw * scale))


_CYCLICAL_COMMODITY_SECTORS = {"basic materials", "materials", "energy"}
_CYCLICAL_COMMODITY_KEYWORDS = (
    "mining", "miner", "metals", "metal", "lithium", "spodumene",
    "copper", "gold", "silver", "coal", "steel", "aluminum", "uranium",
    "oil", "gas", "exploration", "production", "refining", "commodity",
)


def _optional_float(value) -> float | None:
    """Return a finite float or None without converting missing data to zero."""
    return safe_float(value, default=None)


def _optional_int(value) -> int | None:
    v = _optional_float(value)
    return int(v) if v is not None else None


def _price_vs_sma200_stretch(result: dict) -> float | None:
    """Price / SMA200 - 1, using signal-time fields when available."""
    existing = _optional_float(result.get("price_vs_sma200_stretch"))
    if existing is not None:
        return existing

    price = _optional_float(
        result.get("current_price")
        or result.get("price")
        or result.get("_last_price")
    )
    sma_200 = _optional_float(result.get("sma_200"))
    if price is None or sma_200 is None or sma_200 <= 0:
        return None
    return price / sma_200 - 1.0


def _is_cyclical_commodity_candidate(sector: str, industry: str, name: str = "") -> bool:
    """Identify commodity producers without relying on coarse sector tags alone."""
    sector_norm = _normalize_sector(sector or "")
    if sector_norm not in _CYCLICAL_COMMODITY_SECTORS:
        return False

    if not str(industry or "").strip():
        return True
    text = f"{industry or ''} {name or ''}".lower()
    return any(keyword in text for keyword in _CYCLICAL_COMMODITY_KEYWORDS)


def _evaluate_discovery_gates_v2(
    result: dict,
    *,
    sector: str,
    industry: str,
    stretch: float | None,
) -> tuple[str, list[str]]:
    """Shadow gate assessment for the future gate-then-rank architecture.

    Statuses:
      PASS   - no configured gate would object
      REVIEW - missing data or mild threshold breach
      REJECT - active data says the name would fail broad V2 gates
    """
    reasons: list[str] = []
    reject = False
    review = False

    min_cov = f_score_min_coverage(config)
    min_f = int(getattr(config, "DISCOVERY_GATES_V2_F_SCORE_MIN", 5))
    min_gpa = float(getattr(config, "DISCOVERY_GATES_V2_GPA_MIN", 0.15))
    max_stretch = float(getattr(config, "DISCOVERY_GATES_V2_MAX_STRETCH", 0.50))

    f_score = _optional_int(result.get("f_score"))
    f_cov = normalize_f_score_coverage(result.get("f_score_coverage"))
    gpa = _optional_float(result.get("gpa"))
    ev_ebit = _optional_float(result.get("ev_ebit"))
    fcf_yield = _optional_float(result.get("fcf_yield"))
    revenue_growth = _optional_float(result.get("revenue_growth"))

    if f_score is None or (f_cov or 0.0) < min_cov:
        review = True
        reasons.append("F-score missing/low coverage")
    elif f_score < min_f:
        reject = True
        reasons.append(f"F-score {f_score} < {min_f}")

    if gpa is None:
        review = True
        reasons.append("GPA missing")
    elif gpa < min_gpa:
        reject = True
        reasons.append(f"GPA {gpa:.2f} < {min_gpa:.2f}")

    value_ok = (
        (ev_ebit is not None and ev_ebit > 0 and ev_ebit <= 15)
        or (fcf_yield is not None and fcf_yield >= 0.03)
        or (revenue_growth is not None and revenue_growth >= 0.20)
    )
    if not value_ok:
        review = True
        reasons.append("No clear value/FCF/growth support")

    if stretch is None:
        review = True
        reasons.append("SMA200 stretch missing")
    elif stretch > max_stretch:
        review = True
        reasons.append(f"Price {stretch:.0%} above SMA200")

    if _is_cyclical_commodity_candidate(sector, industry, result.get("name", "")):
        reasons.append("Cyclical commodity candidate")

    if reject:
        return "REJECT", reasons
    if review:
        return "REVIEW", reasons
    return "PASS", reasons


def _evaluate_trap_safeguard(
    result: dict,
    *,
    sector: str,
    industry: str,
    stretch: float | None,
) -> tuple[bool, str | None]:
    """Active narrow guard for commodity-cycle momentum traps.

    This is deliberately narrower than the broad V2 gates: it needs a cyclical
    commodity business, a stretched tape, and weak or missing health data.
    """
    if not getattr(config, "DISCOVERY_TRAP_SAFEGUARD_ENABLED", True):
        return False, None
    if not _is_cyclical_commodity_candidate(sector, industry, result.get("name", "")):
        return False, None

    min_stretch = float(getattr(config, "DISCOVERY_TRAP_SAFEGUARD_MIN_STRETCH", 0.50))
    if stretch is None or stretch < min_stretch:
        return False, None

    f_score = _optional_int(result.get("f_score"))
    f_cov = normalize_f_score_coverage(result.get("f_score_coverage"))
    max_f = int(getattr(config, "DISCOVERY_TRAP_SAFEGUARD_F_SCORE_MAX", 3))
    min_cov = f_score_min_coverage(config)
    gpa = _optional_float(result.get("gpa"))
    max_gpa = float(getattr(config, "DISCOVERY_TRAP_SAFEGUARD_GPA_MAX", 0.15))

    fscore_weak_or_missing = f_score is None or (f_cov or 0.0) < min_cov or f_score <= max_f
    gpa_weak_or_missing = gpa is None or gpa < max_gpa
    if not (fscore_weak_or_missing and gpa_weak_or_missing):
        return False, None

    health_bits = []
    if f_score is None or (f_cov or 0.0) < min_cov:
        health_bits.append("F-score unavailable")
    else:
        health_bits.append(f"F-score {f_score}")
    if gpa is None:
        health_bits.append("GPA unavailable")
    else:
        health_bits.append(f"GPA {gpa:.2f}")

    reason = (
        "Commodity-cycle trap safeguard: "
        + ", ".join(health_bits)
        + f", price {stretch:.0%} above SMA200"
    )
    return True, reason


def _gate_review_is_missing_data_only(gate_status: str, gate_reasons: list[str] | None) -> bool:
    """Treat a REVIEW caused only by missing fields as non-fatal for readiness."""
    if str(gate_status or "").upper() != "REVIEW":
        return False
    if not bool(getattr(config, "READY_STRONG_BUY_ALLOW_MISSING_DATA_REVIEW", False)):
        return False
    reasons = [
        str(reason or "").lower()
        for reason in (gate_reasons or [])
        if str(reason or "").strip()
    ]
    if not reasons:
        return False
    missing_markers = (
        "missing",
        "unavailable",
        "not available",
        "insufficient",
        "coverage",
        "no pit",
        "not found",
        "nan",
    )
    return all(any(marker in reason for marker in missing_markers) for reason in reasons)


def apply_action_label(
    result: dict,
    *,
    pillars_all_zero: bool,
    adjusted_aggregate: float,
    sector: str = "",
    industry: str = "",
) -> tuple[str, str, bool]:
    """Pure action-label assignment matching the live discovery gate stack.

    Centralises the threshold + F-score gate + V2-gate + trap-safeguard rules
    that determine STRONG BUY / BUY / NEUTRAL / AVOID / MANUAL REVIEW. Used by
    both the discovery scoring path (advisory) and by the action-label backfill
    path so historical rows in `signal_backtest` can be re-labelled with the
    same logic that fires live.

    Returns:
        action: STRONG BUY | BUY | NEUTRAL | AVOID | INSUFFICIENT DATA | MANUAL REVIEW
        gate_v2_status: PASS | REVIEW | REJECT (advisory)
        trap_triggered: bool

    Note: this helper is side-effect free (no logging, no rank multiplication).
    Callers in the live path apply rank discounts and log gate hits separately
    so this function can be invoked from backfill scripts without polluting logs.
    """
    if pillars_all_zero:
        return ("INSUFFICIENT DATA", "REVIEW", False)

    if adjusted_aggregate >= config.SCORE_STRONG_BUY_THRESHOLD:
        action = "STRONG BUY"
    elif adjusted_aggregate >= config.SCORE_BUY_THRESHOLD:
        action = "BUY"
    elif adjusted_aggregate >= config.SCORE_KEEP_THRESHOLD:
        action = "NEUTRAL"
    else:
        action = "AVOID"

    # Piotroski F-score downgrade (gated by config flag).
    if getattr(config, "F_SCORE_GATE_ENABLED", False):
        fs = _optional_float(result.get("f_score"))
        fs_cov = normalize_f_score_coverage(result.get("f_score_coverage"))
        if (
            fs is not None
            and is_f_score_actionable(fs, fs_cov, config_module=config)
            and fs <= 3
            and action in ("STRONG BUY", "BUY")
        ):
            action = "NEUTRAL"

    stretch = _price_vs_sma200_stretch(result)
    gate_v2_status, _ = _evaluate_discovery_gates_v2(
        result, sector=sector, industry=industry, stretch=stretch
    )
    trap_triggered, _ = _evaluate_trap_safeguard(
        result, sector=sector, industry=industry, stretch=stretch
    )

    v2_shadow = bool(getattr(config, "DISCOVERY_GATES_V2_SHADOW", True))
    gate_first_active = (
        bool(getattr(config, "DISCOVERY_GATE_FIRST_ENABLED", True))
        and not v2_shadow
    )
    v2_active = (
        bool(getattr(config, "DISCOVERY_GATES_V2_ENABLED", False))
        and not v2_shadow
    ) or gate_first_active

    if v2_active and gate_v2_status == "REJECT":
        if action in ("STRONG BUY", "BUY", "NEUTRAL"):
            action = "MANUAL REVIEW"

    if trap_triggered:
        if action in ("STRONG BUY", "BUY", "NEUTRAL"):
            action = "MANUAL REVIEW"

    return (action, gate_v2_status, trap_triggered)


def _prior_passes_ready_contract(prior) -> tuple[bool, list[str], list[str]]:
    """Evaluate the institutional prior with a calibrated near-miss allowance."""
    if getattr(prior, "passes_strong_buy_bar", False):
        return True, [], []

    reasons = list(getattr(prior, "reasons", []) or ["institutional prior below bar"])
    if not bool(getattr(config, "READY_STRONG_BUY_PRIOR_SOFT_PASS_ENABLED", True)):
        return False, reasons, []

    percentile = safe_float(getattr(prior, "percentile", None), default=0.0)
    confidence = safe_float(getattr(prior, "confidence", None), default=0.0)
    coverage = safe_float(getattr(prior, "coverage", None), default=0.0)
    score = safe_float(getattr(prior, "score", None), default=-1.0)

    soft_pct = float(getattr(config, "READY_STRONG_BUY_PRIOR_SOFT_PERCENTILE", 0.80))
    soft_conf = float(getattr(config, "READY_STRONG_BUY_PRIOR_SOFT_CONFIDENCE", 0.55))
    soft_cov = float(getattr(config, "READY_STRONG_BUY_PRIOR_SOFT_COVERAGE", 0.40))
    soft_score = float(getattr(config, "READY_STRONG_BUY_PRIOR_SOFT_MIN_SCORE", 0.0))
    if (
        percentile >= soft_pct
        and confidence >= soft_conf
        and coverage >= soft_cov
        and score >= soft_score
    ):
        return True, [], [
            (
                "institutional prior soft-pass "
                f"(pct {percentile:.0%}, conf {confidence:.0%}, cov {coverage:.0%})"
            )
        ]
    return False, reasons, []


def _ready_contract_score(
    *,
    gate_status: str,
    trap_triggered: bool,
    prior,
    prior_pass: bool,
    entry_stance: str,
    entry_price,
    stop_loss,
    take_profit,
    rr_ratio,
    position_weight,
    data_confidence: float,
    gate_reasons: list[str] | None = None,
) -> float:
    """Continuous readiness score used for calibration and near-ready reports."""
    if trap_triggered:
        return 0.0

    gate_status_norm = str(gate_status or "PASS").upper()
    gate_ok = (
        gate_status_norm == "PASS"
        or _gate_review_is_missing_data_only(gate_status_norm, gate_reasons)
    )
    if gate_status_norm == "REJECT":
        gate_ok = False

    prior_pct = safe_float(getattr(prior, "percentile", None), default=0.0)
    prior_conf = safe_float(getattr(prior, "confidence", None), default=0.0)
    prior_cov = safe_float(getattr(prior, "coverage", None), default=0.0)
    prior_component = 1.0 if prior_pass else np.mean([
        min(1.0, prior_pct / max(float(getattr(config, "INSTITUTIONAL_PRIOR_STRONG_BUY_PERCENTILE", 0.90)), 1e-9)),
        min(1.0, prior_conf / max(float(getattr(config, "INSTITUTIONAL_PRIOR_MIN_CONFIDENCE", 0.55)), 1e-9)),
        min(1.0, prior_cov / max(float(getattr(config, "INSTITUTIONAL_PRIOR_MIN_COVERAGE", 0.45)), 1e-9)),
    ])

    entry_component = {"Ready": 1.0, "Pullback Preferred": 0.55, "Watch Only": 0.0}.get(str(entry_stance), 0.0)
    levels_component = float(
        safe_float(entry_price, default=0.0) > 0
        and safe_float(stop_loss, default=0.0) > 0
        and safe_float(take_profit, default=0.0) > 0
    )
    min_rr = float(getattr(config, "READY_STRONG_BUY_MIN_RR", 1.50))
    rr_component = min(1.0, max(0.0, safe_float(rr_ratio, default=0.0) / max(min_rr, 1e-9)))
    min_conf = float(getattr(config, "READY_STRONG_BUY_MIN_CONFIDENCE", 0.70))
    conf_component = min(1.0, max(0.0, safe_float(data_confidence, default=0.0) / max(min_conf, 1e-9)))
    min_weight = float(getattr(config, "READY_STRONG_BUY_MIN_POSITION_WEIGHT", 0.005))
    weight_component = min(1.0, max(0.0, safe_float(position_weight, default=0.0) / max(min_weight, 1e-9)))

    score = (
        0.20 * float(prior_component)
        + 0.20 * entry_component
        + 0.10 * levels_component
        + 0.20 * rr_component
        + 0.15 * conf_component
        + 0.10 * weight_component
        + 0.05 * (1.0 if gate_ok else 0.0)
    )
    return float(np.clip(score, 0.0, 1.0))


def _evaluate_ready_strong_buy_contract(
    *,
    gate_status: str,
    trap_triggered: bool,
    prior,
    entry_stance: str,
    entry_price,
    stop_loss,
    take_profit,
    rr_ratio,
    position_weight,
    data_confidence: float,
    gate_reasons: list[str] | None = None,
    return_details: bool = False,
):
    """Strict contract for a name to be labelled STRONG BUY and ready now."""
    reasons: list[str] = []
    soft_passes: list[str] = []
    gate_status_norm = str(gate_status or "PASS").upper()
    if gate_status_norm == "REJECT":
        reasons.append(f"gate status {gate_status_norm}")
    elif gate_status_norm != "PASS" and not _gate_review_is_missing_data_only(gate_status_norm, gate_reasons):
        reasons.append(f"gate status {gate_status_norm}")
    if trap_triggered:
        reasons.append("trap safeguard triggered")
    prior_pass, prior_reasons, prior_soft_passes = _prior_passes_ready_contract(prior)
    if not prior_pass:
        reasons.extend(prior_reasons)
    else:
        soft_passes.extend(prior_soft_passes)
    if entry_stance != "Ready":
        reasons.append(f"entry stance is {entry_stance}")
    if not (safe_float(entry_price, default=0.0) > 0):
        reasons.append("entry price missing")
    if not (safe_float(stop_loss, default=0.0) > 0):
        reasons.append("stop missing")
    if not (safe_float(take_profit, default=0.0) > 0):
        reasons.append("target missing")
    min_rr = float(getattr(config, "READY_STRONG_BUY_MIN_RR", 1.50))
    if safe_float(rr_ratio, default=0.0) < min_rr:
        reasons.append(f"R/R below {min_rr:.1f}x")
    min_weight = float(getattr(config, "READY_STRONG_BUY_MIN_POSITION_WEIGHT", 0.005))
    if safe_float(position_weight, default=0.0) < min_weight:
        reasons.append("position size unavailable")
    min_conf = float(getattr(config, "READY_STRONG_BUY_MIN_CONFIDENCE", 0.70))
    if data_confidence < min_conf:
        reasons.append(f"data confidence below {min_conf:.0%}")
    status = "PASS" if not reasons else "FAIL"
    score = _ready_contract_score(
        gate_status=gate_status_norm,
        trap_triggered=trap_triggered,
        prior=prior,
        prior_pass=prior_pass,
        entry_stance=entry_stance,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        rr_ratio=rr_ratio,
        position_weight=position_weight,
        data_confidence=data_confidence,
        gate_reasons=gate_reasons,
    )
    details = {
        "score": round(score, 4),
        "soft_passes": soft_passes,
        "thresholds": {
            "min_rr": min_rr,
            "min_confidence": min_conf,
            "min_position_weight": min_weight,
            "prior_soft_percentile": float(getattr(config, "READY_STRONG_BUY_PRIOR_SOFT_PERCENTILE", 0.80)),
            "prior_soft_confidence": float(getattr(config, "READY_STRONG_BUY_PRIOR_SOFT_CONFIDENCE", 0.55)),
            "prior_soft_coverage": float(getattr(config, "READY_STRONG_BUY_PRIOR_SOFT_COVERAGE", 0.40)),
        },
    }
    if return_details:
        return status, reasons, details
    return status, reasons


def _append_unique_reasons(existing, reasons: list[str]) -> list[str]:
    """Return reasons appended without duplicating text already present."""
    out = list(existing or [])
    seen = {str(item) for item in out}
    for reason in reasons:
        text = str(reason)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _final_strong_buy_contract_blockers(
    candidate,
    *,
    meta_floor: float,
    core_ready_meta_floor: float,
    enforce_action_gate_ceiling: bool = True,
) -> list[str]:
    """Final load-bearing contract for an executable STRONG BUY label."""
    blockers: list[str] = []

    if getattr(candidate, "ticker_identity_warning", None):
        blockers.append("ticker identity warning")
    if getattr(candidate, "trap_safeguard_triggered", False):
        blockers.append("trap safeguard triggered")
    if str(getattr(candidate, "gate_v2_status", "") or "").upper() == "REJECT":
        blockers.append("gate v2 rejected")

    ready_status = str(getattr(candidate, "ready_contract_status", "") or "").upper()
    if ready_status != "PASS":
        blockers.append(f"ready contract {ready_status or 'UNKNOWN'}")
    if not bool(getattr(candidate, "strong_buy_eligible", False)):
        blockers.append("strong-buy eligibility false")

    ceiling = str(getattr(candidate, "action_gate_ceiling", "STRONG BUY") or "STRONG BUY").upper()
    if enforce_action_gate_ceiling and ceiling != "STRONG BUY":
        blockers.append(f"action gate ceiling {ceiling}")

    entry_stance = str(getattr(candidate, "entry_stance", "") or "")
    if entry_stance != "Ready":
        blockers.append(f"entry stance is {entry_stance or 'UNKNOWN'}")
    if not (safe_float(getattr(candidate, "entry_price", None), default=0.0) > 0):
        blockers.append("entry price missing")
    if not (safe_float(getattr(candidate, "stop_loss", None), default=0.0) > 0):
        blockers.append("stop missing")
    if not (safe_float(getattr(candidate, "take_profit", None), default=0.0) > 0):
        blockers.append("target missing")

    min_rr = float(getattr(config, "READY_STRONG_BUY_MIN_RR", 1.50))
    if safe_float(getattr(candidate, "r_r_ratio", None), default=0.0) < min_rr:
        blockers.append(f"R/R below {min_rr:.1f}x")
    min_weight = float(getattr(config, "READY_STRONG_BUY_MIN_POSITION_WEIGHT", 0.005))
    if safe_float(getattr(candidate, "position_weight", None), default=0.0) < min_weight:
        blockers.append("position size unavailable")
    min_conf = float(getattr(config, "READY_STRONG_BUY_MIN_CONFIDENCE", 0.70))
    data_confidence = safe_float(
        getattr(candidate, "effective_data_confidence", None),
        default=safe_float(getattr(candidate, "data_confidence", None), default=1.0),
    )
    if data_confidence < min_conf:
        blockers.append(f"data confidence below {min_conf:.0%}")

    meta_prob = getattr(candidate, "meta_success_prob", None)
    if meta_prob is not None:
        candidate_meta_floor = (
            core_ready_meta_floor
            if str(getattr(candidate, "ready_contract_core_status", "") or "").upper() == "PASS"
            else meta_floor
        )
        if safe_float(meta_prob, default=0.0) < candidate_meta_floor:
            blockers.append(
                f"Meta-label confidence {safe_float(meta_prob, default=0.0):.0%} "
                f"below {candidate_meta_floor:.0%}"
            )

    return blockers


def _demote_failed_strong_buy_contract(candidate, *, blockers: list[str], target_action: str = "BUY") -> None:
    """Demote a failed STRONG BUY while retaining the blocker diagnostics."""
    candidate.action = target_action
    candidate.strong_buy_eligible = False
    candidate.ready_contract_status = "FAIL"
    candidate.ready_contract_reasons = _append_unique_reasons(
        getattr(candidate, "ready_contract_reasons", []),
        blockers,
    )
    candidate.strong_buy_blockers = list(candidate.ready_contract_reasons)


_ACTION_LABEL_RANK = {"MANUAL REVIEW": 0, "AVOID": 1, "NEUTRAL": 2, "BUY": 3, "STRONG BUY": 4}


def _cap_action_label(action: str, ceiling: str) -> str:
    """Return action capped by the stricter of action and ceiling."""
    action_u = str(action or "NEUTRAL").upper()
    ceiling_u = str(ceiling or "STRONG BUY").upper()
    return action_u if _ACTION_LABEL_RANK.get(action_u, 2) <= _ACTION_LABEL_RANK.get(ceiling_u, 4) else ceiling_u


def _failed_strong_buy_target_action(candidate, *, enforce_action_gate_ceiling: bool) -> str:
    ceiling = getattr(candidate, "action_gate_ceiling", "STRONG BUY") if enforce_action_gate_ceiling else "STRONG BUY"
    return _cap_action_label("BUY", ceiling)


def _institutional_prior_weights() -> tuple[float, float]:
    """Return clipped (alpha_weight, rank_weight) to avoid double-counting IP."""
    alpha_w = max(0.0, float(getattr(config, "INSTITUTIONAL_PRIOR_ALPHA_WEIGHT", 0.0)))
    rank_w = max(0.0, float(getattr(config, "INSTITUTIONAL_PRIOR_RANK_WEIGHT", 0.15)))
    max_total = max(0.0, float(getattr(config, "INSTITUTIONAL_PRIOR_MAX_TOTAL_WEIGHT", 0.20)))
    total = alpha_w + rank_w
    if max_total > 0 and total > max_total:
        scale = max_total / total
        logger.warning(
            "Institutional prior weights clipped: alpha=%.3f rank=%.3f max_total=%.3f",
            alpha_w,
            rank_w,
            max_total,
        )
        alpha_w *= scale
        rank_w *= scale
    return alpha_w, rank_w


def _institutional_prior_rank_term(prior, weight: float) -> float:
    """Transparent final-rank contribution from the institutional prior."""
    if prior is None or weight <= 0:
        return 0.0
    score = safe_float(getattr(prior, "score", None), default=0.0)
    confidence = safe_float(getattr(prior, "confidence", None), default=0.0)
    return float(weight * score * max(0.0, min(1.0, confidence)))


def _zscore_array(
    values: list[float | None],
    *,
    fill_z: float = 0.0,
    return_imputed: bool = False,
) -> list[float] | tuple[list[float], list[bool]]:
    """Cross-sectional z-score with explicit missing-data imputation."""
    arr = np.array(
        [float(v) if v is not None and np.isfinite(float(v)) else np.nan for v in values],
        dtype=float,
    )
    mask = ~np.isnan(arr)
    imputed = [not bool(m) for m in mask]
    if mask.sum() < 2:
        out = [0.0 if bool(m) else fill_z for m in mask]
        return (out, imputed) if return_imputed else out
    mean = float(np.mean(arr[mask]))
    std = float(np.std(arr[mask], ddof=0))
    if std < 1e-9:
        out = [0.0 if bool(m) else fill_z for m in mask]
        return (out, imputed) if return_imputed else out
    out = []
    for v in arr:
        if np.isnan(v):
            out.append(fill_z)
        else:
            out.append((float(v) - mean) / std)
    return (out, imputed) if return_imputed else out


def compute_strong_buy_scorecard(candidates: list) -> dict[str, float]:
    """Cross-sectional weighted scorecard (Hull 2018 / Patton-Timmermann 2009).

    Replaces the historic 10-predicate AND-gate with a single continuous score
    that combines the same signals additively.  Multiple weak indicators
    aggregate into a strong one; we keep the trap safeguard as an explicit
    post-veto so commodity-cycle traps still cannot reach STRONG BUY.

    Inputs (all optional — missing values receive a mild confidence-discounted imputation):
        aggregate_score (40%) — pillar-driven signal
        f_score / 9     (15%) — Piotroski quality
        gpa             (10%) — Novy-Marx 2013 gross profitability
        meta_success_prob (10%) — meta-label calibrator output
        institutional_prior_rank (10%) — Asness-Frazzini-Pedersen factor prior
        momentum / quality / value factor scores (5% each)
        stretch - 0.50  (-20% penalty when above) — overextension trap

    Returns:
        {ticker: sb_score} where sb_score is on a roughly z-score scale.
    """
    if not candidates:
        return {}

    def _g(c, name, default=None):
        return getattr(c, name, default)

    agg = [_g(c, "aggregate_score") for c in candidates]
    fscore = [_g(c, "f_score") for c in candidates]
    fcov = [_g(c, "f_score_coverage") for c in candidates]
    gpa = [_g(c, "gpa") for c in candidates]
    meta = [_g(c, "meta_success_prob") for c in candidates]
    ip_pct = [_g(c, "institutional_prior_percentile") for c in candidates]
    mom = [_g(c, "momentum_factor_score") for c in candidates]
    qual = [_g(c, "quality_factor_score") for c in candidates]
    val = [_g(c, "value_factor_score") for c in candidates]
    stretch = [_g(c, "price_vs_sma200_stretch") for c in candidates]

    # F-score is meaningful only with sufficient coverage.
    fscore_filtered = [
        (fs / 9.0) if is_f_score_actionable(fs, fc, config_module=config) else None
        for fs, fc in zip(fscore, fcov)
    ]
    # Meta probability is already 0..1; use raw.
    meta_clean = [(float(m) if m is not None else None) for m in meta]

    impute_missing = bool(getattr(config, "SB_SCORECARD_MISSING_IMPUTATION_ENABLED", True))
    fill_z = float(getattr(config, "SB_SCORECARD_MISSING_FILL_Z", -0.25)) if impute_missing else 0.0
    imputed_weight = float(getattr(config, "SB_SCORECARD_IMPUTED_WEIGHT", 0.50)) if impute_missing else 1.0
    z_agg, imp_agg = _zscore_array(agg, fill_z=fill_z, return_imputed=True)
    z_fs, imp_fs = _zscore_array(fscore_filtered, fill_z=fill_z, return_imputed=True)
    z_gpa, imp_gpa = _zscore_array(gpa, fill_z=fill_z, return_imputed=True)
    z_meta, imp_meta = _zscore_array(meta_clean, fill_z=fill_z, return_imputed=True)
    z_ip, imp_ip = _zscore_array(ip_pct, fill_z=fill_z, return_imputed=True)
    z_mom, imp_mom = _zscore_array(mom, fill_z=fill_z, return_imputed=True)
    z_qual, imp_qual = _zscore_array(qual, fill_z=fill_z, return_imputed=True)
    z_val, imp_val = _zscore_array(val, fill_z=fill_z, return_imputed=True)

    w_agg = float(getattr(config, "SB_SCORECARD_W_AGG", 0.40))
    w_fs = float(getattr(config, "SB_SCORECARD_W_FSCORE", 0.15))
    w_gpa = float(getattr(config, "SB_SCORECARD_W_GPA", 0.10))
    w_meta = float(getattr(config, "SB_SCORECARD_W_META", 0.10))
    w_ip = float(getattr(config, "SB_SCORECARD_W_PRIOR", 0.10))
    w_mom = float(getattr(config, "SB_SCORECARD_W_MOM", 0.05))
    w_qual = float(getattr(config, "SB_SCORECARD_W_QUAL", 0.05))
    w_val = float(getattr(config, "SB_SCORECARD_W_VAL", 0.05))
    stretch_penalty = float(getattr(config, "SB_SCORECARD_STRETCH_PENALTY", 0.20))
    stretch_threshold = float(getattr(config, "SB_SCORECARD_STRETCH_THRESHOLD", 0.50))

    out: dict[str, float] = {}
    for i, c in enumerate(candidates):
        ticker = getattr(c, "ticker", None) or getattr(c, "symbol", None)
        if not ticker:
            continue
        def _term(weight: float, zvals: list[float], imputed: list[bool]) -> float:
            return weight * zvals[i] * (imputed_weight if imputed[i] else 1.0)

        s = (
            _term(w_agg, z_agg, imp_agg)
            + _term(w_fs, z_fs, imp_fs)
            + _term(w_gpa, z_gpa, imp_gpa)
            + _term(w_meta, z_meta, imp_meta)
            + _term(w_ip, z_ip, imp_ip)
            + _term(w_mom, z_mom, imp_mom)
            + _term(w_qual, z_qual, imp_qual)
            + _term(w_val, z_val, imp_val)
        )
        st = stretch[i]
        if st is not None and float(st) > stretch_threshold:
            s -= stretch_penalty * (float(st) - stretch_threshold)
        out[ticker] = float(s)
        # Persist on the candidate for downstream UI / cache.
        try:
            c.sb_score = float(s)
        except (AttributeError, TypeError):
            pass
    return out


def _percentile_regime_label(macro_regime) -> str:
    if isinstance(macro_regime, dict):
        label = (
            macro_regime.get("regime_label")
            or macro_regime.get("regime")
            or macro_regime.get("market_regime")
            or ""
        )
        label = str(label).upper()
        if label:
            return label
        vix = macro_regime.get("vix_percentile")
        if vix is not None:
            try:
                return "BEAR" if float(vix) >= 75 else ("BULL" if float(vix) <= 25 else "NEUTRAL")
            except (TypeError, ValueError):
                pass
    return "NEUTRAL"


def _assign_percentile_actions(candidates: list, *, regime: str = "NEUTRAL") -> None:
    """Assign discovery actions from today's cross-section with safety floors.

    This proposes actions from the day-relative score distribution, then keeps
    hard vetoes (manual review, insufficient data, Ready Strong Buy contract,
    weak F-score cap, meta-label demotion) in force.

    After percentile assignment, Tier 1-4 action gates (Altman Z, Beneish M,
    F-score floor, value caps, quality floors, momentum sanity) further cap
    the action label.  Reasons are recorded on each candidate for UI display.
    """
    if not getattr(config, "USE_PERCENTILE_ACTIONS", True) or not candidates:
        return

    scored = [
        c for c in candidates
        if c.action != "INSUFFICIENT DATA"
        and not getattr(c, "trap_safeguard_triggered", False)
        and getattr(c, "gate_v2_status", "PASS") != "REJECT"
        and safe_float(getattr(c, "aggregate_score", None), default=None) is not None
    ]
    if not scored:
        return

    # Compute the cross-sectional STRONG BUY scorecard once for the cohort.
    # The scorecard is a continuous combination of the same signals that
    # otherwise act as a 10-predicate AND-gate; it widens the STRONG BUY
    # funnel without bypassing the trap safeguard or V2 reject vetoes
    # (already excluded from `scored` above).
    scorecard_enabled = bool(getattr(config, "STRONG_BUY_SCORECARD_ENABLED", True))
    if scorecard_enabled:
        compute_strong_buy_scorecard(scored)

    label = str(regime or "NEUTRAL").upper()
    is_bear = label in ("BEAR", "RISK_OFF", "TRANSITION_DOWN")
    strong_pct = float(getattr(
        config,
        "PERCENTILE_BEAR_STRONG_BUY_PCT" if is_bear else "PERCENTILE_STRONG_BUY_PCT",
        0.03 if is_bear else 0.05,
    ))
    buy_pct = float(getattr(
        config,
        "PERCENTILE_BEAR_BUY_PCT" if is_bear else "PERCENTILE_BUY_PCT",
        0.10 if is_bear else 0.15,
    ))
    neutral_pct = float(getattr(config, "PERCENTILE_NEUTRAL_PCT", 0.50))
    strong_floor = float(getattr(config, "PERCENTILE_STRONG_BUY_MIN_AGG", 0.0))
    buy_floor = float(getattr(config, "PERCENTILE_BUY_MIN_AGG", -0.05))
    min_prior_pct = float(getattr(config, "PERCENTILE_STRONG_BUY_MIN_PRIOR_PCT", 0.85))
    try:
        from engine.auto_tune import get_core_ready_meta_strong_buy_min_prob, get_meta_strong_buy_min_prob
        meta_floor = float(get_meta_strong_buy_min_prob(scored))
        core_ready_meta_floor = float(get_core_ready_meta_strong_buy_min_prob(scored))
    except Exception:
        meta_floor = float(getattr(config, "META_LABEL_STRONG_BUY_MIN_PROB", 0.60))
        core_ready_meta_floor = meta_floor

    ranked = sorted(scored, key=lambda c: safe_float(c.aggregate_score, default=-999.0), reverse=True)
    n = len(ranked)
    strong_cut = max(1, int(np.ceil(n * max(0.0, min(1.0, strong_pct)))))
    buy_cut = max(strong_cut, int(np.ceil(n * max(0.0, min(1.0, buy_pct)))))
    neutral_cut = max(buy_cut, int(np.ceil(n * max(0.0, min(1.0, neutral_pct)))))

    # Scorecard parallel-promotion path: rank by sb_score and find the
    # cohort threshold for the top X%.  A candidate is scorecard-eligible
    # for STRONG BUY when it sits in the top sb-cut AND its sb_score
    # crosses the calibrated absolute threshold.  This widens the funnel
    # without bypassing the trap-safeguard / V2-reject vetoes (excluded
    # from `scored` above) and still respects the F-score floor below.
    sb_rank: dict[str, int] = {}
    sb_threshold = float(getattr(config, "STRONG_BUY_SCORECARD_MIN_Z", 0.50))
    sb_top_pct = float(getattr(config, "STRONG_BUY_SCORECARD_TOP_PCT", 0.05))
    sb_promotions = 0
    if scorecard_enabled:
        sb_ranked = sorted(
            scored,
            key=lambda c: safe_float(getattr(c, "sb_score", None), default=-999.0),
            reverse=True,
        )
        for i, c in enumerate(sb_ranked):
            tk = getattr(c, "ticker", None) or getattr(c, "symbol", None)
            if tk:
                sb_rank[tk] = i
        sb_cut_n = max(1, int(np.ceil(n * max(0.0, min(1.0, sb_top_pct)))))
    else:
        sb_cut_n = 0

    action_counts = {"STRONG BUY": 0, "BUY": 0, "NEUTRAL": 0, "AVOID": 0, "MANUAL REVIEW": 0}
    for idx, c in enumerate(ranked):
        score = safe_float(c.aggregate_score, default=-999.0)
        _fscore_val = safe_float(c.f_score, default=None)
        fscore_block = (
            getattr(config, "F_SCORE_GATE_ENABLED", False)
            and _fscore_val is not None
            and is_f_score_actionable(_fscore_val, getattr(c, "f_score_coverage", None), config_module=config)
            and _fscore_val <= 3
        )

        # Scorecard eligibility for this candidate.
        tk = getattr(c, "ticker", None) or getattr(c, "symbol", None)
        sb_score_val = safe_float(getattr(c, "sb_score", None), default=None)
        scorecard_strong = (
            scorecard_enabled
            and tk is not None
            and sb_rank.get(tk, n + 1) < sb_cut_n
            and sb_score_val is not None
            and sb_score_val >= sb_threshold
            and score >= strong_floor
            and not fscore_block
        )

        if c.action == "MANUAL REVIEW":
            new_action = "MANUAL REVIEW"
        elif (
            idx < strong_cut
            and score >= strong_floor
            and bool(getattr(c, "strong_buy_eligible", False))
            and safe_float(getattr(c, "institutional_prior_percentile", None), default=0.0) >= min_prior_pct
            and not fscore_block
        ):
            meta_prob = getattr(c, "meta_success_prob", None)
            candidate_meta_floor = (
                core_ready_meta_floor
                if str(getattr(c, "ready_contract_core_status", "") or "").upper() == "PASS"
                else meta_floor
            )
            if meta_prob is not None and safe_float(meta_prob, default=0.0) < candidate_meta_floor:
                # Original meta-label gate failure — but if the scorecard
                # confirms strong cohort-relative quality, we keep STRONG BUY
                # rather than demoting to BUY.  The scorecard combines the
                # same signals as the gate but additively.
                if scorecard_strong:
                    new_action = "STRONG BUY"
                    sb_promotions += 1
                else:
                    new_action = "BUY" if score >= buy_floor else "NEUTRAL"
                    c.strong_buy_eligible = False
                    c.ready_contract_status = "FAIL"
                    c.ready_contract_reasons = list(c.ready_contract_reasons or [])
                    meta_reason = (
                        f"Meta-label confidence {safe_float(meta_prob, default=0.0):.0%} below {candidate_meta_floor:.0%}"
                    )
                    c.ready_contract_reasons.append(meta_reason)
                    c.strong_buy_blockers = list(c.ready_contract_reasons)
            else:
                new_action = "STRONG BUY"
        elif scorecard_strong:
            # Parallel scorecard promotion: drops the strong_buy_eligible /
            # IP-percentile / meta-prob conjunctive checks because their
            # aggregate signal is already captured by sb_score.  Trap
            # safeguard and V2-reject still veto via `scored` filter.
            new_action = "STRONG BUY"
            sb_promotions += 1
        elif idx < buy_cut and score >= buy_floor and not fscore_block:
            new_action = "BUY"
        elif idx < neutral_cut or score >= config.SCORE_KEEP_THRESHOLD:
            new_action = "NEUTRAL"
        else:
            new_action = "AVOID"

        if fscore_block and new_action in ("STRONG BUY", "BUY"):
            new_action = "NEUTRAL"
        c.action = new_action
        action_counts[new_action] = action_counts.get(new_action, 0) + 1

    logger.info(
        "Percentile discovery actions applied (%s): strong_cut=%d buy_cut=%d n=%d "
        "scorecard_top=%d sb_promotions=%d counts=%s",
        label,
        strong_cut,
        buy_cut,
        n,
        sb_cut_n,
        sb_promotions,
        action_counts,
    )

    # Tier 1-4 hard gates run AFTER the percentile assignment so they can
    # only tighten the label (never loosen it).  This is where SEB-style
    # distress and TRST-style overbought-overpriced names are stopped.
    if getattr(config, "ACTION_GATES_ENABLED", True):
        try:
            from engine.action_gates import apply_action_gates, cap_action
            # Threshold-learner overlay: if enabled, the active profile may
            # tighten or loosen thresholds vs the base config.  The drift
            # monitor in daily_orchestrator forces the conservative profile
            # when realised IC degrades on consecutive runs.
            gate_config = config
            active_profile = "moderate"
            if getattr(config, "THRESHOLD_LEARNER_ENABLED", True):
                try:
                    from engine.threshold_learner import (
                        get_active_profile_overlay, save_state,
                    )
                    active_profile, overlay, tl_state = get_active_profile_overlay(
                        drift_alert=False, config_module=config,
                    )
                    gate_config = overlay
                    save_state(tl_state, config_module=config)
                    for c in candidates:
                        c.threshold_profile = active_profile
                    logger.info("Threshold-learner active profile: %s", active_profile)
                except Exception as exc:
                    logger.debug("Threshold-learner overlay failed: %s", exc)
            gate_summary = apply_action_gates(candidates, config_module=gate_config)
            shadow = bool(getattr(config, "ACTION_GATES_SHADOW", False))
            for c in candidates:
                ceiling = getattr(c, "action_gate_ceiling", "STRONG BUY")
                if ceiling == "STRONG BUY" or c.action == "MANUAL REVIEW":
                    continue
                capped = cap_action(c.action, ceiling)
                if capped == c.action:
                    continue
                if shadow:
                    # Shadow mode: log only, don't downgrade
                    logger.info(
                        "Action gate shadow: %s would downgrade %s -> %s (%s)",
                        c.ticker, c.action, capped,
                        "; ".join((c.action_gate_reasons or [])[:3]),
                    )
                    continue
                # Capping STRONG BUY also forfeits the strong-buy contract
                if c.action == "STRONG BUY":
                    c.strong_buy_eligible = False
                    if not getattr(c, "ready_contract_reasons", None):
                        c.ready_contract_reasons = []
                    c.ready_contract_reasons = list(c.ready_contract_reasons) + (c.action_gate_reasons or [])
                    c.strong_buy_blockers = list(c.ready_contract_reasons)
                    c.ready_contract_status = "FAIL"
                c.action = capped
            logger.info(
                "Action gates applied: %d downgrades, %d limit-price suggestions, fails=%s",
                gate_summary.get("downgrades", 0),
                gate_summary.get("limits", 0),
                gate_summary.get("tier_fail_counts", {}),
            )
        except Exception as exc:
            logger.exception("Action gates failed (continuing with percentile labels): %s", exc)

    enforce_gate_ceiling = not bool(getattr(config, "ACTION_GATES_SHADOW", False))

    # High-conviction scorecard override (post-action-gates).
    # When the additive scorecard evidence is overwhelming, preserve it as
    # a promotion proposal, then validate the final STRONG BUY contract.
    # Rationale: any single
    # gate (F-score, EV/EBIT, entry-stance) is a noisy filter; a +1.0σ
    # composite combining 8 such signals has substantially higher SNR than
    # any one of them.  Trap-safeguard / V2-reject still veto via the
    # earlier `scored` filter so this cannot resurrect a true value trap.
    if scorecard_enabled and bool(getattr(config, "STRONG_BUY_SCORECARD_OVERRIDE_ENABLED", True)):
        override_z = float(getattr(config, "STRONG_BUY_SCORECARD_OVERRIDE_Z", 1.0))
        override_top_n = int(getattr(config, "STRONG_BUY_SCORECARD_OVERRIDE_TOP_N", 5))
        # Re-rank by sb_score across all original `scored` candidates.
        eligible = [c for c in scored if getattr(c, "sb_score", None) is not None]
        eligible.sort(key=lambda c: safe_float(c.sb_score, default=-999.0), reverse=True)
        n_override = 0
        n_override_capped = 0
        for c in eligible[:override_top_n]:
            sb_val = safe_float(c.sb_score, default=-999.0)
            if sb_val < override_z:
                break
            # Don't resurrect MANUAL REVIEW (trap / V2 reject reached scoring earlier
            # would already have been filtered out of `scored`).
            if c.action in ("MANUAL REVIEW", "INSUFFICIENT DATA"):
                continue
            c.scorecard_override = True
            blockers = _final_strong_buy_contract_blockers(
                c,
                meta_floor=meta_floor,
                core_ready_meta_floor=core_ready_meta_floor,
                enforce_action_gate_ceiling=enforce_gate_ceiling,
            )
            if blockers:
                target = _failed_strong_buy_target_action(c, enforce_action_gate_ceiling=enforce_gate_ceiling)
                c.scorecard_override_blockers = list(blockers)
                if (
                    c.action == "STRONG BUY"
                    or _ACTION_LABEL_RANK.get(target, 2) > _ACTION_LABEL_RANK.get(str(c.action).upper(), 2)
                ):
                    logger.info(
                        "Scorecard override capped: %s sb_score=%+.3f blocked from STRONG BUY -> %s (%s)",
                        c.ticker, sb_val, target, "; ".join(blockers[:3]),
                    )
                    _demote_failed_strong_buy_contract(c, blockers=blockers, target_action=target)
                    n_override_capped += 1
                continue
            if c.action != "STRONG BUY":
                logger.info(
                    "Scorecard override: %s sb_score=%+.3f >= %.2f -> STRONG BUY (was %s)",
                    c.ticker, sb_val, override_z, c.action,
                )
                c.action = "STRONG BUY"
                n_override += 1
        if n_override:
            logger.info("Scorecard override: %d candidates restored to STRONG BUY", n_override)
        if n_override_capped:
            logger.info("Scorecard override: %d candidates capped below STRONG BUY by final contract", n_override_capped)

    n_contract_demotions = 0
    for c in candidates:
        if str(getattr(c, "action", "") or "").upper() != "STRONG BUY":
            continue
        blockers = _final_strong_buy_contract_blockers(
            c,
            meta_floor=meta_floor,
            core_ready_meta_floor=core_ready_meta_floor,
            enforce_action_gate_ceiling=enforce_gate_ceiling,
        )
        if not blockers:
            continue
        target = _failed_strong_buy_target_action(c, enforce_action_gate_ceiling=enforce_gate_ceiling)
        logger.info(
            "Final STRONG BUY contract demotion: %s -> %s (%s)",
            getattr(c, "ticker", ""),
            target,
            "; ".join(blockers[:3]),
        )
        _demote_failed_strong_buy_contract(c, blockers=blockers, target_action=target)
        n_contract_demotions += 1
    if n_contract_demotions:
        logger.info("Final STRONG BUY contract demoted %d candidates", n_contract_demotions)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CandidateRejection:
    """Tracks why a candidate was rejected."""
    ticker: str
    name: str
    exchange: str
    stage: str
    reason: str


@dataclass
class ScoredCandidate:
    """A fully-scored discovery candidate."""
    ticker: str
    name: str
    exchange: str
    country: str
    sector: str
    industry: str
    market_cap: float
    currency: str
    # Scores
    aggregate_score: float
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    forecast_score: float
    action: str
    why: str
    # FX
    fx_penalty_applied: bool
    fx_penalty_pct: float
    # Portfolio fit
    max_correlation: float
    correlated_with: str
    sector_weight_if_added: float
    portfolio_fit_score: float
    # Momentum
    momentum_score: float
    return_90d: float
    return_30d: float
    volume_ratio: float
    return_10d: float
    vol_20d: float | None = None
    # 90-day expected return (unified target)
    expected_return_90d: float = 0.0
    # Entry-quality context
    analyst_target: float | None = None
    analyst_upside: float | None = None
    num_analysts: int | None = None
    insider_buys: int = 0
    insider_sells: int = 0
    insider_net: str = ""
    pe_ratio: float | None = None
    peg_ratio: float | None = None
    revenue_growth: float | None = None
    roe: float | None = None
    short_pct: float | None = None
    beta_90d: float | None = None
    debt_to_equity: float | None = None
    entry_stance: str = "Ready"
    ticker_identity_warning: str | None = None
    # Risk overlay
    parabolic_penalty: float = 0.0
    is_parabolic: bool = False
    earnings_near: bool = False
    earnings_imminent: bool = False
    earnings_days: int | None = None
    cap_tier: str = "unknown"
    confidence_discount: float = 1.0
    effective_data_confidence: float | None = None
    max_weight_scale: float = 1.0
    # Post-earnings + 52w high
    post_earnings_recent: bool = False
    post_earnings_days: int | None = None
    earnings_miss: bool = False
    earnings_miss_pct: float | None = None
    near_52w_high: bool = False
    pct_from_52w_high: float | None = None
    # Entry lens (momentum / value / quality)
    entry_lens: str = "momentum"
    # Trading strategy (entry + stop + sizing)
    entry_price: float | None = None
    entry_method: str = ""
    entry_zone_low: float | None = None
    entry_zone_high: float | None = None
    fill_probability: float | None = None
    stop_loss: float | None = None
    stop_method: str = ""
    stop_distance_pct: float | None = None
    take_profit: float | None = None
    target_method: str = ""
    position_size_shares: int = 0
    position_weight: float = 0.0
    risk_amount: float = 0.0
    r_r_ratio: float | None = None
    sizing_method: str = ""
    kelly_cap_fraction: float | None = None
    support_levels: dict = field(default_factory=dict)
    regime_info: dict = field(default_factory=dict)
    regime: str | None = None
    # Dividend safety
    dividend_yield: float | None = None
    payout_ratio: float | None = None
    ex_dividend_date: str | None = None
    ex_dividend_days: int | None = None
    five_year_avg_yield: float | None = None
    # Balance sheet strength
    balance_sheet_grade: str | None = None
    net_debt_ebitda: float | None = None
    current_ratio: float | None = None
    cash_to_debt: float | None = None
    # Governance red flag
    governance_flag: bool = False
    governance_reasons: list = field(default_factory=list)
    # Asymmetric / binary outcome flag
    asymmetric_risk_flag: bool = False
    asymmetric_risk_reason: str | None = None
    # Fundamental quality (Fama-French 2015 / Asness QMJ)
    quality_score_fundamental: float | None = None
    gross_profitability: float | None = None
    fcf_to_assets: float | None = None
    fcf_yield: float | None = None
    earnings_stability: float | None = None
    eps_growth_variance_5y: float | None = None
    qmj_factor_score: float | None = None
    pead_factor_score: float | None = None
    sue_score: float | None = None
    revision_momentum_3m: float | None = None
    bab_factor_score: float | None = None
    turnover_cost_score: float | None = None
    # Enterprise-grade factor bundle (Loughran-Wellman 2011, Piotroski 2000, Novy-Marx 2013)
    enterprise_value: float | None = None
    ev_ebit: float | None = None
    ev_ebitda: float | None = None
    ebit_yield: float | None = None
    ev_ebit_score: float | None = None
    gpa: float | None = None
    gpa_score: float | None = None
    f_score: int | None = None
    f_score_gate: bool = False
    f_score_score: float | None = None
    f_score_coverage: float | None = None
    # Trend anchor — persisted for stretch diagnostics in signal_backtest
    sma_200: float | None = None
    price_vs_sma200_stretch: float | None = None
    # Distress / quality-of-earnings (Tier-1 action gates)
    altman_z: float | None = None
    altman_zone: str = "unknown"
    altman_coverage: float = 0.0
    beneish_m: float | None = None
    accruals_factor_score: float | None = None
    investment_factor_score: float | None = None
    op_margin_yoy_delta: float | None = None
    rsi: float | None = None
    realized_vol_pctile: float | None = None
    # Action-gate ceiling result (Tier 1-4 combined)
    action_gate_ceiling: str = "STRONG BUY"
    action_gate_reasons: list = field(default_factory=list)
    action_gate_flags: dict = field(default_factory=dict)
    limit_price: float | None = None
    limit_price_method: str | None = None
    limit_price_rationale: str | None = None
    # Active threshold profile (Tier 5 self-learner)
    threshold_profile: str = "moderate"
    # Gate v2 telemetry / safeguards
    gate_v2_status: str = "PASS"
    gate_v2_reasons: list = field(default_factory=list)
    trap_safeguard_triggered: bool = False
    trap_safeguard_reason: str | None = None
    institutional_prior_score: float = 0.0
    institutional_prior_percentile: float = 0.0
    institutional_prior_confidence: float = 0.0
    institutional_prior_coverage: float = 0.0
    institutional_prior_components: dict = field(default_factory=dict)
    institutional_prior_rank: float = 0.0
    ready_contract_core_status: str = "FAIL"
    ready_contract_core_reasons: list = field(default_factory=list)
    ready_contract_status: str = "FAIL"
    ready_contract_reasons: list = field(default_factory=list)
    ready_contract_score: float | None = None
    ready_contract_soft_passes: list = field(default_factory=list)
    strong_buy_blockers: list = field(default_factory=list)
    strong_buy_eligible: bool = False
    ready_lane_score: float | None = None
    ready_lane_missing_fields: list = field(default_factory=list)
    # Factor momentum (Ehsani & Linnainmaa 2022)
    factor_momentum_tilt: dict = field(default_factory=dict)
    network_momentum: float | None = None
    qa_sentiment_score: float | None = None
    # ML alpha (raw prediction before blend)
    ml_alpha_raw: float | None = None
    ml_shadow_score: float | None = None
    meta_success_prob: float | None = None
    sleeve_composite: float | None = None
    sleeve_momentum: float | None = None
    sleeve_quality: float | None = None
    sleeve_value: float | None = None
    sleeve_low_risk: float | None = None
    sleeve_pead: float | None = None
    sleeve_ready: float | None = None
    analysis_degraded: bool = False
    analysis_degraded_reason: str | None = None
    # Alpha-only rank (ignoring portfolio fit — shows pure signal strength)
    alpha_rank: float = 0.0
    # Split final_rank into orthogonal views (roadmap item #5):
    #   selection_rank    — pure signal quality (net-of-cost alpha)
    #   timing_rank       — trend confirmation (momentum)
    #   portfolio_fit_rank — diversification contribution
    # final_rank remains the weighted blend consumed by existing UI/ordering.
    selection_rank: float = 0.0
    timing_rank: float = 0.0
    portfolio_fit_rank: float = 0.0
    # STRONG BUY scorecard (continuous combination of pillar + factor signals;
    # additive substitute for the historic 10-predicate AND-gate).
    sb_score: float | None = None
    scorecard_override: bool = False
    scorecard_override_blockers: list = field(default_factory=list)
    # Distribution-free conformal p-value (Vovk-Gammerman-Shafer 2005);
    # populated when calibration set has ≥ 50 mature 90d returns.
    conformal_p: float | None = None
    # Final
    final_rank: float = 0.0


@dataclass
class DiscoveryResult:
    """Complete discovery run results."""
    candidates: list[ScoredCandidate] = field(default_factory=list)
    rejections: list[CandidateRejection] = field(default_factory=list)
    screened_count: int = 0
    after_momentum_screen: int = 0
    after_quick_filter: int = 0
    after_corr_filter: int = 0
    after_quick_rank: int = 0
    fully_scored: int = 0
    run_time_seconds: float = 0.0
    fx_penalties_applied: int = 0
    error: str | None = None
    stage_timings: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Currency detection
# ---------------------------------------------------------------------------

_EXCHANGE_CURRENCY = {
    "LSE": "GBX", "XETRA": "EUR", "EURONEXT": "EUR", "TSX": "CAD",
    "NYSE": "USD", "NASDAQ": "USD", "AMEX": "USD",
}

_SUFFIX_CURRENCY = {
    ".L": "GBX", ".DE": "EUR", ".PA": "EUR", ".MI": "EUR", ".MC": "EUR",
    ".AS": "EUR", ".BR": "EUR", ".LS": "EUR",
    ".SW": "CHF", ".TO": "CAD", ".AX": "AUD",
    ".T": "JPY", ".HK": "HKD", ".SI": "SGD",
    ".KS": "KRW", ".ST": "SEK", ".CO": "DKK",
    ".HE": "EUR", ".OL": "NOK",
}


def _detect_currency(exchange: str, ticker: str) -> str:
    """Detect currency from exchange and ticker suffix."""
    for suffix, currency in _SUFFIX_CURRENCY.items():
        if ticker.endswith(suffix):
            return currency
    return _EXCHANGE_CURRENCY.get(exchange, "USD")


def _is_gbp_denominated(currency: str) -> bool:
    return currency in ("GBP", "GBX")


def _detect_exchange(ticker: str) -> str:
    """Infer exchange from ticker suffix."""
    suffix_map = {
        ".L": "LSE", ".DE": "XETRA", ".PA": "EURONEXT", ".MI": "MIL",
        ".MC": "BME", ".AS": "AMS", ".SW": "SIX", ".TO": "TSX",
        ".AX": "ASX", ".T": "TSE", ".HK": "HKEX", ".SI": "SGX",
        ".KS": "KRX", ".ST": "OMX", ".CO": "OMX", ".HE": "OMX",
        ".OL": "OSE",
    }
    for suffix, exch in suffix_map.items():
        if ticker.endswith(suffix):
            return exch
    return "US"


_NAME_STOPWORDS = {
    "adr", "ag", "corp", "corporation", "group", "holdings", "holding", "inc",
    "limited", "ltd", "nv", "ordinary", "ordinarys", "plc", "sa", "se", "spa",
    "the",
}

_SECTOR_NORMALIZATION = {
    "basic materials": "materials",
    "communication services": "communication",
    "consumer cyclical": "consumer discretionary",
    "consumer defensive": "consumer staples",
    "financial services": "financials",
}

_COUNTRY_KEYWORDS = {
    "AU": ("australia",),
    "CA": ("canada",),
    "CH": ("switzerland",),
    "DE": ("germany",),
    "DK": ("denmark",),
    "ES": ("spain",),
    "FI": ("finland",),
    "FR": ("france",),
    "GB": ("united kingdom", "uk", "great britain", "england"),
    "HK": ("hong kong",),
    "IT": ("italy",),
    "JP": ("japan",),
    "KR": ("south korea", "korea"),
    "NL": ("netherlands",),
    "NO": ("norway",),
    "SE": ("sweden",),
    "SG": ("singapore",),
    "US": ("united states", "usa", "us"),
}


def _name_tokens(value: str) -> set[str]:
    """Normalize names and tickers into comparable tokens."""
    if not value:
        return set()
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    return {
        token for token in cleaned.split()
        if len(token) >= 2 and token not in _NAME_STOPWORDS
    }


def _normalize_sector(value: str) -> str:
    if not value:
        return ""
    text = str(value).strip().lower()
    return _SECTOR_NORMALIZATION.get(text, text)


def _country_matches(expected_code: str, actual_country: str) -> bool:
    if not expected_code or not actual_country:
        return True
    keywords = _COUNTRY_KEYWORDS.get(str(expected_code).upper())
    if not keywords:
        return True
    actual = str(actual_country).strip().lower()
    return any(keyword in actual for keyword in keywords)


def _compute_ticker_identity_warning(
    ticker: str,
    candidate_meta: dict | None,
    result_name: str,
) -> str | None:
    """Warn when discovery metadata and live quote metadata appear inconsistent."""
    from utils.data_fetch import get_cached_ticker_info, get_ticker_info

    info = get_cached_ticker_info(ticker)
    if (
        not info
        and getattr(config, "DISCOVERY_IDENTITY_WARNING_ALLOW_YAHOO_NETWORK", False)
    ):
        info = get_ticker_info(ticker, allow_network=True)
    if not info:
        return None

    candidate_meta = candidate_meta or {}
    source_name = candidate_meta.get("companyName") or ticker
    source_sector = candidate_meta.get("sector") or ""
    source_country = candidate_meta.get("country") or ""
    actual_name = info.get("longName") or info.get("shortName") or result_name or ticker
    actual_sector = info.get("sector") or ""
    actual_country = info.get("country") or ""

    mismatches: list[str] = []

    source_tokens = _name_tokens(source_name)
    actual_tokens = _name_tokens(actual_name)
    if source_tokens and actual_tokens and source_tokens.isdisjoint(actual_tokens):
        mismatches.append("name")

    if source_sector and actual_sector and _normalize_sector(source_sector) != _normalize_sector(actual_sector):
        mismatches.append("sector")

    if source_country and actual_country and not _country_matches(source_country, actual_country):
        mismatches.append("country")

    if len(mismatches) >= 2:
        return "Verify ticker identity: live quote metadata disagrees with the discovery universe."
    return None


def _derive_entry_stance(
    *,
    governance_flag: bool,
    asymmetric_risk_flag: bool,
    earnings_imminent: bool,
    is_parabolic: bool,
    analyst_upside: float | None,
    near_52w_high: bool,
    return_30d: float,
    insider_sells: int,
    insider_buys: int,
    earnings_near: bool,
) -> str:
    """Classify whether a candidate looks ready, pullback-worthy, or not actionable today."""
    near_high_pullback_ret = float(getattr(config, "READY_ENTRY_NEAR_HIGH_PULLBACK_RET30", 0.12))
    near_high_min_upside = float(getattr(config, "READY_ENTRY_NEAR_HIGH_MIN_UPSIDE", 8.0))
    near_high_requires_pullback = near_52w_high and (
        return_30d >= near_high_pullback_ret
        or (analyst_upside is not None and analyst_upside < near_high_min_upside)
    )
    if (
        governance_flag
        or asymmetric_risk_flag
        or earnings_imminent
        or (is_parabolic and analyst_upside is not None and analyst_upside < 0)
        or (near_52w_high and return_30d >= 0.25)
        or (
            insider_sells > insider_buys
            and analyst_upside is not None
            and analyst_upside < 0
        )
    ):
        return "Watch Only"

    if (
        is_parabolic
        or near_high_requires_pullback
        or (analyst_upside is not None and analyst_upside < 5)
        or insider_sells > insider_buys
        or earnings_near
    ):
        return "Pullback Preferred"

    return "Ready"


# ---------------------------------------------------------------------------
# Stage 1: Universe Assembly (FMP US + yfinance Global)
# ---------------------------------------------------------------------------

def _stage_universe_assembly(
    existing_tickers: set[str],
    progress_callback=None,
) -> list[dict]:
    """Assemble the full candidate universe from multiple sources."""
    all_candidates = []
    seen_symbols = set()

    # --- Part A: FMP Screener for US exchanges ---
    exchanges = getattr(config, "DISCOVERY_EXCHANGES", ["NYSE", "NASDAQ", "AMEX"])
    mcap_min = getattr(config, "DISCOVERY_MIN_MCAP", 50_000_000)
    vol_min = getattr(config, "DISCOVERY_VOLUME_MIN", 50_000)
    fmp_limit = getattr(config, "DISCOVERY_FMP_LIMIT", 1000)

    for i, exchange in enumerate(exchanges):
        if progress_callback:
            progress_callback(f"FMP: Screening {exchange}...", i, len(exchanges) + 1)

        results = screen_stocks(
            exchange=exchange,
            market_cap_min=mcap_min,
            market_cap_max=None,  # No upper cap
            volume_min=vol_min,
            limit=fmp_limit,
        )

        if not results:
            logger.warning("No FMP results for %s", exchange)
            continue

        for stock in results:
            symbol = stock.get("symbol", "")
            if not symbol or symbol in seen_symbols or symbol in existing_tickers:
                continue
            stock["_exchange_query"] = exchange
            stock["_source"] = "fmp"
            stock["_region"] = "US"
            stock["_universe_tier"] = 0
            stock["_index_source"] = "FMP_SCREENER"
            if _candidate_exclusion_reason(stock):
                continue
            seen_symbols.add(symbol)
            all_candidates.append(stock)

    fmp_count = len(all_candidates)
    logger.info("FMP screener: %d US candidates from %d exchanges", fmp_count, len(exchanges))

    # --- Part B: Daily global snapshot (systematic, no weekday rotation) ---
    use_global = getattr(config, "DISCOVERY_USE_GLOBAL_UNIVERSE", True)
    if use_global:
        if progress_callback:
            progress_callback("Loading global universe...", len(exchanges), len(exchanges) + 1)

        try:
            from utils.global_universe import build_daily_universe_snapshot, get_region_for_ticker

            region_caps = getattr(config, "DISCOVERY_GLOBAL_REGION_CAPS", None)
            max_tier = int(getattr(config, "DISCOVERY_GLOBAL_MAX_TIER", 2))
            include_tier2 = bool(getattr(config, "DISCOVERY_GLOBAL_INCLUDE_TIER2_DAILY", True))
            global_entries = build_daily_universe_snapshot(
                exclude_tickers=existing_tickers,
                max_tier=max_tier,
                include_tier2=include_tier2,
                region_caps=region_caps,
            )

            region_counts: dict[str, int] = {}
            tier_counts: dict[int, int] = {}

            for meta in global_entries:
                ticker = meta.ticker
                if ticker in seen_symbols:
                    continue
                seen_symbols.add(ticker)
                exchange = meta.exchange if meta else _detect_exchange(ticker)
                region = get_region_for_ticker(ticker) or meta.country or ""
                region_counts[region] = region_counts.get(region, 0) + 1
                tier_counts[meta.tier] = tier_counts.get(meta.tier, 0) + 1
                all_candidates.append({
                    "symbol": ticker,
                    "companyName": ticker,  # Will be enriched later
                    "country": meta.country if meta else "",
                    "sector": meta.sector if meta else "",
                    "_exchange_query": exchange,
                    "_source": "global_universe_daily",
                    "_region": region,
                    "_universe_tier": meta.tier if meta else None,
                    "_index_source": meta.index_source if meta else "",
                })

            global_count = len(all_candidates) - fmp_count
            logger.info(
                "Global universe daily snapshot: %d candidates added (tiers=%s, regions=%s)",
                global_count,
                tier_counts,
                region_counts,
            )
        except ImportError:
            logger.warning("global_universe module not found, skipping non-US screening")

    # --- Part C: Dynamic universe supplement (reconstituted tickers) ---
    # Preserve the metadata captured by reconstitute_universe() (country,
    # exchange, sector, market_cap, avg_dollar_volume) so downstream gates
    # — especially the region-aware dollar-volume floor at Stage 2 — can
    # evaluate these names on equal footing with static-universe entries.
    dynamic_rejected: set[str] = set()
    try:
        from utils.global_universe import get_dynamic_entries, get_dynamic_rejected
        dynamic_rejected = get_dynamic_rejected()
        dynamic_entries = get_dynamic_entries()
        dynamic_added = 0
        for entry in dynamic_entries:
            ticker = entry.get("ticker", "")
            if not ticker or ticker in seen_symbols or ticker in existing_tickers:
                continue
            dynamic_candidate = {
                "symbol": ticker,
                "companyName": ticker,
                "country": entry.get("country", ""),
                "sector": entry.get("sector", ""),
                "_exchange_query": entry.get("exchange", ""),
                "_source": "dynamic_universe",
                "_region": entry.get("country", ""),
                "_universe_tier": 3,
                "_index_source": "DYNAMIC",
                "_market_cap": entry.get("market_cap") or 0,
                "_avg_dollar_volume": entry.get("avg_dollar_volume") or 0,
            }
            if _candidate_exclusion_reason(dynamic_candidate):
                continue
            seen_symbols.add(ticker)
            all_candidates.append(dynamic_candidate)
            dynamic_added += 1
        if dynamic_added:
            logger.info("Dynamic universe supplement: %d candidates added", dynamic_added)
    except Exception as e:
        logger.debug("Dynamic universe supplement skipped: %s", e)

    # --- Part D: ETF Holdings Decomposition (Petajisto 2011) ---
    # This is now a fallback for interactive/UI-triggered discovery runs that
    # didn't go through the orchestrator's reconstitute_universe() step. When
    # reconstitution DID run, any ETF holding that failed the liquidity /
    # market-cap gates will appear in dynamic_rejected and must NOT be
    # directly re-injected here — that would bypass the exact quality control
    # Part C enforces.
    if getattr(config, "DISCOVERY_USE_ETF_DECOMPOSITION", True):
        try:
            from utils.global_universe import decompose_etf_holdings
            etf_exclude = set(existing_tickers) | dynamic_rejected
            etf_tickers = decompose_etf_holdings(exclude_tickers=etf_exclude)
            etf_added = 0
            for ticker in etf_tickers:
                if ticker in seen_symbols or ticker in existing_tickers:
                    continue
                etf_candidate = {
                    "symbol": ticker,
                    "companyName": ticker,
                    "_exchange_query": "",
                    "_source": "etf_decomposition",
                    "_region": "",
                    "_universe_tier": 2,
                    "_index_source": "ETF",
                }
                if _candidate_exclusion_reason(etf_candidate):
                    continue
                seen_symbols.add(ticker)
                all_candidates.append(etf_candidate)
                etf_added += 1
            if etf_added:
                logger.info("ETF decomposition: %d candidates added", etf_added)
        except Exception as e:
            logger.debug("ETF decomposition skipped: %s", e)

    # --- Part E: Forced coverage list for benchmark-quality names ---
    # Some large-cap names can fall through third-party screeners on a given
    # day. Keep a tiny auditable list for names we explicitly want the funnel
    # to cache and evaluate, without relaxing any later quality or readiness
    # gates.
    forced_added = 0
    for entry in getattr(config, "DISCOVERY_FORCE_INCLUDE_TICKERS", []) or []:
        symbol = str(entry.get("symbol", "") if isinstance(entry, dict) else entry).upper().strip()
        if not symbol or symbol in seen_symbols or symbol in existing_tickers:
            continue
        if isinstance(entry, dict):
            forced_candidate = {
                "symbol": symbol,
                "companyName": entry.get("companyName") or entry.get("name") or symbol,
                "country": entry.get("country", "US"),
                "sector": entry.get("sector", ""),
                "industry": entry.get("industry", ""),
                "_exchange_query": entry.get("exchange", ""),
                "_source": "forced_coverage",
                "_region": entry.get("country", "US"),
                "_universe_tier": 1,
                "_index_source": entry.get("index_source", "FORCED_COVERAGE"),
            }
        else:
            forced_candidate = {
                "symbol": symbol,
                "companyName": symbol,
                "country": "US",
                "_exchange_query": "",
                "_source": "forced_coverage",
                "_region": "US",
                "_universe_tier": 1,
                "_index_source": "FORCED_COVERAGE",
            }
        if _candidate_exclusion_reason(forced_candidate):
            continue
        seen_symbols.add(symbol)
        all_candidates.append(forced_candidate)
        forced_added += 1
    if forced_added:
        logger.info("Forced coverage universe: %d candidates added", forced_added)

    # --- Part F: Challenge lane for reviewed high-conviction ideas ---
    # These names are injected only into the candidate funnel. They must still
    # pass price/liquidity, ranking, deep scoring, action, and readiness gates.
    challenge_added = 0
    for raw in _challenge_entries():
        symbol = raw.get("symbol") if isinstance(raw, dict) else raw
        symbol = str(symbol or "").upper().strip()
        if not symbol or symbol in seen_symbols or symbol in existing_tickers:
            continue
        challenge_candidate = _make_challenge_candidate(symbol, raw)
        if _candidate_exclusion_reason(challenge_candidate):
            continue
        seen_symbols.add(symbol)
        all_candidates.append(challenge_candidate)
        challenge_added += 1
    if challenge_added:
        logger.info("Challenge coverage universe: %d candidates added", challenge_added)

    return all_candidates


# ---------------------------------------------------------------------------
# Stage 2: Momentum Screen (download price data + rank)
# ---------------------------------------------------------------------------

def _compute_momentum_metrics(prices_df: pd.DataFrame, ticker: str) -> dict | None:
    """Compute momentum metrics from a price series."""
    try:
        closes = prices_df["Close"]
        if isinstance(closes, pd.DataFrame):
            closes = closes[ticker] if ticker in closes.columns else None
        if closes is None or len(closes.dropna()) < 30:
            return None

        closes = closes.dropna()
        values = closes.values

        # Returns over different periods
        ret_90d = (values[-1] / values[-min(90, len(values))] - 1) if len(values) >= 20 else 0
        ret_30d = (values[-1] / values[-min(30, len(values))] - 1) if len(values) >= 15 else 0
        ret_10d = (values[-1] / values[-min(10, len(values))] - 1) if len(values) >= 10 else 0

        # Volume analysis
        if "Volume" in prices_df.columns:
            vol = prices_df["Volume"]
            if isinstance(vol, pd.DataFrame):
                vol = vol[ticker] if ticker in vol.columns else None
            if vol is not None and len(vol.dropna()) >= 20:
                vol_vals = vol.dropna().values
                avg_vol_10 = np.mean(vol_vals[-10:]) if len(vol_vals) >= 10 else 0
                avg_vol_60 = np.mean(vol_vals[-60:]) if len(vol_vals) >= 60 else np.mean(vol_vals)
                volume_ratio = avg_vol_10 / max(avg_vol_60, 1)
                avg_volume = np.mean(vol_vals[-20:]) if len(vol_vals) >= 20 else 0
            else:
                volume_ratio = 1.0
                avg_volume = 0
        else:
            volume_ratio = 1.0
            avg_volume = 0

        # 12-minus-1-month momentum (Jegadeesh & Titman, 1993)
        # Skip the most recent month (21 trading days) to avoid short-term reversal
        if len(values) >= 252:
            ret_12m1m = values[-21] / values[-252] - 1
        elif len(values) >= 63:
            ret_12m1m = values[-21] / values[-min(len(values), 252)] - 1
        else:
            ret_12m1m = ret_90d  # fallback for short histories

        # Distance from 52-week high
        high_252 = np.max(values[-min(252, len(values)):])
        pct_from_high = values[-1] / high_252 if high_252 > 0 else 0

        # Price vs SMA-50
        sma_50 = np.mean(values[-min(50, len(values)):])
        above_sma50 = values[-1] > sma_50

        return {
            "ret_90d": float(ret_90d),
            "ret_30d": float(ret_30d),
            "ret_10d": float(ret_10d),
            "ret_12m1m": float(ret_12m1m),
            "volume_ratio": float(volume_ratio),
            "avg_volume": float(avg_volume),
            "pct_from_high": float(pct_from_high),
            "above_sma50": above_sma50,
            "last_price": float(values[-1]),
            "returns": pd.Series(closes).pct_change().dropna().values[-60:],
        }
    except Exception:
        return None


def _stage_momentum_screen(
    candidates: list[dict],
    progress_callback=None,
) -> tuple[list[dict], dict]:
    """Compute momentum scores using feature store (cache-first).

    Returns (filtered_candidates, price_cache).
    Uses the feature store for batch price factors — only downloads
    stale tickers, reuses cached factors for everything else.
    """
    top_n = getattr(config, "MOMENTUM_TOP_N_PRESCREEN", 150)
    min_avg_vol = getattr(config, "DISCOVERY_VOLUME_MIN", 50_000)
    dollar_floors = getattr(config, "DISCOVERY_DOLLAR_VOLUME_FLOORS", {}) or {}

    tickers = [c.get("symbol", "") for c in candidates if c.get("symbol")]
    ticker_to_candidate = {c.get("symbol", ""): c for c in candidates}

    # --- Feature Store: cache-first approach ---
    store = FeatureStore()
    store.load()

    # Check which tickers need a refresh
    stale_tickers = store.get_stale_tickers(tickers, max_age_hours=20)
    fresh_tickers = store.get_fresh_tickers(tickers, max_age_hours=20)

    logger.info("Feature store: %d fresh, %d stale of %d total",
                len(fresh_tickers), len(stale_tickers), len(tickers))

    # Build sector map for relative strength
    sector_map = {c.get("symbol", ""): c.get("sector", "Unknown") for c in candidates}

    # Compute factors only for stale tickers
    if stale_tickers:
        if progress_callback:
            progress_callback(
                f"Computing batch factors for {len(stale_tickers)} tickers...",
                0, len(stale_tickers),
            )
        new_factors = compute_batch_factors(
            stale_tickers,
            batch_size=80,
            sector_map=sector_map,
            progress_callback=progress_callback,
        )
        store.put_batch(new_factors)
        store.save()
        logger.info("Feature store updated: +%d tickers", len(new_factors))

    # Build momentum data from feature store
    all_momentum = {}
    for ticker in tickers:
        feat = store.get(ticker)
        if feat is None:
            continue
        # Convert feature store format to momentum metrics format
        all_momentum[ticker] = {
            "ret_90d": feat.get("ret_90d", 0),
            "ret_30d": feat.get("ret_30d", 0),
            "ret_10d": feat.get("ret_10d", 0),
            "volume_ratio": feat.get("volume_ratio", 1.0),
            "avg_volume": feat.get("avg_volume_20d", 0),
            "avg_dollar_volume": feat.get("avg_dollar_volume", 0),
            "pct_from_high": feat.get("pct_from_high_252d", 0),
            "above_sma50": feat.get("above_sma50", False),
            "last_price": feat.get("last_price", 0),
            "returns": np.array(feat.get("returns_60d", [])),
            # Extra fields from feature store (for downstream use)
            "_beta": feat.get("beta_90d", 1.0),
            "_vol_20d": feat.get("vol_20d", 0),
            "_above_sma200": feat.get("above_sma200", False),
        }

    logger.info("Momentum data available for %d / %d tickers", len(all_momentum), len(tickers))

    # Score and rank — multi-lens approach
    scored = []
    for ticker, m in all_momentum.items():
        cand = ticker_to_candidate.get(ticker)
        if not cand:
            continue

        # Dollar-volume gate (region-aware). FMP-sourced US names are exempt
        # because the FMP screener already enforces its own liquidity floor
        # upstream. For everything else, prefer $-volume over raw shares so
        # expensive high-quality names aren't over-filtered and cheap
        # low-quality names can't slip through on share count alone.
        if cand.get("_source") != "fmp":
            floor = _dollar_volume_floor(cand, dollar_floors)
            dv = m.get("avg_dollar_volume") or 0
            if floor > 0 and dv > 0:
                if dv < floor:
                    continue
            else:
                # Missing dollar-volume data → fall back to raw share floor.
                if m["avg_volume"] < min_avg_vol:
                    continue

        scored.append((ticker, m))

    if not scored:
        return [], {}

    # Percentile-rank all factors across the full universe
    ret_90d_vals = np.array([m["ret_90d"] for _, m in scored])
    ret_30d_vals = np.array([m["ret_30d"] for _, m in scored])
    ret_10d_vals = np.array([m["ret_10d"] for _, m in scored])
    ret_12m1m_vals = np.array([
        m.get("ret_12m1m", 0.60 * m["ret_90d"] + 0.40 * m["ret_30d"])
        for _, m in scored
    ])
    vol_ratio_vals = np.array([m["volume_ratio"] for _, m in scored])
    from_high_vals = np.array([m["pct_from_high"] for _, m in scored])
    # Value lens factors
    vol_20d_vals = np.array([m.get("_vol_20d", 0.3) for _, m in scored])
    # Quality lens: above SMA200 + low vol + near high
    above_sma200_vals = np.array([1.0 if m.get("_above_sma200") else 0.0 for _, m in scored])

    def percentile_rank(arr):
        """Rank values as percentiles (0 to 1)."""
        if len(arr) == 0:
            return arr
        from scipy.stats import rankdata
        return rankdata(arr, method="average") / len(arr)


    try:
        rank_90d = percentile_rank(ret_90d_vals)
        rank_30d = percentile_rank(ret_30d_vals)
        rank_10d = percentile_rank(ret_10d_vals)
        rank_12m1m = percentile_rank(ret_12m1m_vals)
        rank_vol = percentile_rank(vol_ratio_vals)
        rank_high = percentile_rank(from_high_vals)
        # Inverse ranks: low vol = high quality; low pct_from_high = value (far from top)
        rank_low_vol = percentile_rank(-vol_20d_vals)  # lower vol = higher rank
    except ImportError:
        def simple_rank(arr):
            order = np.argsort(np.argsort(arr))
            return order / max(len(arr) - 1, 1)
        rank_90d = simple_rank(ret_90d_vals)
        rank_30d = simple_rank(ret_30d_vals)
        rank_10d = simple_rank(ret_10d_vals)
        rank_12m1m = simple_rank(ret_12m1m_vals)
        rank_vol = simple_rank(vol_ratio_vals)
        rank_high = simple_rank(from_high_vals)
        rank_low_vol = simple_rank(-vol_20d_vals)

    # --- Cross-Sectional Normalization: Relative Strength within Sectors ---
    sector_returns: dict[str, list[tuple[int, float]]] = {}
    for i, (ticker, m) in enumerate(scored):
        sector = ticker_to_candidate.get(ticker, {}).get("sector", "Unknown")
        sector_returns.setdefault(sector, []).append((i, m["ret_90d"]))

    sector_medians: dict[str, float] = {}
    for sector, entries in sector_returns.items():
        returns = [r for _, r in entries]
        sector_medians[sector] = float(np.median(returns)) if returns else 0

    relative_strength = np.zeros(len(scored))
    for i, (ticker, m) in enumerate(scored):
        sector = ticker_to_candidate.get(ticker, {}).get("sector", "Unknown")
        relative_strength[i] = m["ret_90d"] - sector_medians.get(sector, 0)

    rank_relative = percentile_rank(relative_strength)

    # PIT/factor sleeves: attach cheap, cached fundamental/risk metrics before
    # any expensive per-ticker analysis.  This gives high-quality defensive or
    # value names a reserved route through the funnel even when price momentum
    # is not strong enough to win a pure trend screen.
    if getattr(config, "DISCOVERY_STAGE2_FACTOR_SLEEVES_ENABLED", True):
        for i, (ticker, m) in enumerate(scored):
            cand = ticker_to_candidate.get(ticker, {})
            cand["_relative_strength"] = float(relative_strength[i])
            try:
                cand.update(_compute_stage2_factor_metrics(ticker, cand, m))
            except Exception as exc:
                logger.debug("Stage 2 factor metrics failed for %s: %s", ticker, exc)

    # --- Lens 1: Momentum (breakout leaders, trend followers) ---
    # Long-horizon trend proxy: uses true 12-1 momentum when available from
    # the direct-price path, otherwise falls back to the cached long-trend proxy.
    momentum_scores = (
        0.20 * rank_12m1m
        + 0.20 * rank_90d
        + 0.15 * rank_relative
        + 0.15 * rank_30d
        + 0.10 * rank_10d
        + 0.10 * rank_vol
        + 0.10 * rank_high
    )

    # --- Lens 2: Value (beaten-down with improving trend) ---
    # Catches turnarounds: poor 90d return BUT improving 30d/10d + above SMA200
    rank_low_90d = percentile_rank(-ret_90d_vals)  # lower 90d = higher value rank
    value_scores = (
        0.30 * rank_low_90d          # Beaten-down (cheap)
        + 0.25 * rank_30d            # Improving recently
        + 0.20 * rank_10d            # Very recent uptick
        + 0.15 * above_sma200_vals   # Still above long-term trend (not broken)
        + 0.10 * rank_low_vol        # Lower vol = less risky turnaround
    )

    # --- Lens 3: Quality (steady compounders — Novy-Marx 2013 price proxy) ---
    # True fundamental quality is computed in Stage 5b. At the cheap price-only
    # prescreen we use observable proxies: low volatility, persistent long-term
    # trend (above SMA-200), near 52-week high, and return stability (proximity
    # to median 90d return indicates steady trajectory, not erratic).
    median_ret_90d = float(np.median(ret_90d_vals)) if len(ret_90d_vals) > 0 else 0
    ret_stability = percentile_rank(
        -np.abs(ret_90d_vals - median_ret_90d)  # closer to median = more stable = higher rank
    )
    quality_scores = (
        0.35 * rank_low_vol           # Low vol = quality proxy (Novy-Marx 2013)
        + 0.25 * above_sma200_vals    # Persistent long-term trend
        + 0.25 * rank_high            # Near 52w high = steady compounder
        + 0.15 * ret_stability        # Low dispersion of returns
    )

    composite_scores = 0.60 * momentum_scores + 0.40 * value_scores

    # --- Multi-Lens Selection: factor sleeves + price lenses ---
    selected_indices: set[int] = set()
    momentum_selected: set[int] = set()
    value_selected: set[int] = set()
    quality_selected: set[int] = set()
    composite_selected: set[int] = set()
    challenge_selected: set[int] = set()
    factor_sleeve_selected: dict[int, str] = {}
    factor_counts = {"quality": 0, "value": 0, "low_risk": 0, "pead": 0, "defensive_industry": 0}

    def _add_idx(idx: int, bucket: set[int] | None = None, sleeve: str | None = None) -> bool:
        if len(selected_indices) >= top_n:
            return False
        idx = int(idx)
        if idx in selected_indices:
            return False
        selected_indices.add(idx)
        if bucket is not None:
            bucket.add(idx)
        if sleeve:
            factor_sleeve_selected[idx] = sleeve
            factor_counts[sleeve] = factor_counts.get(sleeve, 0) + 1
        return True

    # Defensive sub-industry lane.  Low-volatility, liquid defensive names are
    # often crowded out by a pure global rank, even when they are strong inside
    # their actual economic peer group (e.g. healthcare plans vs pharma).
    if getattr(config, "DISCOVERY_STAGE2_DEFENSIVE_INDUSTRY_LANE_ENABLED", True):
        defensive_budget = int(round(top_n * float(getattr(config, "DISCOVERY_STAGE2_DEFENSIVE_INDUSTRY_RESERVE_PCT", 0.12))))
        defensive_sectors = set(getattr(config, "DISCOVERY_STAGE2_DEFENSIVE_SECTORS", ()))
        defensive_max_per_group = max(1, int(getattr(config, "DISCOVERY_STAGE2_DEFENSIVE_INDUSTRY_MAX_PER_GROUP", 3)))
        defensive_min_score = float(getattr(config, "DISCOVERY_STAGE2_DEFENSIVE_INDUSTRY_MIN_SCORE", 0.55))
        defensive_min_dv = float(getattr(config, "DISCOVERY_STAGE2_DEFENSIVE_INDUSTRY_MIN_DOLLAR_VOLUME", 10_000_000))

        industry_groups: dict[str, list[tuple[float, int]]] = {}
        for idx, (_ticker, _m) in enumerate(scored):
            cand = ticker_to_candidate.get(_ticker, {})
            sector = str(cand.get("sector") or "Unknown").strip()
            if defensive_sectors and sector not in defensive_sectors:
                continue
            score = safe_float(cand.get("_pit_low_risk_score"), default=None)
            if score is None or score < defensive_min_score:
                continue
            if safe_float(_m.get("avg_dollar_volume"), default=0.0) < defensive_min_dv:
                continue
            industry = str(cand.get("industry") or "").strip()
            industry_key = re.sub(r"[^A-Za-z0-9]+", "_", industry.lower()).strip("_")
            group_key = f"{sector}:{industry_key}" if industry_key else sector
            industry_groups.setdefault(group_key, []).append((float(score), idx))

        for entries in industry_groups.values():
            entries.sort(reverse=True)

        # Round-robin by rank within each sub-industry before taking the next
        # stock from the same group.  This preserves breadth while still
        # favouring the strongest low-risk names within every peer set.
        for rank_in_group in range(defensive_max_per_group):
            if factor_counts.get("defensive_industry", 0) >= defensive_budget:
                break
            round_entries: list[tuple[float, int]] = []
            for entries in industry_groups.values():
                if len(entries) > rank_in_group:
                    round_entries.append(entries[rank_in_group])
            for _score, idx in sorted(round_entries, reverse=True):
                if factor_counts.get("defensive_industry", 0) >= defensive_budget:
                    break
                _add_idx(idx, bucket=quality_selected, sleeve="defensive_industry")

    factor_budget = 0
    if getattr(config, "DISCOVERY_STAGE2_FACTOR_SLEEVES_ENABLED", True):
        factor_budget = int(round(top_n * float(getattr(config, "DISCOVERY_STAGE2_FACTOR_RESERVE_PCT", 0.20))))
        min_cov = float(getattr(config, "DISCOVERY_STAGE2_FACTOR_MIN_COVERAGE", 0.25))
        sleeve_specs = [
            ("quality", "_pit_quality_score", getattr(config, "DISCOVERY_STAGE2_FACTOR_QUALITY_PCT", 0.35), quality_selected, "quality"),
            ("value", "_pit_value_score", getattr(config, "DISCOVERY_STAGE2_FACTOR_VALUE_PCT", 0.25), value_selected, "value"),
            ("low_risk", "_pit_low_risk_score", getattr(config, "DISCOVERY_STAGE2_FACTOR_LOW_RISK_PCT", 0.30), quality_selected, "quality"),
            ("pead", "_pit_pead_score", getattr(config, "DISCOVERY_STAGE2_FACTOR_PEAD_PCT", 0.10), momentum_selected, "momentum"),
        ]
        allocated = 0
        for sleeve, key, pct, bucket, _lens in sleeve_specs:
            sleeve_n = int(round(factor_budget * float(pct)))
            if sleeve == sleeve_specs[-1][0]:
                sleeve_n = max(0, factor_budget - allocated)
            allocated += sleeve_n
            if sleeve_n <= 0:
                continue
            scored_sleeve: list[tuple[float, int]] = []
            for idx, (ticker, _m) in enumerate(scored):
                cand = ticker_to_candidate.get(ticker, {})
                score = safe_float(cand.get(key), default=None)
                if score is None:
                    continue
                if sleeve != "low_risk" and safe_float(cand.get("_pit_factor_coverage"), default=0.0) < min_cov:
                    continue
                if sleeve == "low_risk" and score < float(getattr(config, "DISCOVERY_STAGE2_LOW_RISK_MIN_SCORE", 0.55)):
                    continue
                scored_sleeve.append((score, idx))
            for _score, idx in sorted(scored_sleeve, reverse=True):
                if factor_counts.get(sleeve, 0) >= sleeve_n:
                    break
                _add_idx(idx, bucket=bucket, sleeve=sleeve)

    remaining = max(0, top_n - len(selected_indices))
    momentum_pct = float(getattr(config, "DISCOVERY_LENS_MOMENTUM_PCT", 0.50))
    value_pct = float(getattr(config, "DISCOVERY_LENS_VALUE_PCT", 0.25))
    quality_pct = float(getattr(config, "DISCOVERY_LENS_QUALITY_PCT", 0.30))
    total_pct = max(momentum_pct + value_pct + quality_pct + float(getattr(config, "DISCOVERY_LENS_COMPOSITE_PCT", 0.10)), 1e-9)
    n_momentum = int(remaining * momentum_pct / total_pct)
    n_value = int(remaining * value_pct / total_pct)
    n_quality = int(remaining * quality_pct / total_pct)
    n_composite = max(0, remaining - n_momentum - n_value - n_quality)

    def _add_from_order(scores: np.ndarray, limit: int, bucket: set[int]) -> int:
        added = 0
        for idx in np.argsort(scores)[::-1]:
            if added >= limit or len(selected_indices) >= top_n:
                break
            if _add_idx(int(idx), bucket=bucket):
                added += 1
        return added

    momentum_added = _add_from_order(momentum_scores, n_momentum, momentum_selected)
    value_added = _add_from_order(value_scores, n_value, value_selected)
    quality_added = _add_from_order(quality_scores, n_quality, quality_selected)
    composite_added = _add_from_order(composite_scores, n_composite, composite_selected)

    # Fill any leftovers by composite rank so the stage always returns top_n
    # when enough liquid, price-covered candidates exist.
    if len(selected_indices) < top_n:
        _add_from_order(composite_scores, top_n - len(selected_indices), composite_selected)

    challenge_added = 0
    challenge_order = _challenge_ticker_order()
    challenge_reserve_n = max(0, int(getattr(config, "DISCOVERY_CHALLENGE_RESERVE_N", 0)))
    if challenge_order and challenge_reserve_n > 0:
        present = sorted(
            (
                (challenge_order.get(str(ticker).upper(), 9999), idx)
                for idx, (ticker, _m) in enumerate(scored)
                if str(ticker).upper() in challenge_order
            ),
            key=lambda item: item[0],
        )
        target_challenge = min(challenge_reserve_n, len(present), top_n)
        protected = {idx for _order, idx in present if idx in selected_indices}
        challenge_selected.update(protected)
        for _order, idx in present:
            if len(challenge_selected) >= target_challenge:
                break
            if idx in selected_indices:
                continue
            if len(selected_indices) < top_n:
                selected_indices.add(idx)
            else:
                replacement_idx = None
                replacement_score = float("inf")
                for existing_idx in selected_indices:
                    if existing_idx in protected:
                        continue
                    score = float(composite_scores[existing_idx])
                    if replacement_idx is None or score < replacement_score:
                        replacement_idx = existing_idx
                        replacement_score = score
                if replacement_idx is None:
                    break
                selected_indices.remove(replacement_idx)
                selected_indices.add(idx)
            challenge_selected.add(idx)
            protected.add(idx)
            challenge_added += 1

    logger.info(
        "Multi-lens selection: factor=%s + %d momentum + %d value + %d quality + %d composite + %d challenge = %d total",
        factor_counts,
        momentum_added,
        value_added,
        quality_added,
        composite_added,
        challenge_added,
        len(selected_indices),
    )

    # Build output
    filtered = []
    price_cache = {}

    for idx in selected_indices:
        ticker, m = scored[idx]
        cand = ticker_to_candidate[ticker]
        cand["_momentum_score"] = float(momentum_scores[idx])
        cand["_value_score"] = float(value_scores[idx])
        cand["_quality_score"] = float(quality_scores[idx])
        cand["_relative_strength"] = float(relative_strength[idx])
        cand["_sector_median_90d"] = sector_medians.get(
            cand.get("sector", "Unknown"), 0)
        cand["_stage2_factor_sleeve"] = factor_sleeve_selected.get(idx)
        cand["_entry_lens"] = (
            "challenge" if idx in challenge_selected
            else
            "value" if factor_sleeve_selected.get(idx) == "value"
            else "quality" if factor_sleeve_selected.get(idx) in {"quality", "low_risk", "defensive_industry"}
            else "momentum" if idx in momentum_selected
            else "value" if idx in value_selected
            else "quality" if idx in quality_selected
            else "composite"
        )
        cand["_challenge_reserved"] = idx in challenge_selected
        cand["_ret_90d"] = m["ret_90d"]
        cand["_ret_30d"] = m["ret_30d"]
        cand["_ret_10d"] = m["ret_10d"]
        cand["_volume_ratio"] = m["volume_ratio"]
        cand["_pct_from_high"] = m["pct_from_high"]
        cand["_above_sma50"] = m["above_sma50"]
        cand["_above_sma200"] = m.get("_above_sma200", False)
        cand["_vol_20d"] = m.get("_vol_20d", 0)
        cand["_last_price"] = m["last_price"]
        cand["_avg_volume_20d"] = m.get("avg_volume", 0)
        cand["_avg_dollar_volume"] = m.get("avg_dollar_volume", 0)
        filtered.append(cand)

        if "returns" in m and len(m["returns"]) >= 15:
            price_cache[ticker] = m["returns"]

    return filtered, price_cache


# ---------------------------------------------------------------------------
# Stage 3: Quick Filter (no API calls)
# ---------------------------------------------------------------------------

def _stage_quick_filter(
    candidates: list[dict],
    portfolio_sectors: dict[str, float],
    rejections: list[CandidateRejection],
) -> list[dict]:
    """Apply soft penalties for beta, penny stock, and liquidity.

    v4: Graduated penalties replace hard rejection. Only truly extreme
    values (beta > 4.0, price < $0.10) are rejected outright.
    """
    beta_max = getattr(config, "DISCOVERY_BETA_MAX", 2.5)
    passed = []

    for c in candidates:
        symbol = c.get("symbol", "")
        name = c.get("companyName", symbol)
        exchange = c.get("_exchange_query", "")
        penalty = 0.0

        # --- Beta: soft penalty above 2.0, hard reject above 4.0 ---
        beta = c.get("beta") or c.get("_beta")
        if beta is not None:
            if beta > 4.0:
                rejections.append(CandidateRejection(
                    symbol, name, exchange, "quick_filter",
                    f"Beta extreme ({beta:.2f} > 4.0)",
                ))
                continue
            elif beta > beta_max:
                # Graduated: 0 at 2.5, -0.15 at 4.0
                penalty += -0.15 * (beta - beta_max) / (4.0 - beta_max)
            elif beta > 2.0:
                # Mild drag for elevated beta
                penalty += -0.05 * (beta - 2.0) / (beta_max - 2.0)

        # --- Penny stock: soft penalty below $2, hard reject below $0.10 ---
        price = c.get("price") or c.get("_last_price", 0)
        if price is not None and price > 0:
            if price < 0.10:
                rejections.append(CandidateRejection(
                    symbol, name, exchange, "quick_filter",
                    f"Sub-penny stock (price {price:.2f})",
                ))
                continue
            elif price < 1.0:
                penalty += -0.10  # penny stock drag
            elif price < 2.0:
                penalty += -0.05  # low-price drag

        c["_quick_filter_penalty"] = penalty
        passed.append(c)

    return passed


# ---------------------------------------------------------------------------
# Stage 4: Correlation — soft penalty (no longer hard rejection)
# ---------------------------------------------------------------------------

def _stage_correlation_filter(
    candidates: list[dict],
    existing_tickers: list[str],
    rejections: list[CandidateRejection],
    price_cache: dict,
    progress_callback=None,
) -> list[dict]:
    """Compute correlation penalty for each candidate (soft, not hard filter).

    Instead of rejecting candidates with correlation > 0.70, we now assign
    a penalty that scales with correlation strength:
      - corr <= 0.40: no penalty (uncorrelated)
      - corr  0.40-0.70: mild penalty (0.0 to -0.15)
      - corr  0.70-0.90: significant penalty (-0.15 to -0.30)
      - corr > 0.90: heavy penalty (-0.30+, effectively blocks selection)

    All candidates pass through — the penalty is applied in final ranking.
    """
    if not existing_tickers or not candidates:
        for c in candidates:
            c["_max_correlation"] = 0.0
            c["_correlated_with"] = ""
            c["_correlation_penalty"] = 0.0
        return candidates

    # Get existing holdings' returns from feature store first, download only if missing
    if progress_callback:
        progress_callback("Computing correlation penalties...", 0, 1)

    existing_returns = {}
    store = FeatureStore()
    store.load()

    missing_holdings = []
    for ticker in existing_tickers:
        feat = store.get(ticker)
        # Prefer 90d returns (aligned with holding period); fall back to 60d
        _rets = feat.get("returns_90d") or feat.get("returns_60d", []) if feat else []
        if len(_rets) >= 15:
            existing_returns[ticker] = np.array(_rets)
        else:
            missing_holdings.append(ticker)

    # Only download holdings not in the feature store
    if missing_holdings:
        logger.info("Correlation: %d holdings from cache, downloading %d missing",
                     len(existing_returns), len(missing_holdings))
        for ticker in missing_holdings:
            try:
                data = yf.download(ticker, period="90d", progress=False, auto_adjust=True, timeout=30)
                if data is not None and len(data) > 20:
                    close_data = data["Close"]
                    if isinstance(close_data, pd.DataFrame):
                        close_data = close_data.iloc[:, 0]
                    returns = close_data.pct_change().dropna().values[-90:]
                    if len(returns) >= 15:
                        existing_returns[ticker] = returns
            except Exception:
                continue

    if not existing_returns:
        for c in candidates:
            c["_max_correlation"] = 0.0
            c["_correlated_with"] = ""
            c["_correlation_penalty"] = 0.0
        return candidates

    min_len_existing = min(len(v) for v in existing_returns.values())
    passed = []

    for c in candidates:
        symbol = c.get("symbol", "")

        cand_rets = price_cache.get(symbol)
        if cand_rets is None or len(cand_rets) < 15:
            c["_max_correlation"] = 0.0
            c["_correlated_with"] = ""
            c["_correlation_penalty"] = 0.0
            passed.append(c)
            continue

        max_corr = 0.0
        corr_with = ""

        for ex_ticker, ex_rets in existing_returns.items():
            common_len = min(len(cand_rets), len(ex_rets), min_len_existing)
            if common_len < 15:
                continue
            try:
                corr = np.corrcoef(cand_rets[-common_len:], ex_rets[-common_len:])[0, 1]
                if not np.isnan(corr) and abs(corr) > abs(max_corr):
                    max_corr = corr
                    corr_with = ex_ticker
            except Exception:
                continue

        # Soft penalty — scales with correlation strength
        abs_corr = abs(max_corr)
        if abs_corr <= 0.40:
            penalty = 0.0
        elif abs_corr <= 0.70:
            # Linear ramp: 0.0 at 0.40 → -0.15 at 0.70
            penalty = -0.15 * (abs_corr - 0.40) / 0.30
        elif abs_corr <= 0.90:
            # Steeper ramp: -0.15 at 0.70 → -0.30 at 0.90
            penalty = -0.15 - 0.15 * (abs_corr - 0.70) / 0.20
        else:
            # Very high correlation — strong penalty
            penalty = -0.30 - 0.20 * (abs_corr - 0.90) / 0.10

        c["_max_correlation"] = max_corr
        c["_correlated_with"] = corr_with
        c["_correlation_penalty"] = round(penalty, 4)
        passed.append(c)

    return passed


# ---------------------------------------------------------------------------
# Stage 5: Quick Rank (momentum + fundamentals blend)
# ---------------------------------------------------------------------------

def _lightweight_technical_score(ticker: str, price_data: dict) -> float:
    """Fast technical score using cached momentum data — no API calls.

    Lens-aware: quality-lens candidates use a blend that rewards low
    volatility and persistent trend instead of raw momentum, preventing
    the double momentum-filter that kills steady compounders.

    Returns a score in [0, 1].
    """
    momentum = price_data.get("_momentum_score", 0.5)
    above_sma = 0.6 if price_data.get("_above_sma50") else 0.3
    above_sma200 = 0.6 if price_data.get("_above_sma200") else 0.3
    pct_high = price_data.get("_pct_from_high", 0.8)
    # Invert vol_20d: lower vol → higher score (capped at [0, 1])
    vol_20d = price_data.get("_vol_20d", 0.25)
    low_vol_score = max(0.0, min(1.0, (0.50 - vol_20d) / 0.40)) if vol_20d else 0.5

    entry_lens = price_data.get("_entry_lens", "momentum")

    if entry_lens == "quality":
        # Quality blend: lower momentum dependency, reward stability
        return (
            0.25 * momentum
            + 0.25 * above_sma200
            + 0.25 * low_vol_score
            + 0.15 * pct_high
            + 0.10 * above_sma
        )
    elif entry_lens == "value":
        # Value blend: moderate momentum, heavier on SMA recovery signal
        return (
            0.40 * momentum
            + 0.25 * above_sma
            + 0.15 * above_sma200
            + 0.20 * pct_high
        )
    else:
        # Momentum / composite: original blend
        return 0.60 * momentum + 0.20 * above_sma + 0.20 * pct_high


def _cheap_ready_score(candidate: dict) -> float:
    """Fast setup-quality proxy before expensive Stage 6 entry/stop sizing.

    It is intentionally soft: high-alpha names can still pass as watchlist
    ideas, but candidates with a cleaner current entry setup get a tiebreak.
    """
    score = 0.35
    if candidate.get("_above_sma50"):
        score += 0.18
    if candidate.get("_above_sma200"):
        score += 0.18

    rel = safe_float(candidate.get("_relative_strength"), default=0.0)
    score += float(np.clip(rel * 4.0, -0.10, 0.12))

    pct_high = safe_float(candidate.get("_pct_from_high"), default=None)
    if pct_high is not None and pct_high > 0:
        # Sweet spot: close enough to trend leadership, not vertical.
        if 0.88 <= pct_high <= 0.98:
            score += 0.16
        elif 0.78 <= pct_high < 0.88:
            score += 0.08
        elif pct_high > 1.02:
            score -= 0.22
        elif pct_high >= 0.985:
            score -= 0.18

    ret30 = safe_float(candidate.get("_ret_30d"), default=None)
    if ret30 is not None:
        if ret30 >= 0.18:
            score -= 0.20
        elif ret30 >= 0.12:
            score -= 0.08

    vol = safe_float(candidate.get("_vol_20d"), default=None)
    if vol is not None:
        if vol <= 0.30:
            score += 0.12
        elif vol >= 0.60:
            score -= 0.12

    corr_penalty = safe_float(candidate.get("_correlation_penalty"), default=0.0)
    if corr_penalty < 0:
        score += max(-0.15, corr_penalty)

    if safe_float(candidate.get("_avg_dollar_volume"), default=0.0) >= 5_000_000:
        score += 0.06

    return float(np.clip(score, 0.0, 1.0))


def _stage5_value_support(candidate: dict) -> tuple[bool, list[str]]:
    """Cheap value/growth support available before Stage 6."""
    reasons: list[str] = []
    ev_ebit = safe_float(candidate.get("_ev_ebit"), default=None)
    fcf_yield = safe_float(candidate.get("_fcf_yield"), default=None)
    revenue_growth = safe_float(candidate.get("_revenue_growth"), default=None)
    pe = safe_float(candidate.get("_pe_ratio"), default=None)
    peg = safe_float(candidate.get("_peg_ratio"), default=None)

    support = False
    if ev_ebit is not None and 0 < ev_ebit <= 25:
        support = True
        reasons.append("EV/EBIT support")
    if fcf_yield is not None and fcf_yield >= 0.03:
        support = True
        reasons.append("FCF yield support")
    if revenue_growth is not None and revenue_growth >= 0.15:
        support = True
        reasons.append("growth support")
    if pe is not None and 0 < pe <= 30:
        support = True
        reasons.append("P/E support")
    if peg is not None and 0 < peg <= 1.2:
        support = True
        reasons.append("PEG support")
    return support, reasons


def _gate_aware_ready_lane_score(candidate: dict) -> float:
    """Estimate whether a Stage 5 candidate can become ready-to-buy today.

    This is deliberately stricter than ``_cheap_ready_score``. It uses only
    cached/PIT-safe fields, but tries to mirror the later Ready Strong Buy
    contract: acceptable entry setup, enough quality/value evidence, and no
    obvious stretch/valuation flags.
    """
    if _candidate_exclusion_reason(candidate):
        return 0.0

    score = 0.30 * _cheap_ready_score(candidate)

    pct_high = safe_float(candidate.get("_pct_from_high"), default=None)
    ret30 = safe_float(candidate.get("_ret_30d"), default=None)
    ret90 = safe_float(candidate.get("_ret_90d"), default=None)
    vol = safe_float(candidate.get("_vol_20d"), default=None)
    if candidate.get("_above_sma50") and candidate.get("_above_sma200"):
        score += 0.08
    if pct_high is not None:
        if 0.80 <= pct_high <= 0.975:
            score += 0.16
        elif pct_high >= 0.99:
            score -= 0.22
    if ret30 is not None:
        if -0.05 <= ret30 <= 0.12:
            score += 0.12
        elif ret30 >= 0.18:
            score -= 0.18
    if ret90 is not None and ret30 is not None and ret90 > 0 and ret30 < ret90 * 0.75:
        score += 0.05
    if vol is not None:
        if vol <= 0.35:
            score += 0.05
        elif vol >= 0.60:
            score -= 0.08

    f_score = safe_float(candidate.get("_f_score"), default=None)
    f_cov = normalize_f_score_coverage(candidate.get("_f_score_coverage"))
    gpa = safe_float(candidate.get("_gpa"), default=None)
    pit_quality = safe_float(candidate.get("_pit_quality_score"), default=None)
    if is_f_score_actionable(f_score, f_cov, config_module=config):
        score += 0.12 if f_score >= 5 else -0.12
    elif f_score is None:
        score -= 0.04
    if gpa is not None:
        if gpa >= 0.15:
            score += 0.14
        elif gpa >= 0.08:
            score += 0.06
        else:
            score -= 0.08
    elif pit_quality is None:
        score -= 0.04
    if pit_quality is not None and pit_quality > 0:
        score += min(0.08, 0.08 * pit_quality)

    value_support, _ = _stage5_value_support(candidate)
    if value_support:
        score += 0.14
    else:
        score -= 0.08

    ev_ebit = safe_float(candidate.get("_ev_ebit"), default=None)
    pe = safe_float(candidate.get("_pe_ratio"), default=None)
    if ev_ebit is not None and ev_ebit > 50:
        score -= 0.15
    elif ev_ebit is not None and ev_ebit > 30:
        score -= 0.08
    if pe is not None and pe > 60:
        score -= 0.15
    elif pe is not None and pe > 35:
        score -= 0.08

    if safe_float(candidate.get("_avg_dollar_volume"), default=0.0) >= 5_000_000:
        score += 0.04
    return float(np.clip(score, 0.0, 1.0))


def _ready_lane_missing_fields(candidate: dict) -> list[str]:
    missing: list[str] = []
    if not is_f_score_actionable(candidate.get("_f_score"), candidate.get("_f_score_coverage"), config_module=config):
        missing.append("f_score")
    if candidate.get("_gpa") is None:
        missing.append("gpa")
    value_support, _ = _stage5_value_support(candidate)
    if not value_support:
        missing.append("value_support")
    return missing


def _stage_quick_rank(
    candidates: list[dict],
    top_n: int,
    progress_callback=None,
) -> list[dict]:
    """Three-stage ranking: lightweight → medium-cost fundamentals → top N.

    Stage 5a: Score all candidates with fast technical + momentum (no API calls).
              Keep top DISCOVERY_TOP_N_LIGHTWEIGHT (default 150).
    Stage 5b: Medium-cost tier — yfinance .info fundamentals (cached per session).
              P/E, market cap, earnings growth, ROE from yfinance + FMP where available.
    Stage 5c: Multi-lens selection of top N for full deep analysis.
    """
    from utils.data_fetch import get_cached_ticker_info, get_ticker_info

    lightweight_n = getattr(config, "DISCOVERY_TOP_N_LIGHTWEIGHT", 150)

    # --- Stage 5a: Lightweight cache-only ranking (no API calls) ---
    if getattr(config, "DISCOVERY_SLEEVES_ENABLED", True):
        try:
            from engine.sleeves import compute_sleeve_scores
            sleeve_scores = compute_sleeve_scores(candidates)
            for i, c in enumerate(candidates):
                if progress_callback and i % 20 == 0:
                    progress_callback(
                        f"Sleeve ranking... ({i}/{len(candidates)})",
                        i, len(candidates),
                    )
                sym = str(c.get("symbol", "")).upper()
                result = sleeve_scores.get(sym, {})
                per_sleeve = result.get("per_sleeve", {}) or {}
                c["_lightweight_score"] = safe_float(result.get("composite"), default=0.5)
                c["_sleeve_composite"] = safe_float(result.get("composite"), default=None)
                c["_sleeve_breakdown"] = per_sleeve
                c["_sleeve_coverage"] = result.get("coverage", {})
                c["_low_coverage_sleeves"] = result.get("low_coverage_sleeves", [])
                c["sleeve_momentum"] = safe_float(per_sleeve.get("momentum"), default=None)
                c["sleeve_quality"] = safe_float(per_sleeve.get("quality"), default=None)
                c["sleeve_value"] = safe_float(per_sleeve.get("value"), default=None)
                c["sleeve_low_risk"] = safe_float(per_sleeve.get("low_risk"), default=None)
                c["sleeve_pead"] = safe_float(per_sleeve.get("pead"), default=None)
                c["sleeve_ready"] = safe_float(
                    per_sleeve.get("ready", per_sleeve.get("ready_to_buy")),
                    default=None,
                )
        except Exception as exc:
            logger.warning("Sleeve Stage 5a ranking failed; falling back to technical score: %s", exc)
            for c in candidates:
                c["_lightweight_score"] = _lightweight_technical_score(c.get("symbol", ""), c)
    else:
        for i, c in enumerate(candidates):
            if progress_callback and i % 20 == 0:
                progress_callback(
                    f"Lightweight ranking... ({i}/{len(candidates)})",
                    i, len(candidates),
                )
            c["_lightweight_score"] = _lightweight_technical_score(c.get("symbol", ""), c)

    # Include quick filter penalty in lightweight score
    for c in candidates:
        c["_lightweight_score"] = c.get("_lightweight_score", 0) + c.get("_quick_filter_penalty", 0)

    lightweight_pool = list(candidates)
    candidates = _adaptive_promote_candidates(
        candidates,
        target_n=lightweight_n,
        score_key="_lightweight_score",
        preselect_pct=float(getattr(config, "DISCOVERY_LIGHTWEIGHT_PRESELECT_PCT", 0.55)),
        region_floor=int(getattr(config, "DISCOVERY_LIGHTWEIGHT_REGION_FLOOR", 0)),
        sector_floor=int(getattr(config, "DISCOVERY_LIGHTWEIGHT_SECTOR_FLOOR", 0)),
        liquidity_floor=int(getattr(config, "DISCOVERY_LIGHTWEIGHT_LIQUIDITY_FLOOR", 0)),
        lens_floor=int(getattr(config, "DISCOVERY_LIGHTWEIGHT_LENS_FLOOR", 0)),
        sector_max=getattr(config, "DISCOVERY_SLEEVE_SECTOR_MAX", None),
    )
    candidates = _promote_challenge_reserve(
        lightweight_pool,
        candidates,
        target_n=lightweight_n,
        reserve_n=int(getattr(config, "DISCOVERY_CHALLENGE_RESERVE_N", 0)),
        score_key="_lightweight_score",
    )

    if progress_callback:
        progress_callback(f"Medium-cost fundamentals for {len(candidates)}...", 0, len(candidates))

    # --- Stage 5b: Medium-Cost Tier — yfinance .info + FMP fundamentals ---
    # yfinance .info is cached per session (get_ticker_info has _info_cache).
    # This gives us P/E, earnings growth, ROE, market cap for ALL tickers (not just US).
    #
    # Parallelised with ThreadPoolExecutor (4 workers) to cut wall-clock time
    # by ~4x vs serial. Each .info call takes 1-3s network-bound; 4 concurrent
    # calls saturate the connection without triggering yfinance rate limits.

    import concurrent.futures

    def _sf(v):
        """Safe float conversion for yfinance .info values."""
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _prefetch_info(symbol: str) -> tuple[str, dict, str]:
        """Fetch yfinance .info for a single ticker, preferring session cache."""
        cached = get_cached_ticker_info(symbol)
        if cached:
            return (symbol, cached, "cache")
        try:
            timeout = int(getattr(config, "DISCOVERY_INFO_TIMEOUT_SECONDS", 8))
            info = get_ticker_info(symbol, timeout=timeout, allow_network=True)
            return (symbol, info, "network" if info else "empty")
        except Exception:
            return (symbol, {}, "error")

    def _has_stage5b_min_info(info: dict) -> bool:
        if not info:
            return False
        useful_keys = (
            "grossProfits", "grossProfit", "totalAssets", "freeCashflow",
            "trailingPE", "forwardPE", "returnOnEquity", "revenueGrowth",
            "earningsGrowth", "ebit", "ebitda", "enterpriseValue",
        )
        return any(info.get(k) is not None for k in useful_keys)

    def _merge_info(base: dict, incoming: dict) -> dict:
        merged = dict(base or {})
        for key, value in (incoming or {}).items():
            if value is None:
                continue
            if merged.get(key) in (None, ""):
                merged[key] = value
        return merged

    # Pre-fetch yfinance .info with a provider-health circuit breaker.  Yahoo
    # occasionally returns broad 401/crumb failures; when that happens, prefer
    # cache/price/FMP features and keep discovery moving.
    _INFO_WORKERS = max(1, int(getattr(config, "DISCOVERY_INFO_WORKERS", 4)))
    _INFO_PREFETCH_TIMEOUT = float(getattr(config, "DISCOVERY_INFO_PREFETCH_TIMEOUT_SECONDS", 150))
    _INFO_CIRCUIT_MIN = max(1, int(getattr(config, "DISCOVERY_INFO_CIRCUIT_MIN_SAMPLE", 50)))
    _INFO_EMPTY_RATE_CIRCUIT = float(getattr(config, "DISCOVERY_INFO_EMPTY_RATE_CIRCUIT", 0.85))
    symbols = [c.get("symbol", "") for c in candidates]
    info_map: dict[str, dict] = {}
    pit_prefill_count = 0
    fmp_prefill_count = 0

    if getattr(config, "DISCOVERY_PIT_FIRST_STAGE5B", True):
        for c in candidates:
            sym = c.get("symbol", "")
            if not sym:
                continue
            latest, prior, _rd = _pit_latest_and_prior(sym)
            pit_info = _pit_snapshot_to_info(sym, c, latest, prior)
            if pit_info:
                info_map[sym] = _merge_info(info_map.get(sym, {}), pit_info)
                pit_prefill_count += 1

            fmp_info = {
                "marketCap": c.get("_market_cap", c.get("marketCap")),
                "beta": c.get("beta"),
                "sector": c.get("sector"),
                "industry": c.get("industry"),
            }
            fmp_info = {k: v for k, v in fmp_info.items() if v not in (None, "")}
            if fmp_info:
                info_map[sym] = _merge_info(info_map.get(sym, {}), fmp_info)
                fmp_prefill_count += 1

    symbols_for_yf = [
        sym for sym in symbols
        if getattr(config, "DISCOVERY_YFINANCE_INFO_FALLBACK", True)
        and not _has_stage5b_min_info(info_map.get(sym, {}))
    ]
    yahoo_metadata_policy = str(
        getattr(config, "DISCOVERY_YAHOO_METADATA_POLICY", "cache_only") or "cache_only"
    ).lower()
    yahoo_live_metadata_enabled = yahoo_metadata_policy in {"live", "live_fallback", "network"}

    symbol_iter = iter(symbols_for_yf)
    pending: dict[concurrent.futures.Future, str] = {}
    done_count = 0
    empty_count = 0
    cache_count = 0
    network_count = 0
    skipped_count = 0
    _prefetch_start = time.time()

    def _submit_until_full(pool: concurrent.futures.ThreadPoolExecutor) -> None:
        while len(pending) < _INFO_WORKERS:
            try:
                sym = next(symbol_iter)
            except StopIteration:
                return
            pending[pool.submit(_prefetch_info, sym)] = sym

    if not yahoo_live_metadata_enabled:
        for sym in symbols_for_yf:
            cached = get_cached_ticker_info(sym)
            if cached:
                info_map[sym] = _merge_info(info_map.get(sym, {}), cached)
                cache_count += 1
                done_count += 1
            else:
                skipped_count += 1
        if symbols_for_yf:
            logger.info(
                "Stage 5b Yahoo metadata policy=%s: skipped %d live .info calls; using PIT/FMP/cache only",
                yahoo_metadata_policy,
                skipped_count,
            )
    else:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=_INFO_WORKERS)
        try:
            _submit_until_full(pool)
            while pending:
                remaining = _INFO_PREFETCH_TIMEOUT - (time.time() - _prefetch_start)
                if remaining <= 0:
                    logger.warning(
                        "Stage 5b .info prefetch budget exhausted after %d/%d tickers; using fallback for the rest",
                        done_count,
                        len(symbols_for_yf),
                    )
                    break
                done, _ = concurrent.futures.wait(
                    pending,
                    timeout=min(remaining, max(1.0, float(getattr(config, "DISCOVERY_INFO_TIMEOUT_SECONDS", 8)) + 2.0)),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not done:
                    logger.warning(
                        "Stage 5b .info prefetch stalled after %d/%d tickers; using fallback for the rest",
                        done_count,
                        len(symbols_for_yf),
                    )
                    break

                for future in done:
                    sym = pending.pop(future)
                    try:
                        fetched_sym, info, source = future.result(timeout=0)
                        sym = fetched_sym or sym
                        info_map[sym] = _merge_info(info_map.get(sym, {}), info or {})
                    except Exception:
                        source = "error"
                        info_map[sym] = info_map.get(sym, {})

                    done_count += 1
                    if source == "cache":
                        cache_count += 1
                    elif source == "network":
                        network_count += 1
                    else:
                        empty_count += 1

                    if progress_callback and done_count % 50 == 0:
                        progress_callback(
                            f"Fetching fundamentals... ({done_count}/{len(symbols_for_yf)})",
                            done_count, len(symbols_for_yf),
                        )

                if (
                    done_count >= _INFO_CIRCUIT_MIN
                    and empty_count / max(done_count, 1) >= _INFO_EMPTY_RATE_CIRCUIT
                ):
                    logger.warning(
                        "Stage 5b .info circuit breaker fired: empty_rate=%.0f%% after %d responses; using fallback for remaining tickers",
                        100.0 * empty_count / max(done_count, 1),
                        done_count,
                    )
                    break

                _submit_until_full(pool)
        finally:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)

    for sym in symbols:
        info_map.setdefault(sym, {})

    logger.info(
        "Stage 5b: PIT/FMP prefill pit=%d fmp=%d; Yahoo metadata policy=%s fetched %d/%d fallback tickers across %d candidates (cache=%d, network=%d, empty=%d, skipped=%d)",
        pit_prefill_count,
        fmp_prefill_count,
        yahoo_metadata_policy,
        done_count,
        len(symbols_for_yf),
        len(symbols),
        cache_count,
        network_count,
        empty_count,
        skipped_count,
    )
    try:
        atomic_write_json(
            getattr(config, "DISCOVERY_YAHOO_METADATA_HEALTH_PATH", "feature_cache/yahoo_metadata_health.json"),
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "stage": "stage5b",
                "policy": yahoo_metadata_policy,
                "candidates": len(symbols),
                "fallback_tickers": len(symbols_for_yf),
                "pit_prefill": pit_prefill_count,
                "fmp_prefill": fmp_prefill_count,
                "cache": cache_count,
                "network": network_count,
                "empty": empty_count,
                "skipped": skipped_count,
                "circuit_breaker_would_have_sampled": _INFO_CIRCUIT_MIN,
            },
            indent=2,
        )
    except Exception as _health_e:
        logger.debug("Yahoo metadata health write failed: %s", _health_e)

    scored = []
    _stage5_start = time.time()
    _stage5_timeout = max(0.0, float(getattr(config, "DISCOVERY_STAGE5B_SCORING_TIMEOUT", 0) or 0))
    _stage5_min_scored = max(
        0,
        min(
            len(candidates),
            int(getattr(config, "DISCOVERY_STAGE5B_MIN_SCORED_FOR_TIMEOUT", max(top_n, 1)) or 0),
        ),
    )
    _stage5_deadline = _stage5_start + _stage5_timeout if _stage5_timeout > 0 else None
    _fmp_errors = 0
    _factor_errors = 0
    _fmp_topup_calls = 0
    _fmp_topup_skipped = 0
    _fmp_topup_elapsed = 0.0
    _fmp_topup_max = max(0, int(getattr(config, "DISCOVERY_STAGE5B_FMP_TOPUP_MAX_CALLS", 60)))
    _fmp_topup_time_budget = max(
        0.0,
        float(getattr(config, "DISCOVERY_STAGE5B_FMP_TOPUP_TIME_BUDGET", 0) or 0),
    )
    _live_snapshots: dict[str, dict] = {}
    for i, c in enumerate(candidates):
        if (
            _stage5_deadline is not None
            and i > 0
            and len(scored) >= _stage5_min_scored
            and time.time() >= _stage5_deadline
        ):
            logger.warning(
                "Stage 5b scoring deadline hit after %.0fs; using %d/%d scored candidates",
                time.time() - _stage5_start,
                len(scored),
                len(candidates),
            )
            break

        symbol = c.get("symbol", "")
        momentum = c.get("_momentum_score", 0.5)
        value_score = c.get("_value_score", 0.5)
        quality_score = c.get("_quality_score", 0.0)
        fundamental_bonus = 0.0

        # Progress logging every 100 candidates or every 60s
        if i > 0 and (i % 100 == 0):
            _elapsed = time.time() - _stage5_start
            logger.info("Stage 5b scoring: %d/%d candidates (%.0fs elapsed, fmp_err=%d, factor_err=%d)",
                        i, len(candidates), _elapsed, _fmp_errors, _factor_errors)

        info = info_map.get(symbol, {})
        if info:
            c["_stage5b_info"] = dict(info)
            # PIT history ingest: persist today's fundamentals so future
            # backtests / eval loops can retrieve values actually knowable at
            # a given as_of date (45-day filing lag enforced on reads).
            if getattr(config, "DISCOVERY_STAGE5B_PIT_BATCH_RECORD", True):
                _live_snapshots[symbol] = info
            else:
                try:
                    pit_store.record_live_snapshot(symbol, info)
                except Exception:
                    pass
            try:
                qmj_score, qmj_details = _compute_qmj_quality(info)
            except Exception as _qe:
                logger.debug("QMJ quality failed for %s: %s", symbol, _qe)
                qmj_score, qmj_details = 0.0, {}
            quality_score = qmj_score
            _fill_missing_candidate_field(c, "_quality_score", quality_score)
            _fill_missing_candidate_field(c, "_quality_score_fundamental", qmj_score)
            _fill_missing_candidate_field(c, "_gross_profitability", qmj_details.get("gross_profitability"))
            _fill_missing_candidate_field(c, "_fcf_to_assets", qmj_details.get("fcf_to_assets"))
            _fill_missing_candidate_field(c, "_earnings_stability", qmj_details.get("earnings_stability"))
            _fill_missing_candidate_field(c, "_eps_growth_variance_5y", qmj_details.get("eps_growth_variance_5y"))
            fundamental_bonus += 0.12 * qmj_score

            pe = _sf(info.get("trailingPE")) or _sf(info.get("forwardPE"))
            if pe is not None and pe > 0:
                if pe < 12:
                    fundamental_bonus += 0.10
                elif pe < 20:
                    fundamental_bonus += 0.05
                elif pe > 60:
                    fundamental_bonus -= 0.05
                _fill_missing_candidate_field(c, "_pe_ratio", pe)

            eg = _sf(info.get("earningsGrowth"))
            if eg is not None:
                if eg > 0.20:
                    fundamental_bonus += 0.10
                elif eg > 0.05:
                    fundamental_bonus += 0.05
                elif eg < -0.10:
                    fundamental_bonus -= 0.08

            roe = _sf(info.get("returnOnEquity"))
            if roe is not None:
                if roe > 0.20:
                    fundamental_bonus += 0.08
                elif roe > 0.10:
                    fundamental_bonus += 0.04
                elif roe < 0:
                    fundamental_bonus -= 0.08

            mcap = _sf(info.get("marketCap")) or 0
            if 2e9 < mcap < 50e9:
                fundamental_bonus += 0.03
            elif mcap < 300e6:
                fundamental_bonus -= 0.05
            c["_market_cap"] = mcap or c.get("marketCap", 0)

            rev_growth = _sf(info.get("revenueGrowth"))
            if rev_growth is not None:
                if rev_growth > 0.15:
                    fundamental_bonus += 0.10
                elif rev_growth > 0.05:
                    fundamental_bonus += 0.05
                elif rev_growth > 0:
                    fundamental_bonus += 0.02
                c["_revenue_growth"] = rev_growth

            # PEG ratio (Lynch, 1989). Applied for ALL tickers via yfinance —
            # previously this bonus only fired for US tickers through FMP,
            # which gave US names a structural scoring edge. Now symmetric.
            peg = _sf(info.get("trailingPegRatio")) or _sf(info.get("pegRatio"))
            if peg is not None and 0 < peg < 1.0:
                fundamental_bonus += 0.08
                c["_peg_ratio"] = peg
            elif peg is not None and peg > 0:
                c["_peg_ratio"] = peg

            # Extended institutional factors (Cooper 2008, Sloan 1996, Bhandari 1988)
            try:
                ext_factors = compute_all_extended_factors(info, c)
                for fk, fv in ext_factors.items():
                    c[f"_{fk}"] = fv
            except Exception as _ef:
                _factor_errors += 1
                if _factor_errors <= 3:
                    logger.warning("Extended factors failed for %s: %s", symbol, _ef)
                ext_factors = {}

            # Extended factor bonus: reward stocks scoring well on investment+accruals+leverage
            ext_scores = [v for v in (
                ext_factors.get("investment_factor_score"),
                ext_factors.get("accruals_factor_score"),
                ext_factors.get("leverage_factor_score"),
            ) if v is not None]
            if ext_scores:
                fundamental_bonus += 0.08 * float(np.mean(ext_scores))

            c["_yf_sector"] = info.get("sector", "")

        # FMP fallback (US only): fills PEG and revenue-growth gaps when
        # yfinance returned neither. Yfinance coverage of large US names is
        # patchier for PEG than for earnings/revenue, so this is a targeted
        # top-up rather than a US-specific bonus. Non-US tickers already
        # earned the same signals from yfinance above (symmetric path).
        if c.get("_source") == "fmp":
            has_yf_peg = "_peg_ratio" in c
            has_yf_rev_growth = "_revenue_growth" in c
            if not (has_yf_peg and has_yf_rev_growth):
                if _fmp_topup_calls >= _fmp_topup_max:
                    _fmp_topup_skipped += 1
                elif _fmp_topup_time_budget > 0 and _fmp_topup_elapsed >= _fmp_topup_time_budget:
                    _fmp_topup_skipped += 1
                elif _stage5_deadline is not None and (time.time() + 20.0) >= _stage5_deadline:
                    _fmp_topup_skipped += 1
                else:
                    _fmp_topup_calls += 1
                    try:
                        _fmp_call_start = time.time()
                        metrics = get_key_metrics(symbol, period="annual", limit=2)
                        _fmp_topup_elapsed += time.time() - _fmp_call_start
                        if metrics and isinstance(metrics, list) and metrics:
                            m = metrics[0]
                            if not has_yf_rev_growth:
                                rev_growth_fmp = (
                                    m.get("revenuePerShareGrowth")
                                    or m.get("revenueGrowth")
                                )
                                if rev_growth_fmp is not None:
                                    if rev_growth_fmp > 0.15:
                                        fundamental_bonus += 0.10
                                    elif rev_growth_fmp > 0.05:
                                        fundamental_bonus += 0.05
                                    elif rev_growth_fmp > 0:
                                        fundamental_bonus += 0.02
                                    c["_revenue_growth"] = rev_growth_fmp
                            if not has_yf_peg:
                                peg_fmp = m.get("pegRatio")
                                if peg_fmp is not None and 0 < peg_fmp < 1.0:
                                    fundamental_bonus += 0.08
                                    c["_peg_ratio"] = peg_fmp
                                elif peg_fmp is not None and peg_fmp > 0:
                                    c["_peg_ratio"] = peg_fmp
                    except Exception as _fmp_e:
                        _fmp_topup_elapsed += time.time() - locals().get("_fmp_call_start", time.time())
                        _fmp_errors += 1
                        if _fmp_errors <= 5:
                            logger.debug("FMP metrics failed for %s: %s", symbol, _fmp_e)

        # Blend: multi-lens score weighted by entry lens
        entry_lens = c.get("_entry_lens", "momentum")
        if entry_lens == "value":
            combined = (
                0.30 * momentum
                + 0.25 * value_score
                + 0.10 * (0.5 + 0.5 * quality_score)
                + 0.15 * c.get("_lightweight_score", 0.5)
                + 0.20 * (0.5 + fundamental_bonus)
            )
        elif entry_lens == "quality":
            combined = (
                0.20 * momentum
                + 0.10 * value_score
                + 0.30 * (0.5 + 0.5 * quality_score)
                + 0.15 * c.get("_lightweight_score", 0.5)
                + 0.25 * (0.5 + fundamental_bonus)
            )
        else:
            combined = (
                0.45 * momentum
                + 0.05 * value_score
                + 0.10 * (0.5 + 0.5 * quality_score)
                + 0.15 * c.get("_lightweight_score", 0.5)
                + 0.25 * (0.5 + fundamental_bonus)
            )

        ready_score = _cheap_ready_score(c)
        c["_cheap_ready_score"] = ready_score
        ready_lane_score = _gate_aware_ready_lane_score(c)
        c["_ready_lane_score"] = ready_lane_score
        c["_ready_lane_missing_fields"] = _ready_lane_missing_fields(c)
        c["_cheap_quality_score"] = _cheap_quality_score(c)
        if getattr(config, "DISCOVERY_STAGE5_READY_BOOST_ENABLED", True):
            strength = float(getattr(config, "DISCOVERY_STAGE5_READY_BOOST_STRENGTH", 0.30))
            combined *= 1.0 + strength * (ready_score - 0.5)
            combined *= 1.0 + 0.20 * (ready_lane_score - 0.5)

        c["_quick_score"] = combined
        c["_fundamental_bonus"] = fundamental_bonus
        scored.append(c)

    if _live_snapshots:
        try:
            _pit_start = time.time()
            updated = pit_store.record_live_snapshots(_live_snapshots)
            logger.info(
                "Stage 5b PIT live snapshots recorded: %d/%d in %.1fs",
                updated,
                len(_live_snapshots),
                time.time() - _pit_start,
            )
        except Exception as _pit_e:
            logger.debug("Stage 5b PIT batch snapshot failed: %s", _pit_e)

    scored.sort(key=lambda x: x.get("_quick_score", 0), reverse=True)

    _stage5_elapsed = time.time() - _stage5_start
    logger.info("Stage 5b scoring complete: %d candidates in %.1fs (fmp_calls=%d, fmp_skipped=%d, fmp_time=%.1fs, fmp_err=%d, factor_err=%d)",
                len(scored), _stage5_elapsed, _fmp_topup_calls, _fmp_topup_skipped, _fmp_topup_elapsed, _fmp_errors, _factor_errors)

    # Record the full Stage 5b panel for ML training (fixes survivorship bias).
    # All candidates here have fundamentals fetched — marginal cost is SQLite INSERTs.
    try:
        from engine.discovery_backtest import record_stage5b_panel
        _panel_start = time.time()
        record_stage5b_panel(scored)
        logger.info("Stage 5b panel recorded in %.1fs", time.time() - _panel_start)
    except Exception as _e:
        logger.warning("Stage 5b panel recording failed (non-fatal): %s", _e)

    selected = _adaptive_promote_candidates(
        scored,
        target_n=top_n,
        score_key="_quick_score",
        preselect_pct=float(getattr(config, "DISCOVERY_FULL_SCORE_PRESELECT_PCT", 0.55)),
        region_floor=int(getattr(config, "DISCOVERY_FULL_SCORE_REGION_FLOOR", 0)),
        sector_floor=int(getattr(config, "DISCOVERY_FULL_SCORE_SECTOR_FLOOR", 0)),
        liquidity_floor=int(getattr(config, "DISCOVERY_FULL_SCORE_LIQUIDITY_FLOOR", 0)),
        lens_floor=int(getattr(config, "DISCOVERY_FULL_SCORE_LENS_FLOOR", 0)),
        lens_min_counts=getattr(config, "DISCOVERY_FULL_SCORE_LENS_MIN_COUNTS", None),
    )
    if getattr(config, "DISCOVERY_FULL_SCORE_READY_LANE_ENABLED", True):
        ready_lane_reserve = int(round(top_n * float(getattr(config, "DISCOVERY_FULL_SCORE_READY_LANE_PCT", 0.30))))
        selected = _promote_ready_lane(
            scored,
            selected,
            target_n=top_n,
            reserve_n=ready_lane_reserve,
            min_ready_score=float(getattr(config, "DISCOVERY_FULL_SCORE_READY_LANE_MIN_SCORE", 0.58)),
            score_key="_quick_score",
        )
    selected = _promote_ready_reserve(
        scored,
        selected,
        target_n=top_n,
        reserve_n=int(getattr(config, "DISCOVERY_FULL_SCORE_READY_RESERVE", 0)),
        min_ready_score=float(getattr(config, "DISCOVERY_FULL_SCORE_READY_MIN_SCORE", 0.20)),
        score_key="_quick_score",
    )
    if getattr(config, "DISCOVERY_FULL_SCORE_CHEAP_QUALITY_RESERVE_ENABLED", True):
        selected = _promote_cheap_quality_reserve(
            scored,
            selected,
            target_n=top_n,
            reserve_n=int(getattr(config, "DISCOVERY_FULL_SCORE_CHEAP_QUALITY_RESERVE", 20)),
            min_score=float(getattr(config, "DISCOVERY_FULL_SCORE_CHEAP_QUALITY_MIN_SCORE", 0.55)),
            score_key="_quick_score",
        )
    selected = _promote_challenge_reserve(
        scored,
        selected,
        target_n=top_n,
        reserve_n=int(getattr(config, "DISCOVERY_CHALLENGE_RESERVE_N", 0)),
        score_key="_quick_score",
    )

    lens_counts: dict[str, int] = {}
    region_counts: dict[str, int] = {}
    liquidity_counts: dict[str, int] = {}
    for candidate in selected:
        lens = str(candidate.get("_entry_lens", "composite"))
        lens_counts[lens] = lens_counts.get(lens, 0) + 1
        region = _candidate_region(candidate)
        region_counts[region] = region_counts.get(region, 0) + 1
        liquidity = _candidate_liquidity_bucket(candidate)
        liquidity_counts[liquidity] = liquidity_counts.get(liquidity, 0) + 1

    logger.info(
        "Quick rank adaptive selection: %d selected (lenses=%s, regions=%s, liquidity=%s)",
        len(selected),
        lens_counts,
        region_counts,
        liquidity_counts,
    )

    return selected


# ---------------------------------------------------------------------------
# Stage 6: Full Scoring Pipeline
# ---------------------------------------------------------------------------

def _stage_full_scoring(
    candidates: list[dict],
    progress_callback=None,
) -> list[dict]:
    """Run analyse_holding() on each candidate with checkpoint recovery.

    Saves progress every CHECKPOINT_INTERVAL tickers to a checkpoint file.
    On restart, resumes from the last checkpoint instead of re-scoring all
    candidates from scratch.
    """
    import concurrent.futures
    import json
    from pathlib import Path
    from engine.scoring import analyse_holding

    _per_ticker_timeout = getattr(config, "DISCOVERY_PER_TICKER_TIMEOUT", 120)
    _CHECKPOINT_INTERVAL = 20
    _CHECKPOINT_PATH = Path("feature_cache/discovery_checkpoint.json")

    # Clean up orphaned temp file from a prior crash
    _CHECKPOINT_TMP = _CHECKPOINT_PATH.with_suffix(f"{_CHECKPOINT_PATH.suffix}.tmp")
    if _CHECKPOINT_TMP.exists():
        _CHECKPOINT_TMP.unlink(missing_ok=True)

    # --- Checkpoint recovery: reload results from a crashed prior run ---
    results = []
    scored_symbols: set[str] = set()
    start_idx = 0

    try:
        if _CHECKPOINT_PATH.exists():
            # Stale checkpoint detection: if file is > 24 hours old, discard it
            _ckpt_age_s = time.time() - _CHECKPOINT_PATH.stat().st_mtime
            if _ckpt_age_s > 86400:
                logger.info("Checkpoint is %.1f hours old — discarding stale checkpoint",
                            _ckpt_age_s / 3600)
                _CHECKPOINT_PATH.unlink(missing_ok=True)
                raise FileNotFoundError("Stale checkpoint discarded")

            raw = _CHECKPOINT_PATH.read_text(encoding="utf-8")
            if not raw.strip():
                raise ValueError("Empty checkpoint file")
            ckpt = json.loads(raw)
            # Validate checkpoint matches current candidate list
            ckpt_symbols = set(ckpt.get("candidate_symbols", []))
            current_symbols = [c.get("symbol", "") for c in candidates]
            if ckpt_symbols == set(current_symbols):
                results = ckpt.get("results", [])
                scored_symbols = set(r.get("ticker", "") for r in results)
                start_idx = len(results)
                logger.info("Checkpoint recovered: %d/%d already scored (age %.0fmin), resuming",
                            start_idx, len(candidates), _ckpt_age_s / 60)
            else:
                logger.info("Checkpoint stale (candidate list changed: %d vs %d), starting fresh",
                            len(ckpt_symbols), len(set(current_symbols)))
                _CHECKPOINT_PATH.unlink(missing_ok=True)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Corrupt checkpoint deleted: %s", e)
        _CHECKPOINT_PATH.unlink(missing_ok=True)
    except Exception as e:
        logger.debug("Checkpoint load failed: %s", e)
        _CHECKPOINT_PATH.unlink(missing_ok=True)

    def _save_checkpoint():
        """Atomic checkpoint save."""
        try:
            _CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "candidate_symbols": [c.get("symbol", "") for c in candidates],
                "results": results,
                "scored_count": len(results),
                "total": len(candidates),
            }
            atomic_write_json(_CHECKPOINT_PATH, payload, separators=(",", ":"))
        except Exception as e:
            logger.debug("Checkpoint save failed: %s", e)

    # --- Parallel deep scoring with 2 workers ---
    # 2 workers doubles throughput on network-bound scoring while staying
    # under yfinance/FMP rate limits. FinBERT (CPU) is GIL-serialised so
    # safe. Each worker has its own per-ticker timeout.
    _SCORING_WORKERS = getattr(config, "DISCOVERY_SCORING_WORKERS", 2)

    # Build work items (skip already-scored from checkpoint)
    work_items = []
    for c in candidates:
        symbol = c.get("symbol", "")
        if symbol in scored_symbols:
            continue
        exchange = c.get("_exchange_query", "")
        currency = _detect_currency(exchange, symbol)
        price = c.get("price") or c.get("_last_price", 0) or 0
        work_items.append({
            "candidate": c,
            "holding": {
                "ticker": symbol,
                "name": c.get("companyName", symbol),
                "avg_buy_price": price,
                "quantity": 1,
                "currency": currency,
                "_yahoo_info": c.get("_stage5b_info") or {},
                "_allow_yahoo_metadata_network": False,
            },
            "currency": currency,
            "exchange": exchange,
        })

    def _attach_score_metadata(item: dict, result: dict) -> dict:
        c = item["candidate"]
        result["_candidate"] = {k: v for k, v in c.items() if k != "_stage5b_info"}
        result["_currency"] = item["currency"]
        result["_exchange"] = item["exchange"]
        result["_country"] = c.get("country", "")
        result["_sector"] = c.get("sector", "")
        result["_industry"] = c.get("industry", "")
        result["_market_cap"] = c.get("_market_cap", c.get("marketCap", 0))
        result["_max_correlation"] = c.get("_max_correlation", 0)
        result["_correlated_with"] = c.get("_correlated_with", "")
        result["_momentum_score"] = c.get("_momentum_score", 0)
        result["_ret_90d"] = c.get("_ret_90d", 0)
        result["_ret_30d"] = c.get("_ret_30d", 0)
        result["_ret_10d"] = c.get("_ret_10d", 0)
        result["_volume_ratio"] = c.get("_volume_ratio", 1.0)
        result["_avg_volume_20d"] = c.get("_avg_volume_20d", 0)
        result["_beta"] = c.get("_beta")
        result["_vol_20d"] = c.get("_vol_20d", 0)
        result["_above_sma50"] = c.get("_above_sma50", False)
        result["_above_sma200"] = c.get("_above_sma200", False)
        result["_correlation_penalty"] = c.get("_correlation_penalty", 0.0)
        result["_entry_lens"] = c.get("_entry_lens", "momentum")
        result["_quick_filter_penalty"] = c.get("_quick_filter_penalty", 0)
        result["_ready_lane_score"] = c.get("_ready_lane_score")
        result["_ready_lane_missing_fields"] = c.get("_ready_lane_missing_fields", [])
        result["_quality_score_fundamental"] = c.get("_quality_score_fundamental")
        result["_gross_profitability"] = c.get("_gross_profitability")
        result["_fcf_to_assets"] = c.get("_fcf_to_assets")
        result["_earnings_stability"] = c.get("_earnings_stability")
        result["_eps_growth_variance_5y"] = c.get("_eps_growth_variance_5y")
        # Stage 5 carries PIT-safe quality/value evidence. Make it
        # authoritative for ML/gate factor inputs so live rows and replay rows
        # use the same as-of definitions; Stage 6 still fills fields with no
        # PIT surrogate.
        _apply_pit_factor_overrides(result, c)
        return result

    def _fallback_scored_item(item: dict, reason: str) -> dict:
        return _attach_score_metadata(item, _build_partial_score_result(item, reason))

    def _score_one(item: dict) -> dict | None:
        """Score a single candidate with timeout. Returns result or None."""
        symbol = item["holding"]["ticker"]
        _t0 = time.time()

        try:
            result = analyse_holding(item["holding"])
            if not isinstance(result, dict):
                raise TypeError(f"analyse_holding returned {type(result).__name__}")
            _dur = time.time() - _t0
            if _dur > 60:
                logger.warning("Slow scoring: %s took %.1fs", symbol, _dur)
            return _attach_score_metadata(item, result)
        except Exception as e:
            _dur = time.time() - _t0
            logger.warning("Failed to score %s after %.1fs: %s: %s",
                           symbol, _dur, type(e).__name__, e)
            return _fallback_scored_item(item, str(e))

    logger.info("Stage 6: scoring %d candidates with %d workers (checkpoint every %d, timeout %ds/ticker)",
                len(work_items), _SCORING_WORKERS, _CHECKPOINT_INTERVAL, _per_ticker_timeout)

    _stage6_start = time.time()
    _timeout_count = 0
    _error_count = 0
    _slow_tickers: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=_SCORING_WORKERS) as pool:
        future_to_item = {
            pool.submit(_score_one, item): item for item in work_items
        }

        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            symbol = item["holding"]["ticker"]
            try:
                result = future.result(timeout=_per_ticker_timeout)
                if result is not None:
                    results.append(result)

                    if progress_callback:
                        progress_callback(
                            f"Full analysis: {symbol} ({len(results)}/{len(candidates)})",
                            len(results), len(candidates),
                        )

                    # Checkpoint every N tickers
                    if len(results) % _CHECKPOINT_INTERVAL == 0:
                        _save_checkpoint()
                        _elapsed = time.time() - _stage6_start
                        _rate = (len(results) - start_idx) / max(_elapsed, 1)
                        _remaining = (len(candidates) - len(results)) / max(_rate, 0.01)
                        logger.info("Checkpoint saved: %d/%d scored (%.0fs elapsed, ~%.0fs remaining, "
                                    "timeouts=%d, errors=%d)",
                                    len(results), len(candidates), _elapsed, _remaining,
                                    _timeout_count, _error_count)
            except concurrent.futures.TimeoutError:
                _timeout_count += 1
                logger.warning("Timeout scoring %s after %ds — using partial fallback (total timeouts: %d)",
                               symbol, _per_ticker_timeout, _timeout_count)
                future.cancel()  # Prevent queued futures from starting
                results.append(_fallback_scored_item(item, f"Timeout after {_per_ticker_timeout}s"))
                _slow_tickers.append(symbol)
            except Exception as e:
                _error_count += 1
                logger.warning("Future failed for %s (errors: %d): %s: %s",
                               symbol, _error_count, type(e).__name__, e)
                results.append(_fallback_scored_item(item, str(e)))

    _stage6_elapsed = time.time() - _stage6_start
    logger.info("Stage 6 complete: %d scored in %.0fs (timeouts=%d, errors=%d, slow=%s)",
                len(results), _stage6_elapsed, _timeout_count, _error_count,
                _slow_tickers[:10] if _slow_tickers else "none")

    # Final checkpoint save before cleanup
    if results:
        _save_checkpoint()

    # Clean up checkpoint on successful completion
    _CHECKPOINT_PATH.unlink(missing_ok=True)

    return results


# ---------------------------------------------------------------------------
# Stage 7: FX Penalty + Portfolio Fit + Final Ranking
# ---------------------------------------------------------------------------

def _stage_final_ranking(
    scored_results: list[dict],
    portfolio_sectors: dict[str, float],
    holdings: list[dict],
) -> list[ScoredCandidate]:
    """Apply FX penalty, compute portfolio fit, and produce final ranking."""
    fx_fee = getattr(config, "FX_FEE_TIER", 0.0075)
    fx_round_trip_pct = fx_fee * 2 * 100
    sector_max = getattr(
        config,
        "DISCOVERY_SECTOR_PCT_CAP",
        getattr(config, "DISCOVERY_SECTOR_CONCENTRATION_MAX", 0.30),
    )
    corr_threshold = getattr(config, "DISCOVERY_CORRELATION_THRESHOLD", 0.70)
    is_momentum = getattr(config, "DISCOVERY_MODE", "balanced") == "momentum_90d"

    # Get scoring weights — prefer adaptive weights from backtest if available
    try:
        from engine.discovery_backtest import get_adaptive_discovery_weights
        adaptive = get_adaptive_discovery_weights()
    except Exception:
        adaptive = None

    if adaptive:
        scoring_weights = {
            "technical": adaptive.get("technical", 0.25),
            "fundamental": adaptive.get("fundamental", 0.25),
            "sentiment": adaptive.get("sentiment", 0.25),
            "forecast": adaptive.get("forecast", 0.25),
        }
        logger.info("Using adaptive discovery weights from backtest: %s", scoring_weights)
    elif is_momentum:
        scoring_weights = getattr(config, "MOMENTUM_WEIGHTS", config.WEIGHTS)
    else:
        scoring_weights = dict(config.WEIGHTS)

    try:
        from engine.bayesian_learning import get_bayesian_pillar_weights
        scoring_weights = get_bayesian_pillar_weights(scoring_weights, source="discovery")
    except Exception as e:
        logger.warning("Bayesian discovery weights unavailable: %s", e)

    institutional_priors = {}
    if getattr(config, "INSTITUTIONAL_PRIOR_ENABLED", True):
        try:
            institutional_priors = score_universe(scored_results)
            logger.info("Institutional prior scored %d candidates", len(institutional_priors))
        except Exception as e:
            logger.debug("Institutional prior unavailable: %s", e)

    # --- Cross-sectional (optionally sector-neutral) z-scoring + residualisation ---
    _use_zscore = False
    if (getattr(config, "DISCOVERY_CROSS_SECTIONAL_ZSCORE", False)
            and len(scored_results) >= 5):
        _tech_arr = np.array([r.get("technical_score", 0) or 0 for r in scored_results])
        _fund_arr = np.array([r.get("fundamental_score", 0) or 0 for r in scored_results])
        _sent_arr = np.array([r.get("sentiment_score", 0) or 0 for r in scored_results])
        _fcast_arr = np.array([r.get("forecast_score", 0) or 0 for r in scored_results])

        _sector_neutral = bool(getattr(config, "FACTOR_SECTOR_NEUTRAL", False))
        _sectors = [
            (r.get("sector") or r.get("_yf_sector") or "Unknown")
            for r in scored_results
        ]

        def _zfn(arr):
            if _sector_neutral:
                return _sector_neutral_zscore(arr, _sectors)
            return _cross_sectional_zscore(arr)

        # Decorrelate fundamental & forecast pillars from the technical
        # (momentum-heavy) pillar so a high-momentum name isn't triple-
        # counted (same approach as Fama-MacBeth factor regressions).
        try:
            _fund_arr = _residualise_against(_fund_arr, _tech_arr)
            _fcast_arr = _residualise_against(_fcast_arr, _tech_arr)
        except Exception as _re_e:
            logger.debug("Pillar residualisation failed (%s) — using raw pillars", _re_e)

        for arr, key in [
            (_tech_arr, "_z_technical"), (_fund_arr, "_z_fundamental"),
            (_sent_arr, "_z_sentiment"), (_fcast_arr, "_z_forecast"),
        ]:
            z = _zfn(arr)
            for i, r in enumerate(scored_results):
                r[key] = float(z[i])
        _use_zscore = True
        logger.info(
            "%s z-scoring applied to %d candidates (residualised fund/forecast vs technical)",
            "Sector-neutral" if _sector_neutral else "Cross-sectional",
            len(scored_results),
        )

    # --- Adaptive Pillar Weight Discounting ---
    # When a pillar has near-zero cross-sectional std it provides no ranking
    # information. Discount its weight and redistribute to informative pillars.
    # Academic basis: Grinold & Kahn (2000) — weight ∝ IC; IC ≈ 0 → weight ≈ 0.
    _MIN_USEFUL_STD = 0.05
    if len(scored_results) >= 10:
        _pillar_stds = {
            "technical": float(np.std([r.get("technical_score", 0) or 0 for r in scored_results])),
            "fundamental": float(np.std([r.get("fundamental_score", 0) or 0 for r in scored_results])),
            "sentiment": float(np.std([r.get("sentiment_score", 0) or 0 for r in scored_results])),
            "forecast": float(np.std([r.get("forecast_score", 0) or 0 for r in scored_results])),
        }
        _discounted = {}
        _redistributed = 0.0
        for _pk, _pw in scoring_weights.items():
            _pstd = _pillar_stds.get(_pk, 1.0)
            if _pstd < _MIN_USEFUL_STD:
                _discounted[_pk] = _pw * 0.20  # Keep 20% for action thresholds
                _redistributed += _pw * 0.80
                logger.info("Pillar '%s' discounted: std=%.4f < %.2f, weight %.1f%% -> %.1f%%",
                            _pk, _pstd, _MIN_USEFUL_STD, _pw * 100, _discounted[_pk] * 100)
            else:
                _discounted[_pk] = _pw
        if _redistributed > 0:
            _informative_total = sum(v for k, v in _discounted.items()
                                     if _pillar_stds.get(k, 1.0) >= _MIN_USEFUL_STD)
            if _informative_total > 0:
                for _pk in _discounted:
                    if _pillar_stds.get(_pk, 1.0) >= _MIN_USEFUL_STD:
                        _discounted[_pk] += _redistributed * (_discounted[_pk] / _informative_total)
            # Cap any single pillar to prevent one signal from dominating rankings
            _MAX_PILLAR_W = getattr(config, "DISCOVERY_MAX_PILLAR_WEIGHT", 0.70)
            _excess = 0.0
            _informative_keys = [k for k in _discounted
                                 if _pillar_stds.get(k, 1.0) >= _MIN_USEFUL_STD]
            for _pk in _informative_keys:
                if _discounted[_pk] > _MAX_PILLAR_W:
                    _excess += _discounted[_pk] - _MAX_PILLAR_W
                    _discounted[_pk] = _MAX_PILLAR_W
            if _excess > 0:
                _uncapped = [k for k in _informative_keys if _discounted[k] < _MAX_PILLAR_W]
                if _uncapped:
                    _share = _excess / len(_uncapped)
                    for _pk in _uncapped:
                        _discounted[_pk] = min(_MAX_PILLAR_W, _discounted[_pk] + _share)
            scoring_weights = _discounted
            logger.info("Adaptive weights: %s", {k: f"{v:.1%}" for k, v in scoring_weights.items()})
        else:
            logger.info("All pillars above std threshold — no adaptive discounting needed")

    # --- ML Ranker (stacked ensemble, conservative activation) ---
    _use_ml_ranker = False
    _ml_predict = None
    _ml_shadow_predict = None
    _ml_meta_predict = None
    try:
        if not getattr(config, "ML_RANKER_SHADOW_ONLY", True):
            from engine.ml_ranker import (
                train_model,
                predict_alpha,
                predict_meta_success,
                is_available as ml_available,
                is_promotion_eligible,
                get_diagnostics as ml_diagnostics,
            )
            train_model()
            if ml_available():
                _diag = ml_diagnostics()
                if getattr(config, "ML_RANKER_SURFACE_SHADOW", True):
                    _ml_shadow_predict = predict_alpha
                _ml_meta_predict = predict_meta_success
                if is_promotion_eligible():
                    _use_ml_ranker = True
                    _ml_predict = predict_alpha
                    _ml_meta_predict = predict_meta_success
                    logger.info(
                        "ML ranker active for final ranking (n=%s, IC=%s)",
                        _diag.get("n_samples"),
                        _diag.get("oos_rank_ic"),
                    )
                else:
                    logger.info(
                        "ML ranker trained but promotion gate failed; shadow diagnostics enabled (n=%s)",
                        _diag.get("n_samples"),
                    )
        else:
            logger.info("ML ranker held in shadow mode (config)")
    except Exception as e:
        logger.warning("ML ranker unavailable: %s", e)

    kelly_fractions = {}
    try:
        from engine.discovery_backtest import get_kelly_fractions
        kelly_fractions = get_kelly_fractions(source="discovery")
    except Exception as e:
        logger.debug("Discovery Kelly caps unavailable: %s", e)

    portfolio_value_gbp = 100_000.0
    try:
        from engine.portfolio_optimizer import _get_fx_rate

        total = 0.0
        for h in holdings or []:
            qty = safe_float(h.get("quantity"), default=0.0)
            price = safe_float(h.get("current_price"), default=0.0) or safe_float(h.get("avg_buy_price"), default=0.0)
            currency = h.get("currency", "GBP")
            fx_rate = _get_fx_rate(currency)
            gbx_factor = 0.01 if currency == "GBX" else 1.0
            total += qty * price * gbx_factor * fx_rate
        if total > 0:
            portfolio_value_gbp = total
    except Exception as e:
        logger.debug("Portfolio GBP value fallback used in discovery sizing: %s", e)

    # Factor momentum and network momentum context.
    factor_input_rows: dict[str, dict] = {}
    returns_frame = pd.DataFrame()
    store = FeatureStore()
    try:
        store.load()
        store_rows: dict[str, dict] = {}
        for ticker in store.all_tickers():
            store_rows[ticker] = dict(store.get(ticker) or {})
        returns_columns = {}
        for ticker, row in store_rows.items():
            returns = row.get("returns_90d") or row.get("returns_60d") or []
            if isinstance(returns, list) and len(returns) >= 10:
                returns_columns[ticker] = pd.Series(returns[-63:])
        if returns_columns:
            returns_frame = pd.DataFrame(returns_columns)
    except Exception as e:
        logger.debug("Factor momentum base universe unavailable: %s", e)

    for r in scored_results:
        ticker = r.get("ticker", "")
        factor_scores = compute_factor_scores_from_result(r)
        base_row = dict(store.get(ticker) or {})
        base_row.update({k: v for k, v in factor_scores.items() if v is not None})
        base_row.setdefault("returns_90d", base_row.get("returns_60d", []))
        factor_input_rows[ticker] = base_row

    min_factor_universe = int(getattr(config, "FACTOR_MOMENTUM_MIN_UNIVERSE", 10))
    if len(factor_input_rows) < min_factor_universe:
        for ticker in store.all_tickers():
            if ticker in factor_input_rows:
                continue
            factor_input_rows[ticker] = dict(store.get(ticker) or {})
            if len(factor_input_rows) >= min_factor_universe:
                break

    try:
        factor_returns = compute_factor_returns(factor_input_rows)
    except Exception as e:
        logger.debug("Factor momentum computation failed: %s", e)
        factor_returns = load_cached_factor_returns()

    factor_tilt_features = get_factor_tilt_features(factor_returns)
    supply_chain_mapping = load_optional_supply_chain_mapping()
    alpha_center = float(np.mean([r.get("aggregate_score", 0) or 0 for r in scored_results])) if scored_results else 0.0

    # Multi-macro regime for conditional factor timing (Arnott et al. 2021)
    macro_regime = None
    try:
        from engine.regime import get_multi_macro_regime
        macro_regime = get_multi_macro_regime()
    except Exception as e:
        logger.debug("Multi-macro regime fetch failed, trying legacy: %s", e)
        try:
            from utils.data_fetch import get_macro_regime_signals
            macro_regime = get_macro_regime_signals()
        except Exception:
            pass

    run_regime_label = _percentile_regime_label(macro_regime)
    _ip_alpha_weight, _ip_rank_weight = _institutional_prior_weights()
    candidates = []

    for r in scored_results:
        currency = r.get("_currency", "USD")
        exchange = r.get("_exchange", "")
        ticker = r.get("ticker", "")
        sector = r.get("_sector") or r.get("sector", "")
        industry = r.get("_industry") or r.get("industry", "")
        prior = institutional_priors.get(str(ticker).upper(), neutral_prior())
        r["regime"] = run_regime_label
        _candidate_source = r.get("_candidate") or {}
        _sleeve_breakdown = (
            r.get("_sleeve_breakdown")
            or _candidate_source.get("_sleeve_breakdown")
            or {}
        )
        _sleeve_composite = safe_float(
            r.get("_sleeve_composite", _candidate_source.get("_sleeve_composite")),
            default=None,
        )
        _sleeve_momentum = safe_float(
            r.get("sleeve_momentum", _candidate_source.get("sleeve_momentum", _sleeve_breakdown.get("momentum"))),
            default=None,
        )
        _sleeve_quality = safe_float(
            r.get("sleeve_quality", _candidate_source.get("sleeve_quality", _sleeve_breakdown.get("quality"))),
            default=None,
        )
        _sleeve_value = safe_float(
            r.get("sleeve_value", _candidate_source.get("sleeve_value", _sleeve_breakdown.get("value"))),
            default=None,
        )
        _sleeve_low_risk = safe_float(
            r.get("sleeve_low_risk", _candidate_source.get("sleeve_low_risk", _sleeve_breakdown.get("low_risk"))),
            default=None,
        )
        _sleeve_pead = safe_float(
            r.get("sleeve_pead", _candidate_source.get("sleeve_pead", _sleeve_breakdown.get("pead"))),
            default=None,
        )
        _sleeve_ready = safe_float(
            r.get(
                "sleeve_ready",
                _candidate_source.get(
                    "sleeve_ready",
                    _sleeve_breakdown.get("ready", _sleeve_breakdown.get("ready_to_buy")),
                ),
            ),
            default=None,
        )
        r["sleeve_momentum"] = _sleeve_momentum
        r["sleeve_quality"] = _sleeve_quality
        r["sleeve_value"] = _sleeve_value
        r["sleeve_low_risk"] = _sleeve_low_risk
        r["sleeve_pead"] = _sleeve_pead
        r["sleeve_ready"] = _sleeve_ready

        # --- FX Penalty ---
        fx_applied = False
        fx_penalty_pct = 0.0
        forecast_score = r.get("forecast_score", 0)

        if not _is_gbp_denominated(currency):
            fx_applied = True
            fx_penalty_pct = fx_round_trip_pct
            raw_pct = r.get("forecast_pct_change", 0) or 0
            adjusted_pct = raw_pct - fx_penalty_pct
            scale = getattr(config, "FORECAST_SCORE_SCALE", 10.0)
            forecast_score = max(-1.0, min(1.0, adjusted_pct / scale))

        # Recompute aggregate with mode-appropriate weights
        try:
            from engine.regime import get_regime_adjusted_weights
            weights = get_regime_adjusted_weights(scoring_weights)
        except Exception:
            weights = dict(scoring_weights)

        # Scale sentiment contribution by confidence — low-confidence sentiment
        # (few articles, single source) gets less weight in the aggregate.
        sent_confidence = r.get("sentiment_confidence", 1.0) or 1.0
        confidence_floor = getattr(config, "CONFIDENCE_FLOOR", 0.60)
        raw_sent = r.get("sentiment_score", 0) or 0
        sent_effective = raw_sent * max(confidence_floor, sent_confidence)

        if _use_zscore:
            # Use cross-sectionally normalized scores for ranking
            z_sent = r.get("_z_sentiment", raw_sent)
            z_sent_eff = z_sent * max(confidence_floor, sent_confidence)
            adjusted_aggregate = (
                r.get("_z_technical", r.get("technical_score", 0)) * weights.get("technical", 0.30)
                + r.get("_z_fundamental", r.get("fundamental_score", 0)) * weights.get("fundamental", 0.20)
                + z_sent_eff * weights.get("sentiment", 0.20)
                + r.get("_z_forecast", forecast_score) * weights.get("forecast", 0.30)
            )
        else:
            adjusted_aggregate = (
                r.get("technical_score", 0) * weights.get("technical", 0.30)
                + r.get("fundamental_score", 0) * weights.get("fundamental", 0.20)
                + sent_effective * weights.get("sentiment", 0.20)
                + forecast_score * weights.get("forecast", 0.30)
            )

        # --- Risk Overlay ---
        _risk_overlay_failed = False
        try:
            from engine.risk_overlay import apply_risk_overlay
            overlay = apply_risk_overlay(
                r,
                ticker,
                allow_yahoo_earnings=bool(getattr(config, "DISCOVERY_RISK_OVERLAY_ALLOW_YAHOO_EARNINGS", False)),
            )
        except Exception as e:
            logger.warning("Risk overlay failed for %s: %s", ticker, e)
            from engine.risk_overlay import RiskOverlay
            overlay = RiskOverlay()
            _risk_overlay_failed = True

        # Risk overlay penalties — capped to preserve ranking resolution at the tail
        _total_risk_penalty = overlay.parabolic_penalty

        if overlay.earnings_miss and overlay.earnings_miss_pct is not None:
            miss_severity = min(abs(overlay.earnings_miss_pct) / 100.0, 1.0)
            earnings_miss_penalty = 0.05 + 0.15 * miss_severity  # 0.05 to 0.20
            _total_risk_penalty += earnings_miss_penalty
            logger.info("%s: earnings miss penalty %.3f (miss %.1f%%)",
                        ticker, earnings_miss_penalty, overlay.earnings_miss_pct)

        if overlay.near_52w_high:
            _total_risk_penalty += 0.03  # Small drag — momentum already captured

        _MAX_RISK_PENALTY = getattr(config, "DISCOVERY_MAX_RISK_PENALTY", 0.30)
        _total_risk_penalty = min(_total_risk_penalty, _MAX_RISK_PENALTY)
        adjusted_aggregate -= _total_risk_penalty

        # --- Portfolio Fit Score (alpha / fit / confidence separation) ---
        max_corr = abs(r.get("_max_correlation", 0))
        corr_with = r.get("_correlated_with", "")

        # Use soft correlation penalty from Stage 4 (already computed)
        corr_penalty = r.get("_correlation_penalty", 0.0)

        sector_penalty = 0.0
        current_sector_weight = portfolio_sectors.get(sector, 0)
        if current_sector_weight > sector_max:
            sector_penalty = -0.5
        elif current_sector_weight > max(0.20, sector_max * 0.75):
            sector_penalty = -0.25

        # Fit score: 1.0 = perfect fit, 0.0 = poor fit
        portfolio_fit = max(0.0, min(1.0, 1.0 + corr_penalty + sector_penalty))
        sector_weight_if_added = current_sector_weight + max(r.get("position_weight", 0) or 0.05, 0.05)

        # --- Quality Gate ---
        tech_s = r.get("technical_score", 0) or 0
        fund_s = r.get("fundamental_score", 0) or 0
        sent_s = r.get("sentiment_score", 0) or 0
        fcast_s = forecast_score or 0
        pillars_all_zero = (abs(tech_s) + abs(fund_s) + abs(sent_s) + abs(fcast_s)) < 0.001

        # Data confidence heuristic — measures *data quality*, not price state.
        # Pillar coverage counts how many of the four signal pillars carry
        # non-trivial information (|z| > 0.01) at this candidate.  The SMA50
        # "trend" term was removed in 2026-04 because above/below-SMA is a
        # price-level signal already captured by technical_score — using it
        # as a confidence proxy double-counted momentum and penalised
        # counter-trend entries with otherwise-good data coverage.
        _pillar_coverage = sum(
            1 for s in (tech_s, fund_s, sent_s, fcast_s) if abs(s) > 0.01
        ) / 4.0
        _data_confidence = min(1.0, (
            (0.0 if pillars_all_zero else 0.35)      # any pillar signal at all
            + 0.30 * min(sent_confidence, 1.0)       # sentiment depth/diversity
            + 0.20 * _pillar_coverage                # genuine pillar coverage
            + 0.15 * (1.0 if not overlay.earnings_miss else 0.5)  # earnings clarity
        ))
        _data_confidence *= max(confidence_floor, overlay.confidence_discount)
        # Discount confidence when risk data is missing
        if _risk_overlay_failed:
            _data_confidence = max(confidence_floor, _data_confidence * 0.85)
        else:
            _data_confidence = max(confidence_floor, _data_confidence)

        if r.get("analysis_degraded"):
            _data_confidence = max(confidence_floor, _data_confidence * 0.90)

        # --- Momentum Bonus ---
        momentum_score = r.get("_momentum_score", 0.5)
        factor_scores = compute_factor_scores_from_result(r)
        network_momentum = (
            calculate_network_momentum(ticker, supply_chain_mapping, returns_frame)
            if supply_chain_mapping and not returns_frame.empty
            else None
        )
        qa_sentiment_score = r.get("qa_sentiment_score")

        # --- Final Rank: alpha × confidence + fit_adjustment ---
        # Alpha: the pillar-driven quality signal (optionally blended with ML)
        alpha_pre_vol = (
            adjusted_aggregate
            + _quality_overlay_score(r)
            + factor_tilt_adjustment(factor_scores, factor_returns, macro_regime)
            + _ip_alpha_weight * prior.score
        )
        if network_momentum is not None:
            alpha_pre_vol += float(np.clip(network_momentum / 0.20, -1.0, 1.0)) * 0.05
        if qa_sentiment_score is not None:
            alpha_pre_vol += float(np.clip(qa_sentiment_score, -1.0, 1.0)) * 0.05

        # Volatility-managed alpha scaling (bounded to avoid overshooting on low-vol names)
        _vol_20d = r.get("vol_20d") or r.get("_vol_20d") or 0
        if _vol_20d > 0:
            target_vol = getattr(config, "VOL_MANAGED_TARGET_ANN", 0.20)
            floor = getattr(config, "VOL_MANAGED_FLOOR", 0.50)
            cap = getattr(config, "VOL_MANAGED_CAP", 1.25)
            vol_multiplier = float(np.clip(target_vol / _vol_20d, floor, cap))
            alpha = alpha_pre_vol * vol_multiplier
        else:
            alpha = alpha_pre_vol

        ml_alpha_raw = None
        ml_meta_features = None
        meta_success_prob = None
        if _use_ml_ranker or _ml_shadow_predict is not None or _ml_meta_predict is not None:
            ml_features = {
                **{k: v for k, v in factor_scores.items() if v is not None},
                **factor_tilt_features,
                "regime": run_regime_label,
                "technical_score": tech_s,
                "fundamental_score": fund_s,
                "sentiment_score": sent_s,
                "forecast_score": fcast_s,
                "momentum_score": momentum_score,
                "rsi": r.get("rsi"),
                "adx": r.get("adx"),
                "bb_pct": r.get("bb_pct"),
                "pe_ratio": r.get("pe_ratio"),
                "peg_ratio": r.get("peg_ratio"),
                "revenue_growth": r.get("revenue_growth"),
                "roe": r.get("roe"),
                "short_pct": r.get("short_pct"),
                "quality_score_fundamental": r.get("quality_score_fundamental", r.get("_quality_score_fundamental")),
                "gross_profitability": r.get("gross_profitability", r.get("_gross_profitability")),
                "fcf_to_assets": r.get("fcf_to_assets", r.get("_fcf_to_assets")),
                "earnings_stability": r.get("earnings_stability", r.get("_earnings_stability")),
                "eps_growth_variance_5y": r.get("eps_growth_variance_5y", r.get("_eps_growth_variance_5y")),
                "vix_percentile": r.get("vix_percentile"),
                "vol_20d": _vol_20d,
                "return_10d_prior": r.get("return_10d_prior", r.get("_ret_10d")),
                "return_30d_prior": r.get("return_30d_prior", r.get("_ret_30d")),
                "return_90d_prior": r.get("return_90d_prior", r.get("_ret_90d")),
                "f_score": r.get("f_score"),
                "f_score_coverage": r.get("f_score_coverage"),
                "gpa": r.get("gpa"),
                "gpa_score": r.get("gpa_score"),
                "price_vs_sma200_stretch": _price_vs_sma200_stretch(r),
                "institutional_prior_score": prior.score,
                "institutional_prior_percentile": prior.percentile,
                "institutional_prior_confidence": prior.confidence,
                "turnover_cost_score": factor_scores.get("turnover_cost_score"),
                "network_momentum": network_momentum,
                "qa_sentiment_score": qa_sentiment_score,
                "sleeve_momentum": _sleeve_momentum,
                "sleeve_quality": _sleeve_quality,
                "sleeve_value": _sleeve_value,
                "sleeve_low_risk": _sleeve_low_risk,
                "sleeve_pead": _sleeve_pead,
                "sleeve_ready": _sleeve_ready,
                # PEAD factor (Martineau 2022)
                "pead_factor_score": factor_scores.get("pead_factor_score"),
                "sue_score": factor_scores.get("sue_score"),
                "revision_momentum_3m": factor_scores.get("revision_momentum_3m"),
            }
            ml_meta_features = dict(ml_features)
            _predictor = _ml_shadow_predict or _ml_predict
            ml_alpha = _predictor(ml_features) if _predictor is not None else None
            if ml_alpha is not None:
                ml_alpha_raw = float(ml_alpha)
                if _use_ml_ranker:
                    blend = getattr(config, "ML_RANKER_BLEND_PCT", 0.15)
                    # 1-month target is stored as percentage return in the backtest DB.
                    ml_normalized = max(-1.0, min(1.0, ml_alpha / 12.0))
                    alpha = blend * ml_normalized + (1.0 - blend) * alpha
        r["ml_alpha_raw"] = ml_alpha_raw
        r["ml_shadow_score"] = ml_alpha_raw

        # PEAD is now a first-class factor via compute_pead_factor() and
        # factor_tilt_adjustment() (Martineau 2022).  Legacy overlay retained
        # as fallback only when the factor-based PEAD returns None.
        if getattr(config, "PEAD_ENABLED", True):
            pead_via_factor = factor_scores.get("pead_factor_score")
            if pead_via_factor is None:
                # Fallback to legacy overlay when no factor data available
                alpha += _pead_overlay_score(r, overlay)
        else:
            alpha += _pead_overlay_score(r, overlay)
        adjusted_alpha, effective_confidence = adjust_alpha_for_confidence(
            alpha,
            _data_confidence,
            cross_sectional_mean=alpha_center,
        )
        r["effective_data_confidence"] = effective_confidence

        # Alpha-only rank: pure signal strength ignoring portfolio fit.
        # Shows whether a stock was killed for alpha reasons or only for
        # portfolio-shape reasons (sector cap, correlation, FX).
        if is_momentum:
            alpha_only_rank = 0.57 * adjusted_alpha + 0.43 * momentum_score
        else:
            alpha_only_rank = 0.79 * adjusted_alpha + 0.21 * momentum_score

        # Net-of-cost alpha (Novy-Marx & Velikov 2016; Frazzini, Israel & Moskowitz 2018)
        # Deduct estimated round-trip transaction cost from gross alpha
        _avg_dollar_vol = r.get("_avg_dollar_volume") or r.get("avg_dollar_volume") or 0
        _spread_cost = 0.001 + 0.01 * (_vol_20d if _vol_20d > 0 else 0.20)  # Bid-ask proxy
        _impact_cost = 0.0
        if _avg_dollar_vol > 0:
            # Almgren-Chriss square-root impact: assume ~5% of ADV trade size
            participation = 0.05
            _impact_cost = 0.5 * (_vol_20d if _vol_20d > 0 else 0.20) * (participation ** 0.5)
        _fx_cost = 0.0
        if currency not in ("GBP", "GBp"):
            _fx_cost = getattr(config, "FX_FEE_TIER", 0.0075) * 2  # Round-trip
        _total_cost = _spread_cost + _impact_cost + _fx_cost
        # Annualize the cost for comparison with alpha (which is per-period)
        _cost_annualized = _total_cost * (252 / 63)  # ~4 round-trips/year at 63-day holding

        # Final rank formula (net-of-cost, separated concerns):
        #   adjusted_alpha   → "how good is this stock after confidence shrinkage?"
        #   - cost penalty   → "what does it cost to trade?"
        #   + fit_adjustment → "does it improve the portfolio?"
        #   + momentum_bonus → "is the trend confirming?"
        cost_scale = float(getattr(config, "DISCOVERY_COST_PENALTY_SCALE", 0.15))
        net_alpha = adjusted_alpha - _cost_annualized * cost_scale  # Scale cost to alpha units

        if is_momentum:
            _w_sel, _w_ip, _w_tim, _w_fit = 0.30, min(_ip_rank_weight, 0.10), 0.30, 0.30
        else:
            _w_sel, _w_ip, _w_tim, _w_fit = 0.40, _ip_rank_weight, 0.15, 0.30

        # Split rank components (roadmap item #5) — each is on the same scale
        # as final_rank so they can be averaged, sorted, or shown independently.
        selection_rank = _w_sel * net_alpha
        institutional_prior_rank = _institutional_prior_rank_term(prior, _w_ip)
        timing_rank = _w_tim * momentum_score
        portfolio_fit_rank = _w_fit * portfolio_fit
        final_rank = selection_rank + institutional_prior_rank + timing_rank + portfolio_fit_rank

        if r.get("analysis_degraded"):
            _deg = 0.90
            final_rank *= _deg
            selection_rank *= _deg
            institutional_prior_rank *= _deg
            timing_rank *= _deg
            portfolio_fit_rank *= _deg

        # If pillar analysis completely failed, demote heavily
        if pillars_all_zero:
            _dem = 0.30
            final_rank *= _dem
            selection_rank *= _dem
            institutional_prior_rank *= _dem
            timing_rank *= _dem
            portfolio_fit_rank *= _dem
            logger.info("Quality gate: %s has all-zero pillar scores — rank demoted", ticker)

        # Determine action based on aggregate (pillar-driven, not momentum)
        if pillars_all_zero:
            action = "INSUFFICIENT DATA"
        elif adjusted_aggregate >= config.SCORE_STRONG_BUY_THRESHOLD:
            action = "STRONG BUY"
        elif adjusted_aggregate >= config.SCORE_BUY_THRESHOLD:
            action = "BUY"
        elif adjusted_aggregate >= config.SCORE_KEEP_THRESHOLD:
            action = "NEUTRAL"
        else:
            action = "AVOID"

        # F-score hard downgrade (Piotroski 2000): names with very low F-score
        # and decent coverage are likely accounting-weak — cap the action at
        # NEUTRAL even if the aggregate is bullish.  Gated by config flag.
        if getattr(config, "F_SCORE_GATE_ENABLED", False):
            _fs_raw = r.get("f_score")
            _fs = safe_float(_fs_raw, default=None)
            _fs_cov = normalize_f_score_coverage(r.get("f_score_coverage"))
            if (
                _fs is not None
                and is_f_score_actionable(_fs, _fs_cov, config_module=config)
                and _fs <= 3
                and action in ("STRONG BUY", "BUY")
            ):
                logger.info("F-score gate: %s F=%s cov=%.2f — action capped at NEUTRAL", ticker, _fs, _fs_cov)
                action = "NEUTRAL"

        # --- Discovery gates v2 telemetry + narrow active trap safeguard ---
        _stretch_vs_200 = _price_vs_sma200_stretch(r)
        _gate_v2_status, _gate_v2_reasons = _evaluate_discovery_gates_v2(
            r,
            sector=sector,
            industry=industry,
            stretch=_stretch_vs_200,
        )
        _trap_safeguard_triggered, _trap_safeguard_reason = _evaluate_trap_safeguard(
            r,
            sector=sector,
            industry=industry,
            stretch=_stretch_vs_200,
        )

        _v2_shadow = bool(getattr(config, "DISCOVERY_GATES_V2_SHADOW", True))
        _gate_first_active = (
            bool(getattr(config, "DISCOVERY_GATE_FIRST_ENABLED", True))
            and not _v2_shadow
        )
        _v2_active = (
            bool(getattr(config, "DISCOVERY_GATES_V2_ENABLED", False))
            and not _v2_shadow
        ) or _gate_first_active
        if _v2_active and _gate_v2_status == "REJECT":
            if action in ("STRONG BUY", "BUY", "NEUTRAL"):
                action = "MANUAL REVIEW"
            _v2_mult = float(getattr(config, "DISCOVERY_GATES_V2_REJECT_RANK_MULTIPLIER", 0.20))
            final_rank *= _v2_mult
            selection_rank *= _v2_mult
            institutional_prior_rank *= _v2_mult
            timing_rank *= _v2_mult
            portfolio_fit_rank *= _v2_mult
            logger.info("Discovery V2 gate: %s would be blocked (%s)", ticker, "; ".join(_gate_v2_reasons))

        if _trap_safeguard_triggered:
            if action in ("STRONG BUY", "BUY", "NEUTRAL"):
                action = "MANUAL REVIEW"
            _trap_mult = float(getattr(config, "DISCOVERY_TRAP_SAFEGUARD_RANK_MULTIPLIER", 0.25))
            final_rank *= _trap_mult
            selection_rank *= _trap_mult
            institutional_prior_rank *= _trap_mult
            timing_rank *= _trap_mult
            portfolio_fit_rank *= _trap_mult
            if _trap_safeguard_reason:
                r["why"] = (
                    f"{r.get('why', '')} + {_trap_safeguard_reason}"
                    if r.get("why")
                    else _trap_safeguard_reason
                )
            logger.info("Trap safeguard: %s moved to MANUAL REVIEW (%s)", ticker, _trap_safeguard_reason)

        # --- Trading strategy: entry price, stop-loss, position sizing ---
        _cp = r.get("current_price") or 0
        _atr = r.get("atr")
        _sma50 = r.get("sma_50")
        _sma200 = r.get("sma_200")
        _bb_low = r.get("bb_lower")

        _entry_data = {"entry_price": None, "entry_method": "", "entry_zone": (None, None),
                       "fill_probability": None, "all_levels": {}, "discount_pct": 0}
        _stop_data = {"stop_loss": None, "method": "", "stop_distance_pct": 0,
                      "support_levels": {}, "regime": {}}
        _target_data = {"take_profit": None, "method": ""}
        _size_data = {
            "shares": 0,
            "position_weight": 0,
            "risk_amount": 0,
            "r_r_ratio": None,
            "sizing_method": "",
            "kelly_cap_fraction": None,
        }

        if _cp and _cp > 0:
            from engine.stops import (calculate_entry_strategy,
                                      calculate_stop_loss as _calc_stop,
                                      calculate_take_profit as _calc_tp,
                                      calculate_position_size, _realized_volatility)
            try:
                _, _vol_pct = _realized_volatility(ticker)
                _entry_data = calculate_entry_strategy(
                    _cp, _atr, sma_50=_sma50, bb_lower=_bb_low,
                    vol_percentile=_vol_pct, entry_lens=r.get("_entry_lens", "momentum"))
            except Exception as _e:
                logger.warning("%s: entry strategy failed: %s", ticker, _e)

            try:
                _stop_data = _calc_stop(
                    ticker, _atr, _cp, sma_200=_sma200, sma_50=_sma50,
                    bb_lower=_bb_low)
                _target_data = _calc_tp(
                    ticker, _cp, _stop_data.get("stop_loss"),
                    entry_price=_entry_data.get("entry_price"),
                    entry_lens=r.get("_entry_lens", "momentum"))
            except Exception as _e:
                logger.warning("%s: stop/target calc failed: %s", ticker, _e)

            try:
                if (
                    _entry_data.get("entry_price")
                    and _stop_data.get("stop_loss")
                    and not _trap_safeguard_triggered
                    and not (_v2_active and _gate_v2_status == "REJECT")
                ):
                    from engine.portfolio_optimizer import _get_fx_rate
                    _fx_rate = _get_fx_rate(currency)
                    _gbx_factor = 0.01 if currency == "GBX" else 1.0
                    _entry_gbp = _entry_data["entry_price"] * _gbx_factor * _fx_rate
                    _stop_gbp = _stop_data["stop_loss"] * _gbx_factor * _fx_rate
                    _tp_local = _target_data.get("take_profit")
                    _tp_gbp = _tp_local * _gbx_factor * _fx_rate if _tp_local else None
                    _adv_shares = safe_float(r.get("_avg_volume_20d"))
                    _size_data = calculate_position_size(
                        portfolio_value_gbp, _entry_gbp, _stop_gbp,
                        take_profit=_tp_gbp,
                        risk_per_trade_pct=getattr(config, "POSITION_RISK_BUDGET_PCT", 0.01),
                        kelly_cap_fraction=kelly_fractions.get(action),
                        avg_daily_volume_shares=(_adv_shares or None),
                    )
            except Exception as _e:
                logger.warning("%s: position sizing failed: %s", ticker, _e)

        _candidate_meta = r.get("_candidate") or {}
        _analyst_target_raw = r.get("analyst_target")
        _analyst_target = (
            safe_float(_analyst_target_raw)
            if _analyst_target_raw is not None
            else None
        )
        _analyst_upside_raw = r.get("analyst_upside")
        _analyst_upside = (
            safe_float(_analyst_upside_raw)
            if _analyst_upside_raw is not None
            else None
        )
        _num_analysts = r.get("num_analysts")
        try:
            _num_analysts = int(_num_analysts) if _num_analysts is not None else None
        except (TypeError, ValueError):
            _num_analysts = None
        _insider_buys = int(safe_float(r.get("insider_buys"), default=0))
        _insider_sells = int(safe_float(r.get("insider_sells"), default=0))
        _beta_raw = r.get("_beta")
        _beta_90d = safe_float(_beta_raw) if _beta_raw is not None else None
        _debt_to_equity_raw = r.get("debt_to_equity")
        _debt_to_equity = (
            safe_float(_debt_to_equity_raw)
            if _debt_to_equity_raw is not None
            else None
        )
        _ticker_identity_warning = _compute_ticker_identity_warning(
            ticker=ticker,
            candidate_meta=_candidate_meta,
            result_name=r.get("name", ticker),
        )
        _entry_stance = _derive_entry_stance(
            governance_flag=r.get("governance_flag", False),
            asymmetric_risk_flag=r.get("asymmetric_risk_flag", False),
            earnings_imminent=overlay.earnings_imminent,
            is_parabolic=overlay.is_parabolic,
            analyst_upside=_analyst_upside,
            near_52w_high=overlay.near_52w_high,
            return_30d=safe_float(r.get("_ret_30d", 0)),
            insider_sells=_insider_sells,
            insider_buys=_insider_buys,
            earnings_near=overlay.earnings_near,
        )
        if _trap_safeguard_triggered or (_v2_active and _gate_v2_status == "REJECT"):
            _entry_stance = "Watch Only"

        _effective_gate_status = _gate_v2_status if _v2_active else "PASS"
        _effective_gate_reasons = _gate_v2_reasons if _v2_active else []
        _ready_contract_status, _ready_contract_reasons, _ready_contract_details = _evaluate_ready_strong_buy_contract(
            gate_status=_effective_gate_status,
            trap_triggered=_trap_safeguard_triggered,
            prior=prior,
            entry_stance=_entry_stance,
            entry_price=_entry_data.get("entry_price"),
            stop_loss=_stop_data.get("stop_loss"),
            take_profit=_target_data.get("take_profit"),
            rr_ratio=_size_data.get("r_r_ratio"),
            position_weight=_size_data.get("position_weight"),
            data_confidence=effective_confidence,
            gate_reasons=_effective_gate_reasons,
            return_details=True,
        )
        _ready_contract_core_status = _ready_contract_status
        _ready_contract_core_reasons = list(_ready_contract_reasons or [])
        _strong_buy_eligible = _ready_contract_status == "PASS"
        _strong_buy_blockers = list(_ready_contract_reasons or [])
        if (
            _ml_meta_predict is not None
            and ml_meta_features is not None
            and getattr(config, "ML_RANKER_META_PROBA_FOR_ALL", True)
        ):
            try:
                meta_features = dict(ml_meta_features)
                meta_features.update({
                    "fill_probability": _entry_data.get("fill_probability"),
                    "r_r_ratio": _size_data.get("r_r_ratio"),
                    "strong_buy_eligible": 1 if _strong_buy_eligible else 0,
                })
                meta_success_prob = _ml_meta_predict(meta_features)
            except Exception as exc:
                logger.debug("Meta-label diagnostic prediction failed for %s: %s", ticker, exc)

        if action == "STRONG BUY" and not _strong_buy_eligible:
            action = "MANUAL REVIEW" if (
                _trap_safeguard_triggered or (_v2_active and _gate_v2_status == "REJECT")
            ) else "BUY"
            if _ready_contract_reasons:
                logger.info(
                    "Ready Strong Buy contract: %s demoted (%s)",
                    ticker,
                    "; ".join(_ready_contract_reasons[:4]),
                )

        if (
            action == "STRONG BUY"
            and _ml_meta_predict is not None
            and getattr(config, "META_LABEL_STRONG_BUY_GATE_ENABLED", True)
        ):
            try:
                meta_features = dict(ml_meta_features or {})
                meta_features.update({
                    "fill_probability": _entry_data.get("fill_probability"),
                    "r_r_ratio": _size_data.get("r_r_ratio"),
                    "strong_buy_eligible": 1 if _strong_buy_eligible else 0,
                })
                if meta_success_prob is None:
                    meta_success_prob = _ml_meta_predict(meta_features)
                if str(_ready_contract_core_status or "").upper() == "PASS":
                    # Core-ready candidates need the current batch's core-ready
                    # distribution, which is only complete in the percentile
                    # action pass. Defer this gate to avoid stale-cache vetoes.
                    min_meta_prob = None
                else:
                    try:
                        from engine.auto_tune import get_meta_strong_buy_min_prob
                        min_meta_prob = float(get_meta_strong_buy_min_prob())
                    except Exception:
                        min_meta_prob = float(getattr(config, "META_LABEL_STRONG_BUY_MIN_PROB", 0.60))
                if min_meta_prob is not None and meta_success_prob is not None and meta_success_prob < min_meta_prob:
                    action = "BUY"
                    _strong_buy_eligible = False
                    _ready_contract_status = "FAIL"
                    _ready_contract_reasons = list(_ready_contract_reasons or [])
                    meta_reason = f"Meta-label confidence {meta_success_prob:.0%} below {min_meta_prob:.0%}"
                    _ready_contract_reasons.append(meta_reason)
                    _strong_buy_blockers = list(_ready_contract_reasons)
                    meta_mult = float(getattr(config, "META_LABEL_RANK_MULTIPLIER_ON_FAIL", 0.90))
                    final_rank *= meta_mult
                    selection_rank *= meta_mult
                    institutional_prior_rank *= meta_mult
                    logger.info(
                        "Meta-label gate: %s STRONG BUY demoted (prob=%.3f < %.3f)",
                        ticker,
                        meta_success_prob,
                        min_meta_prob,
                    )
            except Exception as exc:
                logger.debug("Meta-label gate failed for %s: %s", ticker, exc)

        r["meta_prob"] = meta_success_prob
        r["meta_success_prob"] = meta_success_prob
        r["strong_buy_blockers"] = list(_strong_buy_blockers or [])
        r["ready_contract_core_status"] = _ready_contract_core_status
        r["ready_contract_core_reasons"] = list(_ready_contract_core_reasons or [])
        r["ready_contract_score"] = _ready_contract_details.get("score")
        r["ready_contract_soft_passes"] = list(_ready_contract_details.get("soft_passes") or [])
        r["gate_v2_status"] = _gate_v2_status
        r["gate_v2_reasons"] = list(_gate_v2_reasons or [])

        candidates.append(ScoredCandidate(
            ticker=ticker,
            name=r.get("name", ticker),
            exchange=exchange,
            country=r.get("_country", ""),
            sector=sector,
            industry=industry,
            market_cap=r.get("_market_cap", 0),
            currency=currency,
            aggregate_score=round(adjusted_aggregate, 3),
            technical_score=round(r.get("technical_score", 0), 3),
            fundamental_score=round(r.get("fundamental_score", 0), 3),
            sentiment_score=round(r.get("sentiment_score", 0), 3),
            forecast_score=round(forecast_score, 3),
            action=action,
            why=r.get("why", ""),
            fx_penalty_applied=fx_applied,
            fx_penalty_pct=round(fx_penalty_pct, 2),
            max_correlation=round(max_corr, 3),
            correlated_with=corr_with,
            sector_weight_if_added=round(sector_weight_if_added, 3),
            portfolio_fit_score=round(portfolio_fit, 3),
            expected_return_90d=round(r.get("expected_return_90d", 0), 4),
            momentum_score=round(momentum_score, 3),
            return_90d=round(r.get("_ret_90d", 0), 4),
            return_30d=round(r.get("_ret_30d", 0), 4),
            return_10d=round(r.get("_ret_10d", 0), 4),
            volume_ratio=round(r.get("_volume_ratio", 1.0), 2),
            vol_20d=safe_float(r.get("_vol_20d")),
            analyst_target=_analyst_target,
            analyst_upside=_analyst_upside,
            num_analysts=_num_analysts,
            insider_buys=_insider_buys,
            insider_sells=_insider_sells,
            insider_net=r.get("insider_net", "") or "",
            pe_ratio=safe_float(r.get("pe_ratio")),
            peg_ratio=safe_float(r.get("peg_ratio")),
            revenue_growth=safe_float(r.get("revenue_growth")),
            roe=safe_float(r.get("roe")),
            short_pct=safe_float(r.get("short_pct")),
            beta_90d=_beta_90d,
            debt_to_equity=_debt_to_equity,
            entry_stance=_entry_stance,
            ticker_identity_warning=_ticker_identity_warning,
            parabolic_penalty=overlay.parabolic_penalty,
            is_parabolic=overlay.is_parabolic,
            earnings_near=overlay.earnings_near,
            earnings_imminent=overlay.earnings_imminent,
            earnings_days=overlay.earnings_days,
            cap_tier=overlay.cap_tier,
            confidence_discount=overlay.confidence_discount,
            effective_data_confidence=effective_confidence,
            max_weight_scale=overlay.max_weight_scale,
            post_earnings_recent=overlay.post_earnings_recent,
            post_earnings_days=overlay.post_earnings_days,
            earnings_miss=overlay.earnings_miss,
            earnings_miss_pct=overlay.earnings_miss_pct,
            near_52w_high=overlay.near_52w_high,
            pct_from_52w_high=overlay.pct_from_52w_high,
            entry_lens=r.get("_entry_lens", "momentum"),
            entry_price=_entry_data.get("entry_price"),
            entry_method=_entry_data.get("entry_method", ""),
            entry_zone_low=(_entry_data.get("entry_zone") or (None, None))[0],
            entry_zone_high=(_entry_data.get("entry_zone") or (None, None))[1],
            fill_probability=_entry_data.get("fill_probability"),
            stop_loss=_stop_data.get("stop_loss"),
            stop_method=_stop_data.get("method", ""),
            stop_distance_pct=_stop_data.get("stop_distance_pct"),
            take_profit=_target_data.get("take_profit"),
            target_method=_target_data.get("method", ""),
            position_size_shares=_size_data.get("shares", 0),
            position_weight=_size_data.get("position_weight", 0),
            risk_amount=_size_data.get("risk_amount", 0),
            r_r_ratio=_size_data.get("r_r_ratio"),
            sizing_method=_size_data.get("sizing_method", ""),
            kelly_cap_fraction=_size_data.get("kelly_cap_fraction"),
            support_levels=_stop_data.get("support_levels", {}),
            regime_info=_stop_data.get("regime", {}),
            regime=run_regime_label,
            # Dividend safety
            dividend_yield=r.get("dividend_yield"),
            payout_ratio=r.get("payout_ratio"),
            ex_dividend_date=r.get("ex_dividend_date"),
            ex_dividend_days=r.get("ex_dividend_days"),
            five_year_avg_yield=r.get("five_year_avg_yield"),
            # Balance sheet strength
            balance_sheet_grade=r.get("balance_sheet_grade"),
            net_debt_ebitda=r.get("net_debt_ebitda"),
            current_ratio=r.get("current_ratio"),
            cash_to_debt=r.get("cash_to_debt"),
            # Governance red flag
            governance_flag=r.get("governance_flag", False),
            governance_reasons=r.get("governance_reasons", []),
            # Asymmetric / binary outcome flag
            asymmetric_risk_flag=r.get("asymmetric_risk_flag", False),
            asymmetric_risk_reason=r.get("asymmetric_risk_reason"),
            quality_score_fundamental=(
                r.get("quality_score_fundamental")
                if r.get("quality_score_fundamental") is not None
                else r.get("_quality_score_fundamental")
            ),
            gross_profitability=(
                r.get("_gross_profitability")
                if r.get("_gross_profitability") is not None
                else r.get("gross_profitability")
            ),
            fcf_to_assets=(
                r.get("_fcf_to_assets")
                if r.get("_fcf_to_assets") is not None
                else r.get("fcf_to_assets")
            ),
            fcf_yield=(
                r.get("_fcf_yield")
                if r.get("_fcf_yield") is not None
                else r.get("fcf_yield")
            ),
            earnings_stability=(
                r.get("earnings_stability")
                if r.get("earnings_stability") is not None
                else r.get("_earnings_stability")
            ),
            eps_growth_variance_5y=(
                r.get("eps_growth_variance_5y")
                if r.get("eps_growth_variance_5y") is not None
                else r.get("_eps_growth_variance_5y")
            ),
            qmj_factor_score=factor_scores.get("qmj_factor_score"),
            pead_factor_score=factor_scores.get("pead_factor_score"),
            sue_score=factor_scores.get("sue_score"),
            revision_momentum_3m=factor_scores.get("revision_momentum_3m"),
            bab_factor_score=factor_scores.get("bab_factor_score"),
            turnover_cost_score=factor_scores.get("turnover_cost_score"),
            # Enterprise factor bundle (roadmap items #1, #2)
            enterprise_value=r.get("enterprise_value"),
            ev_ebit=r.get("ev_ebit"),
            ev_ebitda=r.get("ev_ebitda"),
            ebit_yield=r.get("ebit_yield"),
            ev_ebit_score=r.get("ev_ebit_score"),
            gpa=r.get("gpa"),
            gpa_score=r.get("gpa_score"),
            f_score=r.get("f_score"),
            f_score_gate=bool(r.get("f_score_gate", False)),
            f_score_score=r.get("f_score_score"),
            f_score_coverage=normalize_f_score_coverage(r.get("f_score_coverage")),
            sma_200=r.get("sma_200"),
            price_vs_sma200_stretch=_stretch_vs_200,
            # Distress / quality-of-earnings — `None` means "no data, skip the
            # gate" so we MUST preserve None instead of defaulting to 0.0
            # (which would falsely fail every candidate with missing data).
            altman_z=safe_float(r.get("altman_z"), default=None),
            altman_zone=str(r.get("altman_zone") or "unknown"),
            altman_coverage=float(r.get("altman_coverage") or 0.0),
            beneish_m=safe_float(r.get("beneish_m"), default=None),
            accruals_factor_score=safe_float(
                r.get("accruals_factor_score") or r.get("_accruals_factor_score"), default=None
            ),
            investment_factor_score=safe_float(
                r.get("investment_factor_score") or r.get("_investment_factor_score"), default=None
            ),
            op_margin_yoy_delta=safe_float(r.get("op_margin_yoy_delta"), default=None),
            rsi=safe_float(r.get("rsi"), default=None),
            realized_vol_pctile=safe_float(r.get("realized_vol_pctile"), default=None),
            gate_v2_status=_gate_v2_status,
            gate_v2_reasons=_gate_v2_reasons,
            trap_safeguard_triggered=_trap_safeguard_triggered,
            trap_safeguard_reason=_trap_safeguard_reason,
            institutional_prior_score=prior.score,
            institutional_prior_percentile=prior.percentile,
            institutional_prior_confidence=prior.confidence,
            institutional_prior_coverage=prior.coverage,
            institutional_prior_components=prior.components,
            institutional_prior_rank=round(institutional_prior_rank, 3),
            ready_contract_core_status=_ready_contract_core_status,
            ready_contract_core_reasons=_ready_contract_core_reasons,
            ready_contract_status=_ready_contract_status,
            ready_contract_reasons=_ready_contract_reasons,
            ready_contract_score=safe_float(_ready_contract_details.get("score"), default=None),
            ready_contract_soft_passes=list(_ready_contract_details.get("soft_passes") or []),
            strong_buy_blockers=_strong_buy_blockers,
            strong_buy_eligible=_strong_buy_eligible,
            ready_lane_score=safe_float(r.get("_ready_lane_score"), default=None),
            ready_lane_missing_fields=list(r.get("_ready_lane_missing_fields") or []),
            factor_momentum_tilt=factor_tilt_features,
            network_momentum=network_momentum,
            qa_sentiment_score=qa_sentiment_score,
            ml_alpha_raw=ml_alpha_raw,
            ml_shadow_score=ml_alpha_raw,
            meta_success_prob=meta_success_prob,
            sleeve_composite=_sleeve_composite,
            sleeve_momentum=_sleeve_momentum,
            sleeve_quality=_sleeve_quality,
            sleeve_value=_sleeve_value,
            sleeve_low_risk=_sleeve_low_risk,
            sleeve_pead=_sleeve_pead,
            sleeve_ready=_sleeve_ready,
            analysis_degraded=bool(r.get("analysis_degraded", False)),
            analysis_degraded_reason=r.get("analysis_degraded_reason"),
            alpha_rank=round(alpha_only_rank, 3),
            selection_rank=round(selection_rank, 3),
            timing_rank=round(timing_rank, 3),
            portfolio_fit_rank=round(portfolio_fit_rank, 3),
            final_rank=round(final_rank, 3),
        ))

    # --- Zero-centre portfolio_fit contribution to widen ranking spread ---
    # Most candidates have portfolio_fit ≈ 1.0 (uncorrelated), adding a constant
    # +0.30 floor that collapses ranking resolution. Subtract the median fit
    # contribution so the term only differentiates, not inflates.
    if candidates:
        import statistics
        fit_weight = 0.30  # same weight used in final_rank formula
        fit_values = [c.portfolio_fit_score for c in candidates]
        median_fit = statistics.median(fit_values)
        fit_adjustment = fit_weight * median_fit
        for c in candidates:
            c.final_rank = round(c.final_rank - fit_adjustment, 3)
            # portfolio_fit_rank tracks the same adjustment so the split view
            # stays consistent with final_rank downstream.
            c.portfolio_fit_rank = round(c.portfolio_fit_rank - fit_adjustment, 3)

    _assign_percentile_actions(candidates, regime=run_regime_label)

    # Conformal prediction overlay (Vovk-Gammerman-Shafer 2005; Angelopoulos &
    # Bates 2021).  Computes a distribution-free p-value per candidate from the
    # last 365 days of matured 90d outcomes — gives the UI a calibrated
    # confidence reading even when the parametric meta-label is in cold-start.
    if bool(getattr(config, "CONFORMAL_ENABLED", True)):
        try:
            from engine.conformal import compute_conformal_p, compute_k_ic
            k_ic = compute_k_ic()
            cp = compute_conformal_p(candidates, k_ic=k_ic)
            n_sb = sum(
                1 for c in candidates
                if getattr(c, "action", "") == "STRONG BUY"
                and getattr(c, "conformal_p", 1.0) is not None
                and getattr(c, "conformal_p", 1.0) < 0.05
            )
            logger.info(
                "Conformal overlay: n=%d candidates scored, k_ic=%.4f, %d STRONG BUY at p<0.05",
                len(cp), k_ic, n_sb,
            )
        except Exception as e:
            logger.warning("Conformal overlay failed (non-fatal): %s", e)

    candidates.sort(key=lambda x: x.final_rank, reverse=True)

    # --- Diversified Final Selector ---
    # Enforce sector balance without geographic minimums.
    max_per_sector = getattr(config, "DISCOVERY_MAX_PER_SECTOR", 4)
    max_per_industry = int(getattr(config, "DISCOVERY_MAX_PER_INDUSTRY", max_per_sector))
    use_industry_groups = bool(getattr(config, "DISCOVERY_USE_INDUSTRY_DIVERSIFICATION", True))
    sector_pct_cap = getattr(config, "DISCOVERY_SECTOR_PCT_CAP", sector_max)

    group_counts: dict[str, int] = {}
    sector_weights: dict[str, float] = {}
    diversified = []
    deferred = []  # candidates that exceeded sector cap
    selected_weight = 0.0

    for c in candidates:
        sector = c.sector or "Unknown"
        industry = str(getattr(c, "industry", "") or "").strip()
        industry_key = re.sub(r"[^A-Za-z0-9]+", "_", industry.lower()).strip("_")
        group_key = f"{sector}:{industry_key}" if (use_industry_groups and industry_key) else sector
        count = group_counts.get(group_key, 0)
        max_for_group = max_per_industry if (use_industry_groups and industry_key) else max_per_sector
        candidate_weight = float(c.position_weight or 0.05)
        projected_total = selected_weight + candidate_weight
        projected_sector_weight = sector_weights.get(sector, 0.0) + candidate_weight
        projected_sector_ratio = (
            projected_sector_weight / projected_total if projected_total > 0 else 0.0
        )

        if count < max_for_group and (
            len(diversified) < 3 or projected_sector_ratio <= sector_pct_cap
        ):
            diversified.append(c)
            group_counts[group_key] = count + 1
            sector_weights[sector] = projected_sector_weight
            selected_weight = projected_total
        else:
            deferred.append(c)

    # Append remaining deferred candidates at the end (still available for review)
    diversified.extend(deferred)

    return diversified


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_discovery(
    holdings: list[dict],
    risk_data: dict | None = None,
    progress_callback=None,
) -> DiscoveryResult:
    """Run the full discovery pipeline.

    Args:
        holdings: Current portfolio holdings
        risk_data: Portfolio risk data (optional)
        progress_callback: fn(message, current, total) for progress updates

    Returns:
        DiscoveryResult with ranked candidates, rejections, and stats
    """
    start_time = time.time()
    result = DiscoveryResult()
    rejections: list[CandidateRejection] = []

    def _stage_elapsed():
        return f"[{time.time() - start_time:.0f}s total]"

    def _mark_stage(name: str, stage_start: float, **metrics) -> None:
        elapsed = round(time.time() - stage_start, 1)
        total = round(time.time() - start_time, 1)
        payload = {"seconds": elapsed, "total_seconds": total}
        payload.update(metrics)
        result.stage_timings[name] = payload
        logger.info("Discovery timing: %s took %.1fs (total %.1fs) metrics=%s", name, elapsed, total, metrics)

    # Extract portfolio info
    existing_tickers = {h["ticker"] for h in holdings}
    existing_ticker_list = [h["ticker"] for h in holdings]

    portfolio_sectors: dict[str, float] = {}
    if risk_data and risk_data.get("sector_weights"):
        portfolio_sectors = risk_data["sector_weights"]
    else:
        for h in holdings:
            try:
                profile = get_company_profile(h["ticker"])
                if profile and profile.get("sector"):
                    sector = profile["sector"]
                    portfolio_sectors[sector] = portfolio_sectors.get(sector, 0) + (1 / len(holdings))
            except Exception:
                continue

    # Stage 1: Universe Assembly
    if progress_callback:
        progress_callback("Stage 1: Assembling global universe...", 0, 7)

    _stage_start = time.time()
    candidates = _stage_universe_assembly(existing_tickers, progress_callback)
    candidates = _filter_excluded_candidates(
        candidates,
        rejections,
        stage="universe_exclusion",
    )
    result.screened_count = len(candidates)
    _mark_stage("stage1_universe", _stage_start, candidates=len(candidates), rejections=len(rejections))
    logger.info("Stage 1: Assembled %d candidates (FMP US + global universe) %s", len(candidates), _stage_elapsed())

    if not candidates:
        result.error = "No candidates found from any source"
        result.run_time_seconds = time.time() - start_time
        return result

    # Stage 2: Momentum Screen
    if progress_callback:
        progress_callback("Stage 2: Momentum screening...", 1, 7)

    _stage_start = time.time()
    candidates, price_cache = _stage_momentum_screen(candidates, progress_callback)
    result.after_momentum_screen = len(candidates)
    _mark_stage("stage2_momentum", _stage_start, candidates=len(candidates), price_cache=len(price_cache or {}))
    logger.info("Stage 2: %d candidates after momentum screen %s", len(candidates), _stage_elapsed())

    if not candidates:
        result.error = "No candidates passed momentum screen"
        result.run_time_seconds = time.time() - start_time
        return result

    # Stage 3: Quick Filter
    if progress_callback:
        progress_callback("Stage 3: Applying quick filters...", 2, 7)

    _stage_start = time.time()
    candidates = _stage_quick_filter(candidates, portfolio_sectors, rejections)
    result.after_quick_filter = len(candidates)
    _mark_stage("stage3_quick_filter", _stage_start, candidates=len(candidates), rejections=len(rejections))
    logger.info("Stage 3: %d candidates after quick filter %s", len(candidates), _stage_elapsed())

    # Stage 4: Correlation Filter
    if progress_callback:
        progress_callback("Stage 4: Computing correlations...", 3, 7)

    _stage_start = time.time()
    candidates = _stage_correlation_filter(
        candidates, existing_ticker_list, rejections, price_cache, progress_callback,
    )
    result.after_corr_filter = len(candidates)
    _mark_stage("stage4_correlation", _stage_start, candidates=len(candidates))
    logger.info("Stage 4: %d candidates with correlation penalties (soft filter, no rejections) %s", len(candidates), _stage_elapsed())

    if not candidates:
        result.rejections = rejections
        result.error = "All candidates filtered out"
        result.run_time_seconds = time.time() - start_time
        return result

    # Stage 5: Quick Rank
    if progress_callback:
        progress_callback("Stage 5: Ranking candidates...", 4, 7)

    top_n = getattr(config, "DISCOVERY_TOP_N_FULL_SCORE", 30)
    _stage_start = time.time()
    candidates = _stage_quick_rank(candidates, top_n, progress_callback)
    result.after_quick_rank = len(candidates)
    _mark_stage("stage5_quick_rank", _stage_start, candidates=len(candidates), top_n=top_n)
    logger.info("Stage 5: Top %d candidates for full scoring %s", len(candidates), _stage_elapsed())

    # Stage 6: Full Scoring
    if progress_callback:
        progress_callback("Stage 6: Running full analysis...", 5, 7)

    _stage_start = time.time()
    try:
        scored = _stage_full_scoring(candidates, progress_callback)
    except Exception as _s6_err:
        logger.error("Stage 6 FAILED: %s: %s", type(_s6_err).__name__, _s6_err)
        import traceback
        logger.error("Stage 6 traceback:\n%s", traceback.format_exc())
        result.error = f"Stage 6 failed: {_s6_err}"
        result.run_time_seconds = round(time.time() - start_time, 1)
        _mark_stage("stage6_full_scoring", _stage_start, error=str(_s6_err))
        return result
    result.fully_scored = len(scored)
    _mark_stage("stage6_full_scoring", _stage_start, candidates=len(scored))
    logger.info("Stage 6: %d candidates fully scored %s", len(scored), _stage_elapsed())

    # Stage 7: FX + Fit + Final Ranking
    if progress_callback:
        progress_callback("Stage 7: Computing final rankings...", 6, 7)

    _stage_start = time.time()
    try:
        final_candidates = _stage_final_ranking(scored, portfolio_sectors, holdings)
    except Exception as _s7_err:
        logger.error("Stage 7 FAILED: %s: %s", type(_s7_err).__name__, _s7_err)
        import traceback
        logger.error("Stage 7 traceback:\n%s", traceback.format_exc())
        result.error = f"Stage 7 failed: {_s7_err}"
        result.run_time_seconds = round(time.time() - start_time, 1)
        _mark_stage("stage7_final_ranking", _stage_start, error=str(_s7_err))
        return result

    result.candidates = final_candidates
    result.rejections = rejections
    result.fx_penalties_applied = sum(1 for c in final_candidates if c.fx_penalty_applied)
    result.run_time_seconds = round(time.time() - start_time, 1)
    _mark_stage("stage7_final_ranking", _stage_start, candidates=len(final_candidates))

    logger.info("Discovery complete: %d candidates ranked in %.1fs",
                len(final_candidates), result.run_time_seconds)

    return result
