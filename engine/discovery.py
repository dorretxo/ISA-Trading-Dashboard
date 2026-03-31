"""Global Discovery Engine v4 — Multi-Lens + Wider Funnel.

Architecture:
- Feature Store caches batch price factors daily (cheap, runs on full universe)
- Multi-lens prescreen: momentum + value + quality entry paths (not just momentum)
- Medium-cost tier: lightweight fundamentals on 250 before deep analysis on 60
- Soft thresholds: graduated penalties replace hard cuts
- Cross-sectional normalization: scores ranked within sector/region/cap peers
- Region-balanced sampling: proportional representation across US/EU/UK
- Diversified final selector: sector caps and region floors in output

Funnel stages:
1. Universe Assembly → ~2000-3000 candidates (FMP US + yfinance global)
2. Multi-Lens Screen → ~250 (momentum 50% + value 25% + quality 25%)
3. Quick Filter      → ~220 (soft penalties for beta, penny stock, volume)
4. Correlation Filter → soft penalty (no hard rejection)
5. Quick Rank        → top 60 (medium-cost tier: momentum + lightweight fundamentals)
6. Full Scoring      → 60 scored (analyse_holding pipeline)
7. FX + Fit + Diversification → final ranked list with diversity enforcement
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

import config
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


def _quality_overlay_score(result: dict) -> float:
    """Compact QMJ-style quality overlay from already-fetched fundamentals."""
    score = 0.0

    margin = result.get("profit_margin")
    roe = result.get("roe")
    fcf_yield = result.get("fcf_yield")
    debt_to_equity = result.get("debt_to_equity")

    if margin is not None:
        if margin >= 0.15:
            score += 0.05
        elif margin < 0:
            score -= 0.05

    if roe is not None:
        if roe >= 0.15:
            score += 0.05
        elif roe < 0.05:
            score -= 0.03

    if fcf_yield is not None:
        if fcf_yield >= 0.04:
            score += 0.05
        elif fcf_yield < 0:
            score -= 0.05

    if debt_to_equity is not None:
        if debt_to_equity <= 60:
            score += 0.03
        elif debt_to_equity >= 150:
            score -= 0.04

    return max(-0.12, min(0.12, score))


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
    # 90-day expected return (unified target)
    expected_return_90d: float = 0.0
    # Entry-quality context
    analyst_target: float | None = None
    analyst_upside: float | None = None
    num_analysts: int | None = None
    insider_buys: int = 0
    insider_sells: int = 0
    insider_net: str = ""
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
    from utils.data_fetch import get_ticker_info

    info = get_ticker_info(ticker)
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
        or near_52w_high
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
            seen_symbols.add(symbol)
            stock["_exchange_query"] = exchange
            stock["_source"] = "fmp"
            all_candidates.append(stock)

    fmp_count = len(all_candidates)
    logger.info("FMP screener: %d US candidates from %d exchanges", fmp_count, len(exchanges))

    # --- Part B: Global Universe via curated lists ---
    use_global = getattr(config, "DISCOVERY_USE_GLOBAL_UNIVERSE", True)
    if use_global:
        if progress_callback:
            progress_callback("Loading global universe...", len(exchanges), len(exchanges) + 1)

        try:
            from utils.global_universe import get_full_universe, get_universe_for_rotation
            import datetime as _dt
            _today = _dt.datetime.now().weekday()
            _global_meta = {entry.ticker: entry for entry in get_full_universe()}
            global_tickers = get_universe_for_rotation(
                day_of_week=_today,
                exclude_tickers=existing_tickers,
            )

            for ticker in global_tickers:
                if ticker in seen_symbols:
                    continue
                seen_symbols.add(ticker)
                _meta = _global_meta.get(ticker)
                exchange = _meta.exchange if _meta else _detect_exchange(ticker)
                all_candidates.append({
                    "symbol": ticker,
                    "companyName": ticker,  # Will be enriched later
                    "country": _meta.country if _meta else "",
                    "sector": _meta.sector if _meta else "",
                    "_exchange_query": exchange,
                    "_source": "global_universe",
                })

            global_count = len(all_candidates) - fmp_count
            logger.info("Global universe: %d non-US candidates added", global_count)
        except ImportError:
            logger.warning("global_universe module not found, skipping non-US screening")

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
    min_avg_vol = getattr(config, "MOMENTUM_MIN_AVG_VOLUME", 100_000)

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

        if cand.get("_source") != "fmp" and m["avg_volume"] < min_avg_vol:
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

    # --- Lens 3: Quality (steady outperformers, low vol, near highs) ---
    quality_scores = (
        0.25 * rank_relative          # Outperforming sector peers
        + 0.25 * rank_high            # Near 52w high (consistent)
        + 0.20 * rank_low_vol         # Low volatility
        + 0.15 * above_sma200_vals    # Above long-term trend
        + 0.15 * rank_30d             # Positive recent trend
    )

    # --- Multi-Lens Selection: each lens gets a quota ---
    momentum_pct = getattr(config, "DISCOVERY_LENS_MOMENTUM_PCT", 0.50)
    value_pct = getattr(config, "DISCOVERY_LENS_VALUE_PCT", 0.25)
    quality_pct = getattr(config, "DISCOVERY_LENS_QUALITY_PCT", 0.25)

    n_momentum = int(top_n * momentum_pct)
    n_value = int(top_n * value_pct)
    n_quality = top_n - n_momentum - n_value  # remainder

    selected_indices = set()

    # Momentum picks (best momentum scores)
    momentum_order = np.argsort(momentum_scores)[::-1]
    for idx in momentum_order:
        if len(selected_indices) >= n_momentum:
            break
        selected_indices.add(int(idx))

    # Value picks (best value scores, not already selected)
    value_order = np.argsort(value_scores)[::-1]
    value_added = 0
    for idx in value_order:
        if value_added >= n_value:
            break
        if int(idx) not in selected_indices:
            selected_indices.add(int(idx))
            value_added += 1

    # Quality picks (best quality scores, not already selected)
    quality_order = np.argsort(quality_scores)[::-1]
    quality_added = 0
    for idx in quality_order:
        if quality_added >= n_quality:
            break
        if int(idx) not in selected_indices:
            selected_indices.add(int(idx))
            quality_added += 1

    logger.info("Multi-lens selection: %d momentum + %d value + %d quality = %d total",
                n_momentum, value_added, quality_added, len(selected_indices))

    # --- Region-Balanced Sampling ---
    # Ensure each region gets minimum representation
    region_min_pct = getattr(config, "DISCOVERY_REGION_MIN_PCT", 0.15)
    min_per_region = max(1, int(top_n * region_min_pct))

    def _ticker_region(ticker: str) -> str:
        if any(ticker.endswith(s) for s in (".L",)):
            return "UK"
        if any(ticker.endswith(s) for s in (".DE", ".PA", ".MI", ".MC", ".AS", ".BR",
                                             ".LS", ".SW", ".ST", ".CO", ".HE", ".OL")):
            return "EU"
        if any(ticker.endswith(s) for s in (".T", ".HK", ".SI", ".KS", ".AX")):
            return "APAC"
        return "US"

    region_counts: dict[str, int] = {}
    for idx in selected_indices:
        ticker = scored[idx][0]
        region = _ticker_region(ticker)
        region_counts[region] = region_counts.get(region, 0) + 1

    # Top up under-represented regions from their best momentum+value blend
    blended_scores = 0.5 * momentum_scores + 0.3 * value_scores + 0.2 * quality_scores
    for region in ("US", "EU", "UK"):
        current = region_counts.get(region, 0)
        if current < min_per_region:
            needed = min_per_region - current
            # Find best unselected tickers from this region
            region_candidates = [
                (int(i), blended_scores[i])
                for i in range(len(scored))
                if int(i) not in selected_indices and _ticker_region(scored[i][0]) == region
            ]
            region_candidates.sort(key=lambda x: x[1], reverse=True)
            for idx, _ in region_candidates[:needed]:
                selected_indices.add(idx)
            logger.info("Region balance: topped up %s with %d extra candidates",
                        region, min(needed, len(region_candidates)))

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
        cand["_entry_lens"] = (
            "momentum" if idx in set(int(i) for i in momentum_order[:n_momentum])
            else "value" if idx in set(int(i) for i in value_order[:n_value + n_momentum])
            else "quality"
        )
        cand["_ret_90d"] = m["ret_90d"]
        cand["_ret_30d"] = m["ret_30d"]
        cand["_ret_10d"] = m["ret_10d"]
        cand["_volume_ratio"] = m["volume_ratio"]
        cand["_pct_from_high"] = m["pct_from_high"]
        cand["_above_sma50"] = m["above_sma50"]
        cand["_last_price"] = m["last_price"]
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

    Returns a score in [0, 1] based on:
    - Momentum percentile (already computed)
    - Trend strength (above SMA-50)
    - Distance from 52w high (breakout potential)
    """
    momentum = price_data.get("_momentum_score", 0.5)
    above_sma = 0.6 if price_data.get("_above_sma50") else 0.3
    pct_high = price_data.get("_pct_from_high", 0.8)

    # Blend: momentum is primary, trend and breakout confirm
    return 0.60 * momentum + 0.20 * above_sma + 0.20 * pct_high


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
    Stage 5c: Region-balanced selection of top N for full deep analysis.
    """
    from utils.data_fetch import get_ticker_info

    lightweight_n = getattr(config, "DISCOVERY_TOP_N_LIGHTWEIGHT", 150)

    # --- Stage 5a: Lightweight technical ranking (no API calls) ---
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

    candidates.sort(key=lambda x: x.get("_lightweight_score", 0), reverse=True)
    candidates = candidates[:lightweight_n]

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

    def _prefetch_info(symbol: str) -> tuple[str, dict]:
        """Fetch yfinance .info for a single ticker (runs in thread pool)."""
        try:
            return (symbol, get_ticker_info(symbol))
        except Exception:
            return (symbol, {})

    # Pre-fetch all yfinance .info in parallel (4 workers)
    _INFO_WORKERS = getattr(config, "DISCOVERY_INFO_WORKERS", 4)
    symbols = [c.get("symbol", "") for c in candidates]
    info_map: dict[str, dict] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=_INFO_WORKERS) as pool:
        futures = {pool.submit(_prefetch_info, sym): sym for sym in symbols}
        done_count = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                sym, info = future.result(timeout=45)
                info_map[sym] = info
            except Exception:
                info_map[futures[future]] = {}
            done_count += 1
            if progress_callback and done_count % 50 == 0:
                progress_callback(
                    f"Fetching fundamentals... ({done_count}/{len(symbols)})",
                    done_count, len(symbols),
                )

    logger.info("Stage 5b: pre-fetched .info for %d/%d tickers", len(info_map), len(symbols))

    scored = []
    for i, c in enumerate(candidates):
        symbol = c.get("symbol", "")
        momentum = c.get("_momentum_score", 0.5)
        value_score = c.get("_value_score", 0.5)
        quality_score = c.get("_quality_score", 0.5)
        fundamental_bonus = 0.0

        info = info_map.get(symbol, {})
        if info:
            pe = _sf(info.get("trailingPE")) or _sf(info.get("forwardPE"))
            if pe is not None and pe > 0:
                if pe < 12:
                    fundamental_bonus += 0.10
                elif pe < 20:
                    fundamental_bonus += 0.05
                elif pe > 60:
                    fundamental_bonus -= 0.05
                c["_pe_ratio"] = pe

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

            rev_growth = _sf(info.get("revenueGrowth"))
            if rev_growth is not None and rev_growth > 0.15:
                fundamental_bonus += 0.05

            c["_yf_sector"] = info.get("sector", "")

        # FMP fundamentals bonus (US tickers — additional enrichment)
        if c.get("_source") == "fmp":
            try:
                metrics = get_key_metrics(symbol, period="annual", limit=2)
                if metrics and isinstance(metrics, list) and metrics:
                    m = metrics[0]
                    rev_growth = m.get("revenuePerShareGrowth") or m.get("revenueGrowth")
                    if rev_growth is not None and rev_growth > 0.10:
                        fundamental_bonus += 0.10
                    elif rev_growth is not None and rev_growth > 0:
                        fundamental_bonus += 0.03

                    peg = m.get("pegRatio")
                    if peg is not None and 0 < peg < 1.0:
                        fundamental_bonus += 0.08
            except Exception:
                pass

        # Blend: multi-lens score weighted by entry lens
        entry_lens = c.get("_entry_lens", "momentum")
        if entry_lens == "value":
            combined = (
                0.30 * momentum
                + 0.25 * value_score
                + 0.10 * quality_score
                + 0.15 * c.get("_lightweight_score", 0.5)
                + 0.20 * (0.5 + fundamental_bonus)
            )
        elif entry_lens == "quality":
            combined = (
                0.20 * momentum
                + 0.10 * value_score
                + 0.30 * quality_score
                + 0.15 * c.get("_lightweight_score", 0.5)
                + 0.25 * (0.5 + fundamental_bonus)
            )
        else:
            combined = (
                0.45 * momentum
                + 0.05 * value_score
                + 0.10 * quality_score
                + 0.15 * c.get("_lightweight_score", 0.5)
                + 0.25 * (0.5 + fundamental_bonus)
            )

        c["_quick_score"] = combined
        c["_fundamental_bonus"] = fundamental_bonus
        scored.append(c)

    scored.sort(key=lambda x: x.get("_quick_score", 0), reverse=True)

    # --- Stage 5c: Region-balanced selection ---
    # Ensure minimum representation per region before filling with global best
    region_min_pct = getattr(config, "DISCOVERY_REGION_MIN_PCT", 0.15)
    min_per_region = max(2, int(top_n * region_min_pct))

    def _ticker_region(ticker: str) -> str:
        if ticker.endswith(".L"):
            return "UK"
        if any(ticker.endswith(s) for s in (".DE", ".PA", ".MI", ".MC", ".AS", ".BR",
                                             ".LS", ".SW", ".ST", ".CO", ".HE", ".OL")):
            return "EU"
        if any(ticker.endswith(s) for s in (".T", ".HK", ".SI", ".KS", ".AX")):
            return "APAC"
        return "US"

    # First pass: take best from each region up to minimum
    selected = []
    selected_tickers = set()
    region_selected: dict[str, int] = {}

    for region in ("US", "EU", "UK", "APAC"):
        region_cands = [c for c in scored if _ticker_region(c.get("symbol", "")) == region
                        and c.get("symbol") not in selected_tickers]
        for c in region_cands[:min_per_region]:
            selected.append(c)
            selected_tickers.add(c.get("symbol"))
            region_selected[region] = region_selected.get(region, 0) + 1

    # Second pass: fill remaining slots with global best
    remaining = top_n - len(selected)
    for c in scored:
        if remaining <= 0:
            break
        if c.get("symbol") not in selected_tickers:
            selected.append(c)
            selected_tickers.add(c.get("symbol"))
            remaining -= 1

    logger.info("Quick rank: %d selected (regions: %s)", len(selected),
                {r: region_selected.get(r, 0) for r in ("US", "EU", "UK", "APAC")})

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
    _CHECKPOINT_TMP = _CHECKPOINT_PATH.with_suffix(".tmp")
    if _CHECKPOINT_TMP.exists():
        _CHECKPOINT_TMP.unlink(missing_ok=True)

    # --- Checkpoint recovery: reload results from a crashed prior run ---
    results = []
    scored_symbols: set[str] = set()
    start_idx = 0

    try:
        if _CHECKPOINT_PATH.exists():
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
                logger.info("Checkpoint recovered: %d/%d already scored, resuming",
                            start_idx, len(candidates))
            else:
                logger.info("Checkpoint stale (candidate list changed), starting fresh")
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
            tmp = _CHECKPOINT_PATH.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as _cf:
                _cf.write(json.dumps(payload, default=str, separators=(",", ":")))
                _cf.flush()
                os.fsync(_cf.fileno())
            tmp.replace(_CHECKPOINT_PATH)
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
            },
            "currency": currency,
            "exchange": exchange,
        })

    def _score_one(item: dict) -> dict | None:
        """Score a single candidate with timeout. Returns result or None."""
        symbol = item["holding"]["ticker"]
        try:
            result = analyse_holding(item["holding"])
            c = item["candidate"]
            result["_candidate"] = c
            result["_currency"] = item["currency"]
            result["_exchange"] = item["exchange"]
            result["_country"] = c.get("country", "")
            result["_sector"] = c.get("sector", "")
            result["_industry"] = c.get("industry", "")
            result["_market_cap"] = c.get("marketCap", 0)
            result["_max_correlation"] = c.get("_max_correlation", 0)
            result["_correlated_with"] = c.get("_correlated_with", "")
            result["_momentum_score"] = c.get("_momentum_score", 0)
            result["_ret_90d"] = c.get("_ret_90d", 0)
            result["_ret_30d"] = c.get("_ret_30d", 0)
            result["_volume_ratio"] = c.get("_volume_ratio", 1.0)
            result["_beta"] = c.get("_beta")
            result["_entry_lens"] = c.get("_entry_lens", "momentum")
            result["_quick_filter_penalty"] = c.get("_quick_filter_penalty", 0)
            return result
        except Exception as e:
            logger.warning("Failed to score %s: %s", symbol, e)
            return None

    logger.info("Stage 6: scoring %d candidates with %d workers (checkpoint every %d)",
                len(work_items), _SCORING_WORKERS, _CHECKPOINT_INTERVAL)

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
                        logger.info("Checkpoint saved: %d/%d scored",
                                    len(results), len(candidates))
            except concurrent.futures.TimeoutError:
                logger.warning("Timeout scoring %s after %ds — skipping",
                               symbol, _per_ticker_timeout)
                future.cancel()  # Prevent queued futures from starting
            except Exception as e:
                logger.warning("Failed to score %s: %s", symbol, e)

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
    sector_max = getattr(config, "DISCOVERY_SECTOR_CONCENTRATION_MAX", 0.40)
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

    # --- Cross-sectional z-scoring of pillar scores ---
    _use_zscore = False
    if (getattr(config, "DISCOVERY_CROSS_SECTIONAL_ZSCORE", False)
            and len(scored_results) >= 5):
        _tech_arr = np.array([r.get("technical_score", 0) or 0 for r in scored_results])
        _fund_arr = np.array([r.get("fundamental_score", 0) or 0 for r in scored_results])
        _sent_arr = np.array([r.get("sentiment_score", 0) or 0 for r in scored_results])
        _fcast_arr = np.array([r.get("forecast_score", 0) or 0 for r in scored_results])
        for arr, key in [
            (_tech_arr, "_z_technical"), (_fund_arr, "_z_fundamental"),
            (_sent_arr, "_z_sentiment"), (_fcast_arr, "_z_forecast"),
        ]:
            z = _cross_sectional_zscore(arr)
            for i, r in enumerate(scored_results):
                r[key] = float(z[i])
        _use_zscore = True
        logger.info("Cross-sectional z-scoring applied to %d candidates", len(scored_results))

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
    try:
        if not getattr(config, "ML_RANKER_SHADOW_ONLY", True):
            from engine.ml_ranker import train_model, predict_alpha, is_available as ml_available
            train_model()
            if ml_available():
                _use_ml_ranker = True
                _ml_predict = predict_alpha
                logger.info("ML ranker active for final ranking")
        else:
            logger.info("ML ranker held in shadow mode")
    except Exception as e:
        logger.debug("ML ranker unavailable: %s", e)

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

    candidates = []

    for r in scored_results:
        currency = r.get("_currency", "USD")
        exchange = r.get("_exchange", "")
        ticker = r.get("ticker", "")
        sector = r.get("_sector", "")

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
        raw_sent = r.get("sentiment_score", 0) or 0
        # At confidence 0.3 or below, sentiment contributes only 30% of its score
        sent_effective = raw_sent * max(0.3, sent_confidence)

        if _use_zscore:
            # Use cross-sectionally normalized scores for ranking
            z_sent = r.get("_z_sentiment", raw_sent)
            z_sent_eff = z_sent * max(0.3, sent_confidence)
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
            overlay = apply_risk_overlay(r, ticker)
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
        elif current_sector_weight > 0.25:
            sector_penalty = -0.3

        # Fit score: 1.0 = perfect fit, 0.0 = poor fit
        portfolio_fit = max(0.0, min(1.0, 1.0 + corr_penalty + sector_penalty))
        sector_weight_if_added = current_sector_weight + 0.05

        # --- Quality Gate ---
        tech_s = r.get("technical_score", 0) or 0
        fund_s = r.get("fundamental_score", 0) or 0
        sent_s = r.get("sentiment_score", 0) or 0
        fcast_s = forecast_score or 0
        pillars_all_zero = (abs(tech_s) + abs(fund_s) + abs(sent_s) + abs(fcast_s)) < 0.001

        # Data confidence heuristic (empirically calibrated, not theoretically derived).
        # 0.40: pillar data availability (binary: are any non-zero?)
        # 0.30: sentiment data quality (article count, source diversity)
        # 0.15: trend confirmation (above SMA-50 = structural uptrend intact)
        # 0.15: earnings clarity (no recent miss = fundamental thesis intact)
        # Revisit weighting after 6+ months of evaluation data.
        _data_confidence = min(1.0, (
            (0.0 if pillars_all_zero else 0.4)  # pillar data available
            + 0.3 * min(sent_confidence, 1.0)   # sentiment data quality
            + 0.15 * (1.0 if r.get("_above_sma50") else 0.5)  # trend confirmation
            + 0.15 * (1.0 if not overlay.earnings_miss else 0.5)  # earnings clarity
        ))
        _data_confidence *= max(0.3, overlay.confidence_discount)
        # Discount confidence when risk data is missing
        if _risk_overlay_failed:
            _data_confidence *= 0.5

        # --- Momentum Bonus ---
        momentum_score = r.get("_momentum_score", 0.5)

        # --- Final Rank: alpha × confidence + fit_adjustment ---
        # Alpha: the pillar-driven quality signal (optionally blended with ML)
        alpha_pre_vol = adjusted_aggregate + _quality_overlay_score(r)

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

        if _use_ml_ranker:
            ml_features = {
                "technical_score": tech_s, "fundamental_score": fund_s,
                "sentiment_score": sent_s, "forecast_score": fcast_s,
                "momentum_score": momentum_score,
                "rsi": r.get("rsi"), "adx": r.get("adx"), "bb_pct": r.get("bb_pct"),
                "pe_ratio": r.get("pe_ratio"), "peg_ratio": r.get("peg_ratio"),
                "revenue_growth": r.get("revenue_growth"), "roe": r.get("roe"),
                "short_pct": r.get("short_pct"),
                "vix_percentile": r.get("vix_percentile"), "vol_20d": r.get("vol_20d"),
                "return_10d_prior": r.get("return_10d_prior"),
                "return_30d_prior": r.get("return_30d_prior"),
                "return_90d_prior": r.get("return_90d_prior"),
            }
            ml_alpha = _ml_predict(ml_features)
            if ml_alpha is not None:
                blend = getattr(config, "ML_RANKER_BLEND_PCT", 0.15)
                # Normalize ml_alpha (pct return) to [-1,1] scale
                ml_normalized = max(-1.0, min(1.0, ml_alpha / 15.0))
                alpha = blend * ml_normalized + (1.0 - blend) * alpha

        alpha += _pead_overlay_score(r, overlay)

        # Final rank formula (separated concerns):
        #   alpha × confidence → "how good is this stock, and how sure are we?"
        #   + fit_adjustment   → "does it improve the portfolio?"
        #   + momentum_bonus   → "is the trend confirming?"
        if is_momentum:
            final_rank = (
                0.40 * (alpha * _data_confidence)
                + 0.30 * momentum_score
                + 0.30 * portfolio_fit
            )
        else:
            final_rank = (
                0.55 * (alpha * _data_confidence)
                + 0.15 * momentum_score
                + 0.30 * portfolio_fit
            )

        # If pillar analysis completely failed, demote heavily
        if pillars_all_zero:
            final_rank *= 0.30
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
                if _entry_data.get("entry_price") and _stop_data.get("stop_loss"):
                    from engine.portfolio_optimizer import _get_fx_rate
                    _fx_rate = _get_fx_rate(currency)
                    _gbx_factor = 0.01 if currency == "GBX" else 1.0
                    _entry_gbp = _entry_data["entry_price"] * _gbx_factor * _fx_rate
                    _stop_gbp = _stop_data["stop_loss"] * _gbx_factor * _fx_rate
                    _tp_local = _target_data.get("take_profit")
                    _tp_gbp = _tp_local * _gbx_factor * _fx_rate if _tp_local else None
                    _size_data = calculate_position_size(
                        portfolio_value_gbp, _entry_gbp, _stop_gbp,
                        take_profit=_tp_gbp,
                        risk_per_trade_pct=getattr(config, "POSITION_RISK_BUDGET_PCT", 0.01),
                        kelly_cap_fraction=kelly_fractions.get(action))
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

        candidates.append(ScoredCandidate(
            ticker=ticker,
            name=r.get("name", ticker),
            exchange=exchange,
            country=r.get("_country", ""),
            sector=sector,
            industry=r.get("_industry", ""),
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
            volume_ratio=round(r.get("_volume_ratio", 1.0), 2),
            analyst_target=_analyst_target,
            analyst_upside=_analyst_upside,
            num_analysts=_num_analysts,
            insider_buys=_insider_buys,
            insider_sells=_insider_sells,
            insider_net=r.get("insider_net", "") or "",
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

    candidates.sort(key=lambda x: x.final_rank, reverse=True)

    # --- Diversified Final Selector ---
    # Enforce sector caps and region floors so results aren't dominated
    # by one sector or region. Pure rank ordering is preserved within constraints.
    max_per_sector = getattr(config, "DISCOVERY_MAX_PER_SECTOR", 4)
    min_regions = getattr(config, "DISCOVERY_MIN_REGIONS", 2)

    def _candidate_region(ticker: str) -> str:
        if ticker.endswith(".L"):
            return "UK"
        if any(ticker.endswith(s) for s in (".DE", ".PA", ".MI", ".MC", ".AS", ".BR",
                                             ".LS", ".SW", ".ST", ".CO", ".HE", ".OL")):
            return "EU"
        if any(ticker.endswith(s) for s in (".T", ".HK", ".SI", ".KS", ".AX")):
            return "APAC"
        return "US"

    sector_counts: dict[str, int] = {}
    diversified = []
    deferred = []  # candidates that exceeded sector cap

    for c in candidates:
        sector = c.sector or "Unknown"
        count = sector_counts.get(sector, 0)
        if count < max_per_sector:
            diversified.append(c)
            sector_counts[sector] = count + 1
        else:
            deferred.append(c)

    # Check region diversity in top 10
    top_10_regions = set(_candidate_region(c.ticker) for c in diversified[:10])
    if len(top_10_regions) < min_regions and deferred:
        # Swap in candidates from missing regions
        missing_regions = {"US", "EU", "UK"} - top_10_regions
        for region in missing_regions:
            region_cands = [c for c in deferred if _candidate_region(c.ticker) == region]
            if region_cands:
                # Insert the best candidate from this region into position ~8-10
                best = region_cands[0]
                insert_pos = min(9, len(diversified) - 1)
                diversified.insert(insert_pos, best)
                deferred.remove(best)
                logger.info("Diversity: inserted %s (%s) at position %d for region %s",
                            best.ticker, best.sector, insert_pos, region)

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

    candidates = _stage_universe_assembly(existing_tickers, progress_callback)
    result.screened_count = len(candidates)
    logger.info("Stage 1: Assembled %d candidates (FMP US + global universe)", len(candidates))

    if not candidates:
        result.error = "No candidates found from any source"
        result.run_time_seconds = time.time() - start_time
        return result

    # Stage 2: Momentum Screen
    if progress_callback:
        progress_callback("Stage 2: Momentum screening...", 1, 7)

    candidates, price_cache = _stage_momentum_screen(candidates, progress_callback)
    result.after_momentum_screen = len(candidates)
    logger.info("Stage 2: %d candidates after momentum screen", len(candidates))

    if not candidates:
        result.error = "No candidates passed momentum screen"
        result.run_time_seconds = time.time() - start_time
        return result

    # Stage 3: Quick Filter
    if progress_callback:
        progress_callback("Stage 3: Applying quick filters...", 2, 7)

    candidates = _stage_quick_filter(candidates, portfolio_sectors, rejections)
    result.after_quick_filter = len(candidates)
    logger.info("Stage 3: %d candidates after quick filter", len(candidates))

    # Stage 4: Correlation Filter
    if progress_callback:
        progress_callback("Stage 4: Computing correlations...", 3, 7)

    candidates = _stage_correlation_filter(
        candidates, existing_ticker_list, rejections, price_cache, progress_callback,
    )
    result.after_corr_filter = len(candidates)
    logger.info("Stage 4: %d candidates with correlation penalties (soft filter, no rejections)", len(candidates))

    if not candidates:
        result.rejections = rejections
        result.error = "All candidates filtered out"
        result.run_time_seconds = time.time() - start_time
        return result

    # Stage 5: Quick Rank
    if progress_callback:
        progress_callback("Stage 5: Ranking candidates...", 4, 7)

    top_n = getattr(config, "DISCOVERY_TOP_N_FULL_SCORE", 30)
    candidates = _stage_quick_rank(candidates, top_n, progress_callback)
    result.after_quick_rank = len(candidates)
    logger.info("Stage 5: Top %d candidates for full scoring", len(candidates))

    # Stage 6: Full Scoring
    if progress_callback:
        progress_callback("Stage 6: Running full analysis...", 5, 7)

    scored = _stage_full_scoring(candidates, progress_callback)
    result.fully_scored = len(scored)
    logger.info("Stage 6: %d candidates fully scored", len(scored))

    # Stage 7: FX + Fit + Final Ranking
    if progress_callback:
        progress_callback("Stage 7: Computing final rankings...", 6, 7)

    final_candidates = _stage_final_ranking(scored, portfolio_sectors, holdings)
    result.candidates = final_candidates
    result.rejections = rejections
    result.fx_penalties_applied = sum(1 for c in final_candidates if c.fx_penalty_applied)
    result.run_time_seconds = round(time.time() - start_time, 1)

    logger.info("Discovery complete: %d candidates ranked in %.1fs",
                len(final_candidates), result.run_time_seconds)

    return result
