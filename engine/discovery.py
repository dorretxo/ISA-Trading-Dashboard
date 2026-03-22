"""Global Discovery Engine v2 — Expanded Universe + Momentum Screening.

Hybrid approach:
- FMP Screener for US stocks (NYSE, NASDAQ, AMEX) — no cap restrictions
- Curated global universe (~800 tickers) screened via yfinance momentum

Funnel stages:
1. Universe Assembly → ~2000-3000 candidates (FMP US + yfinance global)
2. Momentum Screen  → ~150 (price data download + momentum rank)
3. Quick Filter     → ~120 (beta, penny stock, volume)
4. Correlation Filter → ~80-100 (vs existing holdings)
5. Quick Rank       → top 30 (momentum + fundamentals blend)
6. Full Scoring     → 30 scored (analyse_holding pipeline)
7. FX Penalty + Portfolio Fit → final ranked list
"""

import logging
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

logger = logging.getLogger(__name__)


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
    # Risk overlay
    parabolic_penalty: float = 0.0
    is_parabolic: bool = False
    earnings_near: bool = False
    earnings_imminent: bool = False
    earnings_days: int | None = None
    cap_tier: str = "unknown"
    confidence_discount: float = 1.0
    max_weight_scale: float = 1.0
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
            from utils.global_universe import get_global_universe
            global_tickers = get_global_universe(exclude_tickers=existing_tickers)

            for ticker in global_tickers:
                if ticker in seen_symbols:
                    continue
                seen_symbols.add(ticker)
                exchange = _detect_exchange(ticker)
                all_candidates.append({
                    "symbol": ticker,
                    "companyName": ticker,  # Will be enriched later
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
    """Download price data and compute momentum scores. Returns (filtered, price_cache)."""
    top_n = getattr(config, "MOMENTUM_TOP_N_PRESCREEN", 150)
    min_avg_vol = getattr(config, "MOMENTUM_MIN_AVG_VOLUME", 100_000)

    tickers = [c.get("symbol", "") for c in candidates if c.get("symbol")]
    ticker_to_candidate = {c.get("symbol", ""): c for c in candidates}

    # Download price data in batches
    batch_size = 80
    all_momentum = {}

    for batch_start in range(0, len(tickers), batch_size):
        batch = tickers[batch_start:batch_start + batch_size]
        if progress_callback:
            progress_callback(
                f"Downloading prices... ({batch_start}/{len(tickers)})",
                batch_start, len(tickers),
            )

        try:
            data = yf.download(
                batch, period="120d", progress=False, auto_adjust=True,
                threads=True,
            )
            if data is None or data.empty:
                continue

            if len(batch) == 1:
                # Single ticker — different DataFrame structure
                metrics = _compute_momentum_metrics(data, batch[0])
                if metrics:
                    all_momentum[batch[0]] = metrics
            else:
                for sym in batch:
                    try:
                        # Extract single-ticker data from multi-ticker DataFrame
                        single_data = pd.DataFrame()
                        for col in ["Close", "Volume", "High", "Low", "Open"]:
                            if col in data.columns:
                                if isinstance(data[col], pd.DataFrame) and sym in data[col].columns:
                                    single_data[col] = data[col][sym]
                                elif isinstance(data[col], pd.Series):
                                    single_data[col] = data[col]
                        if not single_data.empty and "Close" in single_data.columns:
                            metrics = _compute_momentum_metrics(single_data, sym)
                            if metrics:
                                all_momentum[sym] = metrics
                    except Exception:
                        continue
        except Exception as e:
            logger.warning("Batch download failed (%d-%d): %s", batch_start, batch_start + batch_size, e)
            continue

    logger.info("Momentum data downloaded for %d / %d tickers", len(all_momentum), len(tickers))

    # Score and rank by momentum
    scored = []
    for ticker, m in all_momentum.items():
        # Skip low-volume stocks (unless from FMP which already filters)
        cand = ticker_to_candidate.get(ticker)
        if not cand:
            continue

        if cand.get("_source") != "fmp" and m["avg_volume"] < min_avg_vol:
            continue

        # Composite momentum score (raw — will be percentile-ranked)
        scored.append((ticker, m))

    if not scored:
        return [], {}

    # Percentile-rank the momentum factors
    ret_90d_vals = np.array([m["ret_90d"] for _, m in scored])
    ret_30d_vals = np.array([m["ret_30d"] for _, m in scored])
    ret_10d_vals = np.array([m["ret_10d"] for _, m in scored])
    vol_ratio_vals = np.array([m["volume_ratio"] for _, m in scored])
    from_high_vals = np.array([m["pct_from_high"] for _, m in scored])

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
        rank_vol = percentile_rank(vol_ratio_vals)
        rank_high = percentile_rank(from_high_vals)
    except ImportError:
        # Fallback if scipy not available
        def simple_rank(arr):
            order = np.argsort(np.argsort(arr))
            return order / max(len(arr) - 1, 1)
        rank_90d = simple_rank(ret_90d_vals)
        rank_30d = simple_rank(ret_30d_vals)
        rank_10d = simple_rank(ret_10d_vals)
        rank_vol = simple_rank(vol_ratio_vals)
        rank_high = simple_rank(from_high_vals)

    # --- Relative Strength within Sectors ---
    # Compare each stock's 90d return to its sector median. A +5% stock
    # in a -10% sector is stronger than a +15% stock in a +20% sector.
    sector_returns = {}
    for i, (ticker, m) in enumerate(scored):
        sector = ticker_to_candidate.get(ticker, {}).get("sector", "Unknown")
        if sector not in sector_returns:
            sector_returns[sector] = []
        sector_returns[sector].append((i, m["ret_90d"]))

    # Compute sector medians and relative strength
    sector_medians = {}
    for sector, entries in sector_returns.items():
        returns = [r for _, r in entries]
        sector_medians[sector] = float(np.median(returns)) if returns else 0

    relative_strength = np.zeros(len(scored))
    for i, (ticker, m) in enumerate(scored):
        sector = ticker_to_candidate.get(ticker, {}).get("sector", "Unknown")
        median = sector_medians.get(sector, 0)
        # Relative strength = how much this stock outperforms its sector
        relative_strength[i] = m["ret_90d"] - median

    rank_relative = percentile_rank(relative_strength)

    # Composite momentum score (now includes relative strength)
    momentum_scores = (
        0.25 * rank_90d +    # Absolute 90-day return
        0.20 * rank_relative + # Relative strength vs sector peers
        0.20 * rank_30d +    # Recent trend confirmation
        0.10 * rank_10d +    # Very recent momentum
        0.10 * rank_vol +    # Volume confirmation
        0.15 * rank_high     # Near 52-week high = breakout
    )

    # Sort by momentum score, keep top N
    indices = np.argsort(momentum_scores)[::-1][:top_n]

    filtered = []
    price_cache = {}

    for idx in indices:
        ticker, m = scored[idx]
        cand = ticker_to_candidate[ticker]
        cand["_momentum_score"] = float(momentum_scores[idx])
        cand["_relative_strength"] = float(relative_strength[idx])
        cand["_sector_median_90d"] = sector_medians.get(
            cand.get("sector", "Unknown"), 0)
        cand["_ret_90d"] = m["ret_90d"]
        cand["_ret_30d"] = m["ret_30d"]
        cand["_ret_10d"] = m["ret_10d"]
        cand["_volume_ratio"] = m["volume_ratio"]
        cand["_pct_from_high"] = m["pct_from_high"]
        cand["_above_sma50"] = m["above_sma50"]
        cand["_last_price"] = m["last_price"]
        filtered.append(cand)

        # Cache returns for correlation filter
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
    """Filter by beta, penny stock, and basic sanity checks."""
    beta_max = getattr(config, "DISCOVERY_BETA_MAX", 2.5)
    passed = []

    for c in candidates:
        symbol = c.get("symbol", "")
        name = c.get("companyName", symbol)
        exchange = c.get("_exchange_query", "")

        # Beta filter (only available for FMP-sourced candidates)
        beta = c.get("beta")
        if beta is not None and beta > beta_max:
            rejections.append(CandidateRejection(
                symbol, name, exchange, "quick_filter",
                f"Beta too high ({beta:.2f} > {beta_max})",
            ))
            continue

        # Penny stock filter
        price = c.get("price") or c.get("_last_price", 0)
        if price is not None and 0 < price < 1.0:
            rejections.append(CandidateRejection(
                symbol, name, exchange, "quick_filter",
                f"Penny stock (price {price})",
            ))
            continue

        passed.append(c)

    return passed


# ---------------------------------------------------------------------------
# Stage 4: Correlation Filter (reuse cached price data)
# ---------------------------------------------------------------------------

def _stage_correlation_filter(
    candidates: list[dict],
    existing_tickers: list[str],
    rejections: list[CandidateRejection],
    price_cache: dict,
    progress_callback=None,
) -> list[dict]:
    """Remove candidates highly correlated with existing holdings."""
    corr_threshold = getattr(config, "DISCOVERY_CORRELATION_THRESHOLD", 0.70)

    if not existing_tickers or not candidates:
        return candidates

    # Download existing holdings' return data
    if progress_callback:
        progress_callback("Downloading portfolio prices for correlation...", 0, 1)

    existing_returns = {}
    for ticker in existing_tickers:
        try:
            data = yf.download(ticker, period="90d", progress=False, auto_adjust=True)
            if data is not None and len(data) > 20:
                returns = data["Close"].pct_change().dropna().values[-60:]
                if len(returns) >= 15:
                    existing_returns[ticker] = returns
        except Exception:
            continue

    if not existing_returns:
        return candidates

    min_len_existing = min(len(v) for v in existing_returns.values())
    passed = []

    for c in candidates:
        symbol = c.get("symbol", "")
        name = c.get("companyName", symbol)
        exchange = c.get("_exchange_query", "")

        cand_rets = price_cache.get(symbol)
        if cand_rets is None or len(cand_rets) < 15:
            # No price data — keep (benefit of doubt)
            c["_max_correlation"] = 0.0
            c["_correlated_with"] = ""
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

        if abs(max_corr) > corr_threshold:
            rejections.append(CandidateRejection(
                symbol, name, exchange, "correlation",
                f"High correlation ({max_corr:.2f}) with {corr_with}",
            ))
        else:
            c["_max_correlation"] = max_corr
            c["_correlated_with"] = corr_with
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
    """Two-stage ranking: lightweight on many, FMP fundamentals on fewer.

    Stage 5a: Score all candidates with fast technical + momentum metrics (no API calls).
              Keep top DISCOVERY_TOP_N_LIGHTWEIGHT (default 100).
    Stage 5b: Enrich top candidates with FMP fundamentals bonus.
              Return top DISCOVERY_TOP_N_FULL_SCORE (default 30).
    """
    lightweight_n = getattr(config, "DISCOVERY_TOP_N_LIGHTWEIGHT", 100)

    # --- Stage 5a: Lightweight technical ranking (no API calls) ---
    for i, c in enumerate(candidates):
        if progress_callback and i % 20 == 0:
            progress_callback(
                f"Lightweight ranking... ({i}/{len(candidates)})",
                i, len(candidates),
            )
        c["_lightweight_score"] = _lightweight_technical_score(c.get("symbol", ""), c)

    candidates.sort(key=lambda x: x.get("_lightweight_score", 0), reverse=True)
    candidates = candidates[:lightweight_n]

    if progress_callback:
        progress_callback(f"Enriching top {len(candidates)} with fundamentals...", 0, len(candidates))

    # --- Stage 5b: FMP fundamentals enrichment on reduced set ---
    scored = []
    for i, c in enumerate(candidates):
        symbol = c.get("symbol", "")
        if progress_callback and i % 10 == 0:
            progress_callback(
                f"Quick-ranking... ({i}/{len(candidates)})",
                i, len(candidates),
            )

        momentum = c.get("_momentum_score", 0.5)
        fundamental_bonus = 0.0

        # FMP fundamentals bonus (US tickers only — FMP Starter)
        if c.get("_source") == "fmp":
            try:
                metrics = get_key_metrics(symbol, period="annual", limit=2)
                if metrics and isinstance(metrics, list) and metrics:
                    m = metrics[0]
                    rev_growth = m.get("revenuePerShareGrowth") or m.get("revenueGrowth")
                    if rev_growth is not None and rev_growth > 0.10:
                        fundamental_bonus += 0.15
                    elif rev_growth is not None and rev_growth > 0:
                        fundamental_bonus += 0.05

                    roe = m.get("roe")
                    if roe is not None and roe > 0.15:
                        fundamental_bonus += 0.10
                    elif roe is not None and roe < 0:
                        fundamental_bonus -= 0.10

                    peg = m.get("pegRatio")
                    if peg is not None and 0 < peg < 1.0:
                        fundamental_bonus += 0.10
            except Exception:
                pass

        # Blend: 60% momentum + 20% lightweight + 20% fundamentals
        combined = (
            0.60 * momentum
            + 0.20 * c.get("_lightweight_score", 0.5)
            + 0.20 * (0.5 + fundamental_bonus)
        )
        c["_quick_score"] = combined
        scored.append(c)

    scored.sort(key=lambda x: x.get("_quick_score", 0), reverse=True)
    return scored[:top_n]


# ---------------------------------------------------------------------------
# Stage 6: Full Scoring Pipeline
# ---------------------------------------------------------------------------

def _stage_full_scoring(
    candidates: list[dict],
    progress_callback=None,
) -> list[dict]:
    """Run analyse_holding() on each candidate."""
    from engine.scoring import analyse_holding

    results = []
    for i, c in enumerate(candidates):
        symbol = c.get("symbol", "")
        name = c.get("companyName", symbol)
        exchange = c.get("_exchange_query", "")
        currency = _detect_currency(exchange, symbol)
        price = c.get("price") or c.get("_last_price", 0) or 0

        if progress_callback:
            progress_callback(
                f"Full analysis: {symbol} ({i + 1}/{len(candidates)})",
                i, len(candidates),
            )

        synthetic_holding = {
            "ticker": symbol,
            "name": name,
            "avg_buy_price": price,
            "quantity": 1,
            "currency": currency,
        }

        try:
            result = analyse_holding(synthetic_holding)
            result["_candidate"] = c
            result["_currency"] = currency
            result["_exchange"] = exchange
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
            results.append(result)
        except Exception as e:
            logger.warning("Failed to score %s: %s", symbol, e)
            continue

    return results


# ---------------------------------------------------------------------------
# Stage 7: FX Penalty + Portfolio Fit + Final Ranking
# ---------------------------------------------------------------------------

def _stage_final_ranking(
    scored_results: list[dict],
    portfolio_sectors: dict[str, float],
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

        adjusted_aggregate = (
            r.get("technical_score", 0) * weights.get("technical", 0.30)
            + r.get("fundamental_score", 0) * weights.get("fundamental", 0.20)
            + r.get("sentiment_score", 0) * weights.get("sentiment", 0.20)
            + forecast_score * weights.get("forecast", 0.30)
        )

        # --- Risk Overlay ---
        try:
            from engine.risk_overlay import apply_risk_overlay
            overlay = apply_risk_overlay(r, ticker)
        except Exception as e:
            logger.warning("Risk overlay failed for %s: %s", ticker, e)
            from engine.risk_overlay import RiskOverlay
            overlay = RiskOverlay()

        # Apply parabolic penalty (directly reduces alpha signal)
        adjusted_aggregate -= overlay.parabolic_penalty

        # --- Portfolio Fit Score ---
        max_corr = abs(r.get("_max_correlation", 0))
        corr_with = r.get("_correlated_with", "")

        corr_penalty = 0.0
        if max_corr > 0.40:
            corr_penalty = -0.5 * min(1.0, (max_corr - 0.40) / (corr_threshold - 0.40))

        sector_penalty = 0.0
        current_sector_weight = portfolio_sectors.get(sector, 0)
        if current_sector_weight > sector_max:
            sector_penalty = -0.5
        elif current_sector_weight > 0.25:
            sector_penalty = -0.3

        portfolio_fit = max(0.0, min(1.0, 1.0 + corr_penalty + sector_penalty))
        sector_weight_if_added = current_sector_weight + 0.05

        # --- Momentum Bonus in Final Rank ---
        momentum_score = r.get("_momentum_score", 0.5)

        # Final rank: blend aggregate, momentum, and fit
        if is_momentum:
            # Momentum mode: 50% aggregate + 25% momentum + 25% fit
            final_rank = 0.50 * adjusted_aggregate + 0.25 * momentum_score + 0.25 * portfolio_fit
        else:
            # Balanced mode: 70% aggregate + 30% fit
            final_rank = 0.70 * adjusted_aggregate + 0.30 * portfolio_fit

        # Determine action
        if adjusted_aggregate >= config.SCORE_STRONG_BUY_THRESHOLD:
            action = "STRONG BUY"
        elif adjusted_aggregate >= config.SCORE_BUY_THRESHOLD:
            action = "BUY"
        elif adjusted_aggregate >= config.SCORE_KEEP_THRESHOLD:
            action = "NEUTRAL"
        else:
            action = "AVOID"

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
            parabolic_penalty=overlay.parabolic_penalty,
            is_parabolic=overlay.is_parabolic,
            earnings_near=overlay.earnings_near,
            earnings_imminent=overlay.earnings_imminent,
            earnings_days=overlay.earnings_days,
            cap_tier=overlay.cap_tier,
            confidence_discount=overlay.confidence_discount,
            max_weight_scale=overlay.max_weight_scale,
            final_rank=round(final_rank, 3),
        ))

    candidates.sort(key=lambda x: x.final_rank, reverse=True)
    return candidates


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
    logger.info("Stage 4: %d candidates after correlation filter", len(candidates))

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

    final_candidates = _stage_final_ranking(scored, portfolio_sectors)
    result.candidates = final_candidates
    result.rejections = rejections
    result.fx_penalties_applied = sum(1 for c in final_candidates if c.fx_penalty_applied)
    result.run_time_seconds = round(time.time() - start_time, 1)

    logger.info("Discovery complete: %d candidates ranked in %.1fs",
                len(final_candidates), result.run_time_seconds)

    return result
