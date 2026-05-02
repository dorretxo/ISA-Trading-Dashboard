"""Factor Momentum & Network Momentum module.

Implements dynamic factor tilt weights based on recent factor portfolio
performance (Ehsani & Linnainmaa 2022, "Factor Momentum and the Momentum
Factor", Journal of Finance). Factors that have performed well recently
tend to continue performing well — this module captures that signal.

Also provides a placeholder for supply-chain / network momentum
(Cohen & Frazzini 2008, "Economic Links and Predictable Returns").

Degrades gracefully: returns neutral tilts when data is insufficient.
"""

import json
import logging
import os
import time

import numpy as np

import config
from utils.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)

_CACHE_FILE = os.path.join("feature_cache", "factor_momentum.json")


def _default_tilts() -> dict:
    """Neutral tilts when data is insufficient."""
    return {
        "momentum_tilt": 0.0,
        "value_tilt": 0.0,
        "quality_tilt": 0.0,
        "momentum_ret_1m": 0.0,
        "momentum_ret_3m": 0.0,
        "value_ret_1m": 0.0,
        "value_ret_3m": 0.0,
        "quality_ret_1m": 0.0,
        "quality_ret_3m": 0.0,
        "computed_at": None,
    }


def compute_factor_returns(feature_store_data: dict) -> dict:
    """Build long/short quintile portfolios and compute rolling factor returns.

    Uses factor-aware rows (already batch-computed, no API calls) to construct:
    - Momentum factor: long top-20% by momentum factor score, short bottom-20%
    - Value factor: long top-20% by value factor score, short bottom-20%
    - Quality factor: long top-20% by quality factor score, short bottom-20%

    Returns dict with 1m/3m rolling factor returns and tilt weights in [-1, 1].
    Tilt interpretation: +1 = factor is hot (overweight), -1 = factor is cold.

    Args:
        feature_store_data: dict of {ticker: feature_dict} from FeatureStore or
            scored discovery candidates. When explicit factor scores are absent,
            price-based fallbacks are used so the function remains backward compatible.
    """
    min_universe = int(getattr(config, "FACTOR_MOMENTUM_MIN_UNIVERSE", 10))
    if not feature_store_data or len(feature_store_data) < min_universe:
        logger.info(
            "Factor momentum: insufficient data (%d tickers, need %d+)",
            len(feature_store_data) if feature_store_data else 0,
            min_universe,
        )
        return _default_tilts()

    tickers = list(feature_store_data.keys())
    n = len(tickers)
    quintile_size = max(1, n // 5)

    # Extract factor scores
    momentum_vals = []
    value_vals = []
    quality_vals = []
    ret_arrays = {}

    for t in tickers:
        f = feature_store_data[t]
        momentum_score = f.get("momentum_factor_score")
        if momentum_score is None:
            momentum_score = f.get("ret_90d", 0) or 0
        momentum_vals.append(momentum_score)

        value_score = f.get("value_factor_score")
        if value_score is None:
            # Backward-compatible value proxy: beaten-down stocks
            value_score = -(f.get("pct_from_high_252d", 1) or 1)
        value_vals.append(value_score)

        quality_score = f.get("quality_factor_score")
        if quality_score is None:
            # Backward-compatible quality proxy when fundamentals are absent
            vol = f.get("vol_20d", 0.3) or 0.3
            above_200 = 1.0 if f.get("above_sma200", False) else 0.0
            quality_score = above_200 - vol
        quality_vals.append(quality_score)

        # Store return arrays for portfolio return calculation
        returns = f.get("returns_90d", f.get("returns_60d", []))
        if isinstance(returns, (list, np.ndarray)) and len(returns) > 0:
            ret_arrays[t] = np.array(returns, dtype=np.float64)

    result = _default_tilts()

    for factor_name, factor_vals in [
        ("momentum", momentum_vals),
        ("value", value_vals),
        ("quality", quality_vals),
    ]:
        # Rank tickers by factor
        sorted_indices = np.argsort(factor_vals)
        long_tickers = [tickers[i] for i in sorted_indices[-quintile_size:]]
        short_tickers = [tickers[i] for i in sorted_indices[:quintile_size]]

        # Compute equal-weight portfolio returns (using available return arrays)
        long_rets = [ret_arrays[t] for t in long_tickers if t in ret_arrays]
        short_rets = [ret_arrays[t] for t in short_tickers if t in ret_arrays]

        if not long_rets or not short_rets:
            continue

        # Align to shortest array, compute L/S spread
        min_len = min(
            min(len(r) for r in long_rets),
            min(len(r) for r in short_rets),
        )
        lookback_1m = int(getattr(config, "FACTOR_MOMENTUM_LOOKBACK_1M", 21))
        lookback_3m = int(getattr(config, "FACTOR_MOMENTUM_LOOKBACK_3M", 63))
        require_3m = bool(getattr(config, "FACTOR_MOMENTUM_REQUIRE_3M", True))
        if min_len < lookback_1m or (require_3m and min_len < lookback_3m):
            continue

        long_portfolio = np.mean([r[-min_len:] for r in long_rets], axis=0)
        short_portfolio = np.mean([r[-min_len:] for r in short_rets], axis=0)
        ls_daily = long_portfolio - short_portfolio

        ret_1m = float(np.sum(ls_daily[-lookback_1m:])) if len(ls_daily) >= lookback_1m else 0.0
        ret_3m = float(np.sum(ls_daily[-lookback_3m:])) if len(ls_daily) >= lookback_3m else 0.0

        # Tilt: sign and magnitude of recent factor return, scaled to [-1, 1]
        # Use 1m return as primary signal, 3m as confirmation
        spread_window = ls_daily[-lookback_3m:] if len(ls_daily) >= lookback_3m else ls_daily[-lookback_1m:]
        spread_vol = float(np.std(spread_window) * np.sqrt(max(len(spread_window), 1)))
        blended_ret = 0.35 * ret_1m + 0.65 * ret_3m
        if spread_vol > 1e-8:
            raw_signal = blended_ret / spread_vol
        elif abs(blended_ret) > 1e-8:
            raw_signal = float(np.sign(blended_ret) * 3.0)
        else:
            raw_signal = 0.0
        tilt = float(np.tanh(raw_signal / 2.0))
        if ret_1m and ret_3m and np.sign(ret_1m) != np.sign(ret_3m):
            tilt *= float(getattr(config, "FACTOR_MOMENTUM_CONFIRMATION_MULT", 0.25))
        cap = float(getattr(config, "FACTOR_MOMENTUM_TILT_CAP", 0.35))
        tilt = float(np.clip(tilt, -cap, cap))

        result[f"{factor_name}_ret_1m"] = round(ret_1m, 6)
        result[f"{factor_name}_ret_3m"] = round(ret_3m, 6)
        result[f"{factor_name}_tilt"] = round(tilt, 4)

    result["computed_at"] = time.time()

    # Persist to cache
    try:
        os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
        atomic_write_json(_CACHE_FILE, result, indent=2)
    except Exception as e:
        logger.warning("Factor momentum: cache write failed: %s", e)

    logger.info(
        "Factor momentum: mom_tilt=%.2f val_tilt=%.2f qual_tilt=%.2f",
        result["momentum_tilt"], result["value_tilt"], result["quality_tilt"],
    )
    return result


def load_cached_factor_returns() -> dict:
    """Load cached factor returns if fresh, else return default tilts."""
    ttl = getattr(config, "FACTOR_MOMENTUM_CACHE_TTL", 86400)
    try:
        if os.path.exists(_CACHE_FILE):
            with open(_CACHE_FILE) as f:
                data = json.load(f)
            computed_at = data.get("computed_at", 0)
            if computed_at and (time.time() - computed_at) < ttl:
                return data
    except Exception as e:
        logger.debug("Factor momentum: cache read failed: %s", e)
    return _default_tilts()


def get_factor_tilt_features(factor_returns: dict) -> dict:
    """Convert factor returns dict to flat ML feature dict.

    Returns 9 features suitable for adding to ML ranker input space:
    - 3 tilt values (momentum, value, quality)
    - 6 raw returns (1m and 3m for each factor)
    """
    return {
        "factor_mom_momentum_tilt": factor_returns.get("momentum_tilt", 0.0),
        "factor_mom_value_tilt": factor_returns.get("value_tilt", 0.0),
        "factor_mom_quality_tilt": factor_returns.get("quality_tilt", 0.0),
        "factor_mom_momentum_ret_1m": factor_returns.get("momentum_ret_1m", 0.0),
        "factor_mom_momentum_ret_3m": factor_returns.get("momentum_ret_3m", 0.0),
        "factor_mom_value_ret_1m": factor_returns.get("value_ret_1m", 0.0),
        "factor_mom_value_ret_3m": factor_returns.get("value_ret_3m", 0.0),
        "factor_mom_quality_ret_1m": factor_returns.get("quality_ret_1m", 0.0),
        "factor_mom_quality_ret_3m": factor_returns.get("quality_ret_3m", 0.0),
    }


def load_optional_supply_chain_mapping(path: str | None = None) -> dict:
    """Load a local supplier/customer mapping if one has been provided."""
    if path is None:
        path = os.path.join("feature_cache", "supply_chain_mapping.json")

    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception as e:
        logger.debug("Supply-chain mapping unavailable: %s", e)
    return {}


def calculate_network_momentum(
    ticker: str,
    supply_chain_mapping: dict | None = None,
    returns_df=None,
) -> float | None:
    """Calculate 1-month lagged return of key suppliers/customers.

    Placeholder for Cohen & Frazzini (2008) "Economic Links and Predictable
    Returns". When supply chain data is available (e.g., from FactSet Revere
    or SEC 10-K supplier disclosures), this computes the equal-weighted
    1-month lagged return of a company's disclosed suppliers and customers.

    The academic finding: a 1-standard-deviation increase in customer stock
    returns predicts a 1.4% monthly excess return in supplier stocks, with
    the effect persisting for 1-2 months due to limited investor attention.

    Args:
        ticker: Stock symbol to compute network momentum for.
        supply_chain_mapping: Dict mapping ticker → list of supplier/customer tickers.
            Expected format: {"AAPL": ["TSM", "QCOM", "AVGO", ...]}
        returns_df: DataFrame with columns as tickers, rows as dates, values as returns.

    Returns:
        Float network momentum score, or None if data unavailable.
    """
    if supply_chain_mapping is None or returns_df is None:
        return None

    linked_tickers = supply_chain_mapping.get(ticker, [])
    if not linked_tickers:
        return None

    try:
        import pandas as pd

        lookback = getattr(config, "FACTOR_MOMENTUM_LOOKBACK_1M", 21)
        available = [t for t in linked_tickers if t in returns_df.columns]
        if not available:
            return None

        # Lagged 1-month equal-weighted return of linked companies
        linked_returns = returns_df[available].iloc[-lookback:]
        network_ret = float(linked_returns.mean(axis=1).sum())
        return round(network_ret, 6)
    except Exception:
        return None
