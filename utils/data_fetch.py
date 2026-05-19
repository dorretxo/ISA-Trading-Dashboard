"""Shared data fetching utilities with session-level caching."""

import json
import logging
import socket
import time
from pathlib import Path
from threading import Lock

import pandas as pd
import yfinance as yf

import config
from utils.atomic_io import atomic_write_json

# Enforce a global socket timeout so that yfinance HTTP calls (which hold the
# GIL inside C extensions like ssl/urllib3) cannot block indefinitely.  This is
# the belt-and-suspenders fix for GIL-blocked thread timeouts.
_SOCKET_TIMEOUT = getattr(config, "SOCKET_TIMEOUT", 45)
socket.setdefaulttimeout(_SOCKET_TIMEOUT)

# Session-level cache for yfinance data
_price_cache: dict[str, pd.DataFrame] = {}
_info_cache: dict[str, dict] = {}
_macro_cache: dict[str, pd.DataFrame] = {}
_reddit_cache: dict[str, tuple[list, float]] = {}  # {ticker: (posts, timestamp)}
_info_stats_lock = Lock()
_info_stats: dict[str, int] = {
    "cache": 0,
    "network": 0,
    "empty": 0,
    "error": 0,
    "timeout": 0,
    "skipped_cache_only": 0,
}


def _portfolio_path() -> Path:
    return Path(__file__).parent.parent / config.PORTFOLIO_FILE


def load_portfolio() -> list[dict]:
    """Load portfolio holdings from JSON file."""
    with open(_portfolio_path(), "r") as f:
        data = json.load(f)
    return data["holdings"]


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


def load_portfolio_full() -> dict:
    """Load the full portfolio file including trade_history."""
    with open(_portfolio_path(), "r") as f:
        return json.load(f)


def save_portfolio(data: dict) -> None:
    """Atomic write of portfolio data."""
    path = _portfolio_path()
    atomic_write_json(path, data, indent=4, default=str)


def record_sale(
    ticker: str,
    sell_price: float,
    quantity: int | float,
    sell_date: str,
    notes: str = "",
) -> dict | None:
    """Record a stock sale: remove/reduce from holdings, add to trade_history.

    Returns the trade record, or None if ticker not found.
    """
    data = load_portfolio_full()
    holdings = data.get("holdings", [])

    # Find the holding
    idx = None
    for i, h in enumerate(holdings):
        if h["ticker"].upper() == ticker.upper():
            idx = i
            break

    if idx is None:
        return None

    holding = holdings[idx]
    sold_qty = min(quantity, holding["quantity"])

    # Build trade record
    trade = {
        "ticker": holding["ticker"],
        "name": holding.get("name", holding["ticker"]),
        "currency": holding.get("currency", "GBP"),
        "buy_price": holding["avg_buy_price"],
        "sell_price": sell_price,
        "quantity": sold_qty,
        "buy_date": holding.get("buy_date"),
        "sell_date": sell_date,
        "pnl": round((sell_price - holding["avg_buy_price"]) * sold_qty, 2),
        "pnl_pct": round((sell_price - holding["avg_buy_price"]) / holding["avg_buy_price"] * 100, 2),
        "notes": notes,
    }

    # Update or remove the holding
    remaining = holding["quantity"] - sold_qty
    if remaining <= 0:
        holdings.pop(idx)
    else:
        holdings[idx]["quantity"] = remaining

    # Append to trade history
    if "trade_history" not in data:
        data["trade_history"] = []
    data["trade_history"].append(trade)

    save_portfolio(data)
    return trade


def get_price_history(ticker: str) -> pd.DataFrame:
    """Fetch price history with caching. Returns OHLCV DataFrame.

    Delegates to ``utils.price_store`` so live discovery scoring and offline
    replay read from the same on-disk backing store.  Before this change the
    live path hit yfinance directly while replay read pickled cache files,
    producing systematic drift on price-derived factor scores
    (technical_score, turnover_cost_score, return_*_prior, momentum_*) when
    the two side data sources diverged.

    Behaviour: checks ``feature_cache/price_history/<TICKER>.pkl`` first;
    triggers a yfinance fetch only when the cached frame is missing or its
    most recent bar is more than ~3 days stale.  The fetch always writes
    back through to the on-disk cache so subsequent replay reads see the
    same series.  Failures return an empty frame, matching the prior contract.
    """
    ticker_key = str(ticker or "").upper().strip()
    yahoo_key = _resolve_yahoo_symbol(ticker_key)
    if ticker_key in _price_cache:
        return _price_cache[ticker_key]
    if yahoo_key in _price_cache:
        _price_cache[ticker_key] = _price_cache[yahoo_key]
        return _price_cache[yahoo_key]
    if _is_blocked_symbol(yahoo_key):
        df = pd.DataFrame()
        _price_cache[ticker_key] = df
        return df

    df = pd.DataFrame()
    try:
        # Local import to avoid a circular dependency at module load time.
        from utils import price_store

        today = pd.Timestamp.today().normalize()
        period_days = int(getattr(config, "PRICE_HISTORY_DAYS", 365))
        start = (today - pd.Timedelta(days=period_days)).date().isoformat()
        end = (today + pd.Timedelta(days=1)).date().isoformat()

        # ensure_price_history checks cache freshness inside download_price_history
        # and only triggers a yfinance batch when the cache is missing or stale.
        df = price_store.ensure_price_history(ticker_key, start=start, end=end)

        # Preserve the prior normalisation: drop trailing NaN-Close rows so
        # downstream technical analysis doesn't divide by NaN.
        if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
            close = _close_series(df)
            last_valid = close.last_valid_index()
            if last_valid is not None:
                df = df.loc[:last_valid]
            else:
                df = pd.DataFrame()
    except Exception as exc:
        logger.debug("get_price_history(%s) failed: %s", ticker_key, exc)
        df = pd.DataFrame()

    _price_cache[ticker_key] = df
    if yahoo_key and yahoo_key != ticker_key:
        _price_cache[yahoo_key] = df
    return df


def reset_ticker_info_stats() -> None:
    """Reset process-local Yahoo metadata counters."""
    with _info_stats_lock:
        for key in _info_stats:
            _info_stats[key] = 0


def get_ticker_info_stats() -> dict[str, int]:
    """Return process-local Yahoo metadata counters."""
    with _info_stats_lock:
        return dict(_info_stats)


def _record_info_stat(key: str) -> None:
    with _info_stats_lock:
        _info_stats[key] = int(_info_stats.get(key, 0)) + 1


def set_cached_ticker_info(ticker: str, info: dict | None) -> None:
    """Seed the session metadata cache without making a Yahoo request."""
    if not ticker or not isinstance(info, dict):
        return
    ticker_key = str(ticker).upper().strip()
    _info_cache[ticker_key] = dict(info)
    yahoo_key = _resolve_yahoo_symbol(ticker_key)
    if yahoo_key and yahoo_key != ticker_key:
        _info_cache[yahoo_key] = dict(info)


def get_ticker_info(ticker: str, timeout: int = 30, *, allow_network: bool | None = None) -> dict:
    """Fetch ticker info (fundamentals, name, etc.) with caching and timeout.

    Uses a thread pool to enforce a hard timeout on yfinance .info calls,
    which can hang indefinitely on delisted or problematic tickers.
    """
    ticker_key = str(ticker or "").upper().strip()
    yahoo_key = _resolve_yahoo_symbol(ticker_key)
    if ticker_key in _info_cache:
        _record_info_stat("cache")
        return _info_cache[ticker_key]
    if yahoo_key in _info_cache:
        _record_info_stat("cache")
        _info_cache[ticker_key] = _info_cache[yahoo_key]
        return _info_cache[yahoo_key]

    if _is_blocked_symbol(yahoo_key):
        _record_info_stat("skipped_cache_only")
        _info_cache[ticker_key] = {}
        return {}

    if allow_network is False:
        _record_info_stat("skipped_cache_only")
        return {}

    import concurrent.futures

    def _fetch():
        return yf.Ticker(yahoo_key).info or {}

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        info = pool.submit(_fetch).result(timeout=timeout)
        pool.shutdown(wait=False)
        if info:
            _record_info_stat("network")
        else:
            _record_info_stat("empty")
    except concurrent.futures.TimeoutError:
        pool.shutdown(wait=False)
        _record_info_stat("timeout")
        logging.getLogger(__name__).debug("Ticker info timeout for %s after %ds", ticker_key, timeout)
        info = {}
    except Exception as exc:
        pool.shutdown(wait=False)
        _record_info_stat("error")
        logging.getLogger(__name__).debug("Ticker info failed for %s: %s", ticker_key, exc)
        info = {}

    _info_cache[ticker_key] = info
    if yahoo_key and yahoo_key != ticker_key:
        _info_cache[yahoo_key] = info
    return info


def get_cached_ticker_info(ticker: str) -> dict:
    """Return cached ticker info without making a network call."""
    ticker_key = str(ticker or "").upper().strip()
    yahoo_key = _resolve_yahoo_symbol(ticker_key)
    return _info_cache.get(ticker_key) or _info_cache.get(yahoo_key, {})


def _normalise_price_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return a single-ticker OHLCV frame from yfinance's possible shapes."""
    if df.empty:
        return df

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        symbol = str(ticker or "").upper().strip()
        for level in range(out.columns.nlevels):
            raw_values = list(out.columns.get_level_values(level))
            upper_values = [str(v).upper().strip() for v in raw_values]
            if symbol and symbol in upper_values:
                matched = raw_values[upper_values.index(symbol)]
                out = out.xs(matched, axis=1, level=level, drop_level=True)
                break

        if isinstance(out.columns, pd.MultiIndex):
            field_names = {"OPEN", "HIGH", "LOW", "CLOSE", "ADJ CLOSE", "VOLUME"}
            for level in range(out.columns.nlevels):
                values = {str(v).upper().strip() for v in out.columns.get_level_values(level)}
                if values & field_names:
                    out.columns = out.columns.get_level_values(level)
                    break
            else:
                out.columns = out.columns.get_level_values(0)

    # If flattening still leaves duplicate OHLCV columns, keep the first
    # non-null value per row for each field. This avoids pandas returning a
    # Series/DataFrame when callers ask for df["Close"].
    if out.columns.has_duplicates:
        collapsed = {}
        for col in dict.fromkeys(out.columns):
            subset = out.loc[:, out.columns == col]
            if isinstance(subset, pd.DataFrame) and subset.shape[1] > 1:
                collapsed[col] = subset.bfill(axis=1).iloc[:, 0]
            else:
                collapsed[col] = subset.iloc[:, 0] if isinstance(subset, pd.DataFrame) else subset
        out = pd.DataFrame(collapsed, index=out.index)

    return out


def _scalar(val) -> float:
    """Extract a scalar float from a value that may be a pandas Series or scalar."""
    if isinstance(val, pd.DataFrame):
        values = pd.to_numeric(pd.Series(val.to_numpy().ravel()), errors="coerce").dropna()
        if values.empty:
            raise ValueError("no numeric scalar available")
        return float(values.iloc[0])
    if isinstance(val, pd.Series):
        values = pd.to_numeric(val, errors="coerce").dropna()
        if values.empty:
            raise ValueError("no numeric scalar available")
        return float(values.iloc[0])
    if hasattr(val, "item"):
        try:
            return float(val.item())
        except ValueError:
            values = pd.to_numeric(pd.Series(list(val)), errors="coerce").dropna()
            if values.empty:
                raise
            return float(values.iloc[0])
    return float(val)


def _close_series(df: pd.DataFrame) -> pd.Series:
    """Return a numeric Close series, tolerating duplicate Close columns."""
    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype="float64")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.apply(_scalar, axis=1)
    return pd.to_numeric(close, errors="coerce").dropna()


def get_current_price(ticker: str) -> float | None:
    """Get the most recent closing price."""
    df = get_price_history(ticker)
    if df.empty:
        return None
    close = _close_series(df)
    if close.empty:
        return None
    return _scalar(close.iloc[-1])


def get_daily_change(ticker: str) -> float | None:
    """Get the daily percentage change."""
    df = get_price_history(ticker)
    if df.empty or len(df) < 2:
        return None
    close = _close_series(df)
    if len(close) < 2:
        return None
    latest = _scalar(close.iloc[-1])
    previous = _scalar(close.iloc[-2])
    if previous == 0:
        return None
    return ((latest - previous) / previous) * 100


def get_macro_data() -> dict[str, pd.DataFrame]:
    """Fetch macro indicator histories (VIX, bonds, USD, oil) with caching."""
    if _macro_cache:
        return _macro_cache

    for name, ticker in config.MACRO_TICKERS.items():
        try:
            df = yf.download(ticker, period=f"{config.MACRO_LOOKBACK}d", progress=False, auto_adjust=True, timeout=30)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                _macro_cache[name] = df
        except Exception:
            pass

    return _macro_cache


def get_macro_regime_signals() -> dict:
    """Compute macro regime signals for factor timing (Arnott et al. 2019).

    Returns dict with:
    - term_spread: 10Y-2Y yield spread (expansion/contraction indicator)
    - credit_spread_ratio: HY vs IG bond return differential (risk appetite)
    - regime: "expansion", "contraction", or "neutral"
    - factor_tilts: recommended factor weight adjustments
    """
    extended_tickers = getattr(config, "MACRO_TICKERS_EXTENDED", {})
    if not extended_tickers:
        return {"regime": "neutral", "factor_tilts": {}}

    macro_prices: dict[str, float] = {}
    for name, ticker in extended_tickers.items():
        try:
            df = yf.download(ticker, period="5d", progress=False, auto_adjust=True, timeout=15)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                macro_prices[name] = _scalar(df["Close"].iloc[-1])
        except Exception:
            pass

    # Term structure: 10Y yield - 2Y yield proxy
    # ^TNX is in percentage (e.g., 4.5 = 4.5%), ^IRX is 13-week T-bill rate
    yield_10y = macro_prices.get("bonds_10y", 0)
    yield_2y = macro_prices.get("bonds_2y", 0)
    # ^IRX is quoted as yield * 10 (e.g., 45 = 4.5%) — normalize
    if yield_2y > 20:
        yield_2y = yield_2y / 10.0

    term_spread = yield_10y - yield_2y if yield_10y and yield_2y else None

    # Credit spread proxy: HYG vs LQD price ratio change (inverse of spread)
    hyg = macro_prices.get("hy_spread")
    lqd = macro_prices.get("ig_spread")
    credit_spread_ratio = (hyg / lqd) if hyg and lqd and lqd > 0 else None

    # Determine regime
    expansion_threshold = getattr(config, "MACRO_TERM_SPREAD_EXPANSION", 1.0)
    contraction_threshold = getattr(config, "MACRO_TERM_SPREAD_CONTRACTION", 0.0)

    if term_spread is not None and term_spread > expansion_threshold:
        regime = "expansion"
    elif term_spread is not None and term_spread < contraction_threshold:
        regime = "contraction"
    else:
        regime = "neutral"

    # Factor tilts based on regime (Arnott, Harvey, Kalesnik & Linnainmaa 2019)
    if regime == "expansion":
        factor_tilts = {
            "momentum_tilt": 0.10,      # Momentum works in trending markets
            "investment_tilt": 0.08,     # Low-investment firms outperform in expansions
            "quality_tilt": -0.05,       # Quality premium compressed
            "volatility_tilt": -0.05,    # Low-vol underperforms in risk-on
            "reversal_tilt": -0.03,      # Reversal weaker in trending markets
        }
    elif regime == "contraction":
        factor_tilts = {
            "momentum_tilt": -0.08,      # Momentum crashes in regime shifts
            "investment_tilt": -0.03,     # Less discriminating
            "quality_tilt": 0.12,        # Flight to quality
            "volatility_tilt": 0.10,     # Low-vol premium spikes
            "reversal_tilt": 0.05,       # Mean reversion stronger
        }
    else:
        factor_tilts = {}

    return {
        "term_spread": term_spread,
        "credit_spread_ratio": credit_spread_ratio,
        "yield_10y": yield_10y,
        "yield_2y": yield_2y,
        "regime": regime,
        "factor_tilts": factor_tilts,
    }


def get_reddit_posts(ticker: str) -> list[dict]:
    """Fetch recent Reddit posts mentioning this ticker with TTL caching."""
    import requests as req

    # Check TTL cache
    if ticker in _reddit_cache:
        posts, ts = _reddit_cache[ticker]
        if time.time() - ts < config.REDDIT_CACHE_TTL:
            return posts

    # Clean ticker for search (strip exchange suffixes)
    search_term = ticker.split(".")[0]
    all_posts = []

    headers = {"User-Agent": "ISADashboard/1.0"}

    for sub in config.REDDIT_SUBREDDITS:
        try:
            url = (
                f"https://www.reddit.com/r/{sub}/search.json"
                f"?q={search_term}&sort=new&limit={config.REDDIT_POST_LIMIT}&restrict_sr=on&t=week"
            )
            resp = req.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data", {}).get("children", [])
                for item in data:
                    post = item.get("data", {})
                    all_posts.append({
                        "title": post.get("title", ""),
                        "subreddit": sub,
                        "score": post.get("score", 0),
                        "num_comments": post.get("num_comments", 0),
                        "upvote_ratio": post.get("upvote_ratio", 0.5),
                    })
        except Exception:
            continue

    _reddit_cache[ticker] = (all_posts, time.time())
    return all_posts


def get_insider_transactions(ticker: str) -> dict:
    """Fetch recent insider transactions (buys/sells) from yfinance.

    Returns dict with buys, sells, net_label, and recent transaction list.
    """
    try:
        t = yf.Ticker(ticker)
        # yfinance provides insider_purchases (aggregated) and insider_transactions (detailed)
        txns = getattr(t, "insider_transactions", None)
        if txns is None or (hasattr(txns, "empty") and txns.empty):
            return {"buys": 0, "sells": 0, "net_label": "N/A", "recent": []}

        # Filter to recent transactions within lookback
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=config.INSIDER_LOOKBACK_DAYS)
        if "Start Date" in txns.columns:
            date_col = "Start Date"
        elif "Date" in txns.columns:
            date_col = "Date"
        else:
            # Can't determine dates, use all rows
            date_col = None

        if date_col:
            txns[date_col] = pd.to_datetime(txns[date_col], errors="coerce")
            txns = txns[txns[date_col] >= cutoff]

        buys = 0
        sells = 0
        recent = []

        # Look for transaction type column
        text_col = None
        for col_name in ["Text", "Transaction", "Type"]:
            if col_name in txns.columns:
                text_col = col_name
                break

        shares_col = None
        for col_name in ["Shares", "Number of Shares"]:
            if col_name in txns.columns:
                shares_col = col_name
                break

        insider_col = None
        for col_name in ["Insider", "Insider Trading", "Name"]:
            if col_name in txns.columns:
                insider_col = col_name
                break

        for _, row in txns.head(20).iterrows():
            text = str(row.get(text_col, "")).lower() if text_col else ""
            is_buy = "purchase" in text or "buy" in text or "acquisition" in text
            is_sell = "sale" in text or "sell" in text or "disposition" in text

            if is_buy:
                buys += 1
            elif is_sell:
                sells += 1

            recent.append({
                "insider": str(row[insider_col]) if insider_col else "Unknown",
                "type": "Buy" if is_buy else "Sell" if is_sell else text[:30],
                "shares": int(row[shares_col]) if shares_col and pd.notna(row.get(shares_col)) else 0,
                "date": str(row[date_col].date()) if date_col and pd.notna(row.get(date_col)) else "Unknown",
            })

        if buys > sells:
            net_label = "Net Buying"
        elif sells > buys:
            net_label = "Net Selling"
        elif buys == 0 and sells == 0:
            net_label = "N/A"
        else:
            net_label = "Mixed"

        return {"buys": buys, "sells": sells, "net_label": net_label, "recent": recent[:10]}
    except Exception:
        return {"buys": 0, "sells": 0, "net_label": "N/A", "recent": []}


def clear_cache():
    """Clear all cached data (useful for manual refresh)."""
    _price_cache.clear()
    _info_cache.clear()
    _macro_cache.clear()
    _reddit_cache.clear()
