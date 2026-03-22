"""Shared data fetching utilities with session-level caching."""

import json
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

import config

# Session-level cache for yfinance data
_price_cache: dict[str, pd.DataFrame] = {}
_info_cache: dict[str, dict] = {}
_macro_cache: dict[str, pd.DataFrame] = {}
_reddit_cache: dict[str, tuple[list, float]] = {}  # {ticker: (posts, timestamp)}


def _portfolio_path() -> Path:
    return Path(__file__).parent.parent / config.PORTFOLIO_FILE


def load_portfolio() -> list[dict]:
    """Load portfolio holdings from JSON file."""
    with open(_portfolio_path(), "r") as f:
        data = json.load(f)
    return data["holdings"]


def load_portfolio_full() -> dict:
    """Load the full portfolio file including trade_history."""
    with open(_portfolio_path(), "r") as f:
        return json.load(f)


def save_portfolio(data: dict) -> None:
    """Atomic write of portfolio data."""
    path = _portfolio_path()
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)
    tmp.replace(path)


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
    """Fetch price history with caching. Returns OHLCV DataFrame."""
    if ticker in _price_cache:
        return _price_cache[ticker]

    period = f"{config.PRICE_HISTORY_DAYS}d"
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            _price_cache[ticker] = pd.DataFrame()
            return pd.DataFrame()
        # Flatten multi-level columns if present (yfinance sometimes returns them)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        _price_cache[ticker] = df
    except Exception:
        _price_cache[ticker] = pd.DataFrame()

    return _price_cache[ticker]


def get_ticker_info(ticker: str) -> dict:
    """Fetch ticker info (fundamentals, name, etc.) with caching."""
    if ticker in _info_cache:
        return _info_cache[ticker]

    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}

    _info_cache[ticker] = info
    return info


def get_current_price(ticker: str) -> float | None:
    """Get the most recent closing price."""
    df = get_price_history(ticker)
    if df.empty:
        return None
    return float(df["Close"].iloc[-1])


def get_daily_change(ticker: str) -> float | None:
    """Get the daily percentage change."""
    df = get_price_history(ticker)
    if df.empty or len(df) < 2:
        return None
    latest = float(df["Close"].iloc[-1])
    previous = float(df["Close"].iloc[-2])
    if previous == 0:
        return None
    return ((latest - previous) / previous) * 100


def get_macro_data() -> dict[str, pd.DataFrame]:
    """Fetch macro indicator histories (VIX, bonds, USD, oil) with caching."""
    if _macro_cache:
        return _macro_cache

    for name, ticker in config.MACRO_TICKERS.items():
        try:
            df = yf.download(ticker, period=f"{config.MACRO_LOOKBACK}d", progress=False, auto_adjust=True)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                _macro_cache[name] = df
        except Exception:
            pass

    return _macro_cache


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
