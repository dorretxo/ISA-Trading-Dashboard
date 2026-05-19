"""Cached historical OHLCV price store.

The replay and label engines need price history repeatedly. This module keeps
that IO batched and local: one yfinance download per ticker batch, then all
future slices come from ``feature_cache/price_history``.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)

PRICE_CACHE_DIR = Path(getattr(config, "HISTORICAL_PRICE_CACHE_DIR", "feature_cache/price_history"))
_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


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


def _safe_ticker_key(ticker: str) -> str:
    key = str(ticker or "").upper().strip()
    return re.sub(r"[^A-Z0-9._=-]+", "_", key)


def _cache_path(ticker: str, cache_dir: Path = PRICE_CACHE_DIR) -> Path:
    return cache_dir / f"{_safe_ticker_key(_resolve_yahoo_symbol(ticker) or ticker)}.pkl"


def _coerce_ts(value) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        return pd.Timestamp(value).tz_localize(None).normalize()
    except Exception:
        return None


def _normalize_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out.index = out.index.tz_localize(None).normalize()
    else:
        out.index = pd.to_datetime(out.index, errors="coerce").tz_localize(None).normalize()
    out = out[~out.index.isna()]
    keep = [c for c in _PRICE_COLUMNS if c in out.columns]
    out = out[keep]
    return out.sort_index()


def load_price_history(ticker: str, *, cache_dir: Path = PRICE_CACHE_DIR) -> pd.DataFrame:
    symbol = _resolve_yahoo_symbol(ticker)
    if _is_blocked_symbol(symbol):
        return pd.DataFrame()
    path = _cache_path(ticker, cache_dir)
    if not path.exists():
        return pd.DataFrame()
    try:
        return _normalize_frame(pd.read_pickle(path))
    except Exception as exc:
        logger.debug("Price cache read failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def save_price_history(ticker: str, frame: pd.DataFrame, *, cache_dir: Path = PRICE_CACHE_DIR) -> None:
    symbol = _resolve_yahoo_symbol(ticker)
    if _is_blocked_symbol(symbol):
        return
    frame = _normalize_frame(frame)
    if frame.empty:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(_cache_path(ticker, cache_dir))


def _merge_frames(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        return _normalize_frame(new)
    if new is None or new.empty:
        return _normalize_frame(old)
    merged = pd.concat([_normalize_frame(old), _normalize_frame(new)])
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged.sort_index()


def get_price_history(
    ticker: str,
    *,
    start: str | date | datetime | None = None,
    end: str | date | datetime | None = None,
    cache_dir: Path = PRICE_CACHE_DIR,
) -> pd.DataFrame:
    frame = load_price_history(ticker, cache_dir=cache_dir)
    if frame.empty:
        return frame
    start_ts = _coerce_ts(start)
    end_ts = _coerce_ts(end)
    if start_ts is not None:
        frame = frame.loc[frame.index >= start_ts]
    if end_ts is not None:
        frame = frame.loc[frame.index <= end_ts]
    return frame


def _extract_download_frame(data: pd.DataFrame, ticker: str, requested: list[str]) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    if not isinstance(data.columns, pd.MultiIndex):
        return _normalize_frame(data)

    ticker_key = str(ticker).upper()
    levels = data.columns.names
    # group_by="ticker" gives columns like (AAPL, Close). The default gives
    # (Close, AAPL). Support both so tests and yfinance versions stay stable.
    first_level = [str(v).upper() for v in data.columns.get_level_values(0).unique()]
    second_level = [str(v).upper() for v in data.columns.get_level_values(1).unique()]
    if ticker_key in first_level:
        try:
            return _normalize_frame(data.xs(ticker, axis=1, level=0, drop_level=True))
        except Exception:
            pass
    if ticker_key in second_level:
        try:
            return _normalize_frame(data.xs(ticker, axis=1, level=1, drop_level=True))
        except Exception:
            pass
    # yfinance may rewrite dots/hyphens in rare cases. Fall back to requested
    # order when a batch contains exactly one ticker.
    if len(requested) == 1:
        try:
            return _normalize_frame(data.droplevel(0, axis=1))
        except Exception:
            return pd.DataFrame()
    logger.debug("Could not split downloaded frame for %s (levels=%s)", ticker, levels)
    return pd.DataFrame()


def download_price_history(
    tickers: Iterable[str],
    *,
    start: str | date | datetime,
    end: str | date | datetime | None = None,
    batch_size: int | None = None,
    force: bool = False,
    cache_dir: Path = PRICE_CACHE_DIR,
) -> dict[str, pd.DataFrame]:
    """Download/update cached OHLCV history and return frames by ticker."""
    requested = sorted({str(t).upper().strip() for t in tickers if str(t or "").strip()})
    if not requested:
        return {}
    requests_by_symbol: dict[str, list[str]] = {}
    skipped = 0
    for original in requested:
        symbol = _resolve_yahoo_symbol(original)
        if not symbol or _is_blocked_symbol(symbol):
            skipped += 1
            continue
        requests_by_symbol.setdefault(symbol, []).append(original)
    if skipped:
        logger.info("Price cache skipped %d excluded/quarantined tickers before Yahoo download", skipped)
    clean = sorted(requests_by_symbol)
    if not clean:
        return {}
    batch_size = int(batch_size or getattr(config, "PRICE_CACHE_BATCH_SIZE", 75))
    batch_size = max(1, batch_size)
    start_ts = _coerce_ts(start) or (pd.Timestamp.today().normalize() - pd.Timedelta(days=3650))
    end_ts = _coerce_ts(end) or (pd.Timestamp.today().normalize() + pd.Timedelta(days=1))

    results: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for ticker in clean:
        cached = load_price_history(ticker, cache_dir=cache_dir)
        if not force and not cached.empty and cached.index.min() <= start_ts and cached.index.max() >= end_ts - pd.Timedelta(days=3):
            frame = get_price_history(ticker, start=start_ts, end=end_ts, cache_dir=cache_dir)
            for original in requests_by_symbol.get(ticker, [ticker]):
                results[original] = frame
        else:
            if not cached.empty:
                frame = get_price_history(ticker, start=start_ts, end=end_ts, cache_dir=cache_dir)
                for original in requests_by_symbol.get(ticker, [ticker]):
                    results[original] = frame
            missing.append(ticker)

    for offset in range(0, len(missing), batch_size):
        batch = missing[offset:offset + batch_size]
        try:
            data = yf.download(
                batch,
                start=start_ts.date().isoformat(),
                end=(end_ts + pd.Timedelta(days=1)).date().isoformat(),
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as exc:
            logger.warning("Price cache download failed for batch %s: %s", batch[:5], exc)
            continue
        for ticker in batch:
            downloaded = _extract_download_frame(data, ticker, batch)
            if downloaded.empty:
                continue
            merged = _merge_frames(load_price_history(ticker, cache_dir=cache_dir), downloaded)
            save_price_history(ticker, merged, cache_dir=cache_dir)
            frame = get_price_history(ticker, start=start_ts, end=end_ts, cache_dir=cache_dir)
            for original in requests_by_symbol.get(ticker, [ticker]):
                results[original] = frame
        logger.info("Price cache: processed %d-%d/%d", offset + 1, min(offset + batch_size, len(missing)), len(missing))

    return results


def ensure_price_history(
    ticker: str,
    *,
    start: str | date | datetime,
    end: str | date | datetime | None = None,
    force: bool = False,
) -> pd.DataFrame:
    frames = download_price_history([ticker], start=start, end=end, force=force)
    return frames.get(str(ticker).upper().strip(), pd.DataFrame())
