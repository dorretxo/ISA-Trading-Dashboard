"""Feature Store — daily cache for batch-computed price factors.

Stores per-ticker technical factors computed from bulk price downloads.
Designed to be the "cheap batch factor layer" in a tiered discovery funnel:

  Universe (2700) → Feature Store (batch price factors for all)
                  → Momentum ranking (top 150 from cached factors)
                  → Deep analysis (top 30-50 only)

The store is sharded by date so partial refreshes and historical lookback
are straightforward. The interface is abstract enough to swap JSON for
Parquet later if needed.

Usage:
    store = FeatureStore()
    store.load()

    # Check if today's factors are fresh
    if not store.is_fresh(ticker, max_age_hours=24):
        # Compute and save
        store.put(ticker, {"ret_90d": 0.15, "ret_30d": 0.05, ...})

    # Retrieve
    factors = store.get(ticker)

    # Batch operations
    stale = store.get_stale_tickers(all_tickers, max_age_hours=24)
    store.save()
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TickerFeatures:
    """Batch-computed price factors for a single ticker."""
    ticker: str
    timestamp: str  # ISO format

    # Returns
    ret_90d: float = 0.0
    ret_30d: float = 0.0
    ret_10d: float = 0.0

    # Volatility
    vol_20d: float = 0.0      # 20-day realised vol (annualised)
    vol_60d: float = 0.0      # 60-day realised vol (annualised)

    # Trend / MA
    above_sma50: bool = False
    above_sma200: bool = False
    sma50_slope: float = 0.0   # 5-day change in SMA-50 (normalised)

    # Volume / liquidity
    volume_ratio: float = 1.0  # 10d avg / 60d avg
    avg_volume_20d: float = 0  # 20-day average daily volume
    avg_dollar_volume: float = 0  # avg_volume * last_price

    # Distance from extremes
    pct_from_high_252d: float = 0.0  # price / 252d high
    pct_from_low_252d: float = 0.0   # price / 252d low

    # Beta (computed vs SPY if available)
    beta_90d: float = 1.0

    # Price data
    last_price: float = 0.0

    # 60-day daily returns (for correlation computation downstream)
    # Stored as list[float] — not displayed, used by correlation filter
    returns_60d: list = field(default_factory=list)

    # 90-day daily returns (aligned with 90-day holding period for correlation)
    returns_90d: list = field(default_factory=list)

    # Sector relative strength
    sector: str = ""
    sector_median_90d: float = 0.0
    relative_strength: float = 0.0  # ret_90d - sector_median


# ---------------------------------------------------------------------------
# Feature Store
# ---------------------------------------------------------------------------

_STORE_DIR = "feature_cache"
_STORE_VERSION = 1


class FeatureStore:
    """Daily cache for batch-computed price factors.

    Storage: one JSON file per date in feature_cache/ directory.
    Each file maps ticker -> TickerFeatures dict.

    Interface is deliberately simple so the backing format can be swapped
    to Parquet or SQLite later without changing callers.
    """

    def __init__(self, base_dir: Path | str | None = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent / _STORE_DIR
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = {}  # ticker -> features dict
        self._dirty = False
        self._loaded_date: str | None = None

    def _date_file(self, date_str: str | None = None) -> Path:
        """Path to the store file for a given date."""
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._base_dir / f"features_{date_str}.json"

    def load(self, date_str: str | None = None) -> int:
        """Load features for a given date. Returns number of tickers loaded."""
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        path = self._date_file(date_str)
        if path.exists():
            try:
                with open(path, "r") as f:
                    raw = json.load(f)
                self._data = raw.get("features", {})
                self._loaded_date = date_str
                logger.info("Feature store loaded: %d tickers for %s", len(self._data), date_str)
                return len(self._data)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Feature store corrupt for %s: %s", date_str, e)
                self._data = {}
        else:
            self._data = {}
            self._loaded_date = date_str
        return 0

    def save(self) -> None:
        """Persist current features to disk."""
        if not self._dirty:
            return

        date_str = self._loaded_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self._date_file(date_str)

        payload = {
            "version": _STORE_VERSION,
            "date": date_str,
            "ticker_count": len(self._data),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "features": self._data,
        }

        # Atomic write via temp file
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)

        self._dirty = False
        logger.info("Feature store saved: %d tickers for %s", len(self._data), date_str)

    def put(self, ticker: str, features: TickerFeatures | dict) -> None:
        """Store features for a ticker."""
        if isinstance(features, TickerFeatures):
            d = asdict(features)
            # Convert numpy arrays to lists for JSON
            for _rk in ("returns_60d", "returns_90d"):
                if isinstance(d.get(_rk), np.ndarray):
                    d[_rk] = d[_rk].tolist()
        else:
            d = features
            for _rk in ("returns_60d", "returns_90d"):
                if isinstance(d.get(_rk), np.ndarray):
                    d[_rk] = d[_rk].tolist()

        self._data[ticker] = d
        self._dirty = True

    def put_batch(self, features_map: dict[str, TickerFeatures | dict]) -> None:
        """Store features for multiple tickers at once."""
        for ticker, features in features_map.items():
            self.put(ticker, features)

    def get(self, ticker: str) -> dict | None:
        """Retrieve features for a ticker. Returns None if not cached."""
        return self._data.get(ticker)

    def get_many(self, tickers: list[str]) -> dict[str, dict]:
        """Retrieve features for multiple tickers. Skips missing ones."""
        return {t: self._data[t] for t in tickers if t in self._data}

    def has(self, ticker: str) -> bool:
        """Check if a ticker has cached features."""
        return ticker in self._data

    def is_fresh(self, ticker: str, max_age_hours: float = 24) -> bool:
        """Check if a ticker's features are fresh enough."""
        d = self._data.get(ticker)
        if not d:
            return False
        try:
            ts = datetime.fromisoformat(d["timestamp"])
            age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
            return age_hours <= max_age_hours
        except (KeyError, ValueError):
            return False

    def get_stale_tickers(self, tickers: list[str], max_age_hours: float = 24) -> list[str]:
        """Return tickers that need a refresh."""
        return [t for t in tickers if not self.is_fresh(t, max_age_hours)]

    def get_fresh_tickers(self, tickers: list[str], max_age_hours: float = 24) -> list[str]:
        """Return tickers that have fresh cached features."""
        return [t for t in tickers if self.is_fresh(t, max_age_hours)]

    def all_tickers(self) -> list[str]:
        """Return all cached ticker symbols."""
        return list(self._data.keys())

    def count(self) -> int:
        """Number of tickers in store."""
        return len(self._data)

    def cleanup_old(self, keep_days: int = 7) -> int:
        """Remove store files older than keep_days. Returns count removed."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
        removed = 0
        for f in self._base_dir.glob("features_*.json"):
            try:
                date_str = f.stem.replace("features_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if file_date < cutoff:
                    f.unlink()
                    removed += 1
            except (ValueError, OSError):
                continue
        if removed:
            logger.info("Cleaned up %d old feature store files", removed)
        return removed


# ---------------------------------------------------------------------------
# Batch factor computation
# ---------------------------------------------------------------------------

def compute_batch_factors(
    tickers: list[str],
    batch_size: int = 80,
    spy_returns: np.ndarray | None = None,
    sector_map: dict[str, str] | None = None,
    progress_callback=None,
) -> dict[str, TickerFeatures]:
    """Compute price-derived factors for a list of tickers using bulk downloads.

    This is the "cheap batch factor layer" — only uses yfinance bulk price data,
    no per-ticker API calls. Designed to run on 600-800 tickers in ~2 minutes.

    Args:
        tickers: List of ticker symbols
        batch_size: Download batch size (default 80, matching discovery.py)
        spy_returns: Pre-computed SPY returns for beta calculation
        sector_map: {ticker: sector} for relative strength
        progress_callback: Optional (message, current, total) callback

    Returns:
        {ticker: TickerFeatures} for all successfully computed tickers
    """
    import yfinance as yf

    now_iso = datetime.now(timezone.utc).isoformat()
    results = {}

    # Download SPY for beta computation if not provided
    if spy_returns is None:
        try:
            spy_data = yf.download("SPY", period="120d", progress=False, auto_adjust=True)
            if spy_data is not None and len(spy_data) >= 30:
                spy_close = spy_data["Close"]
                # yfinance may return DataFrame even for single ticker
                if isinstance(spy_close, pd.DataFrame):
                    spy_close = spy_close.iloc[:, 0]
                spy_returns = spy_close.pct_change().dropna().values[-60:]
        except Exception:
            spy_returns = None

    for batch_start in range(0, len(tickers), batch_size):
        batch = tickers[batch_start:batch_start + batch_size]
        if progress_callback:
            progress_callback(
                f"Computing factors... ({batch_start}/{len(tickers)})",
                batch_start, len(tickers),
            )

        try:
            data = yf.download(
                batch, period="270d", progress=False, auto_adjust=True,
                threads=True, timeout=60,
            )
            if data is None or data.empty:
                continue

            for sym in batch:
                try:
                    features = _extract_ticker_features(
                        data, sym, len(batch), now_iso, spy_returns,
                    )
                    if features:
                        results[sym] = features
                except Exception:
                    continue

        except Exception as e:
            logger.warning("Batch factor download failed (%d-%d): %s",
                           batch_start, batch_start + batch_size, e)
            continue

    # Compute sector relative strength
    if sector_map:
        _compute_sector_relative_strength(results, sector_map)

    logger.info("Computed batch factors for %d / %d tickers", len(results), len(tickers))
    return results


def _extract_ticker_features(
    data: pd.DataFrame,
    ticker: str,
    batch_len: int,
    timestamp: str,
    spy_returns: np.ndarray | None,
) -> TickerFeatures | None:
    """Extract all price-derived features for a single ticker from bulk download data."""
    try:
        # Extract close prices — yfinance always returns MultiIndex columns
        close_data = data["Close"]
        if isinstance(close_data, pd.DataFrame):
            if ticker in close_data.columns:
                closes = close_data[ticker]
            else:
                return None
        else:
            closes = close_data  # Series fallback

        volumes = None
        try:
            vol_data = data["Volume"]
            if isinstance(vol_data, pd.DataFrame) and ticker in vol_data.columns:
                volumes = vol_data[ticker]
            elif isinstance(vol_data, pd.Series):
                volumes = vol_data
        except (KeyError, TypeError):
            pass

        closes = closes.dropna()
        if len(closes) < 30:
            return None

        values = closes.values
        n = len(values)

        # Returns
        ret_90d = (values[-1] / values[-min(90, n)] - 1) if n >= 20 else 0
        ret_30d = (values[-1] / values[-min(30, n)] - 1) if n >= 15 else 0
        ret_10d = (values[-1] / values[-min(10, n)] - 1) if n >= 10 else 0

        # Daily returns
        daily_returns = pd.Series(closes).pct_change().dropna().values
        returns_60d = daily_returns[-60:] if len(daily_returns) >= 15 else []
        returns_90d = daily_returns[-90:] if len(daily_returns) >= 15 else []

        # Volatility (annualised)
        vol_20d = float(np.std(daily_returns[-20:]) * np.sqrt(252)) if len(daily_returns) >= 20 else 0
        vol_60d = float(np.std(daily_returns[-60:]) * np.sqrt(252)) if len(daily_returns) >= 60 else vol_20d

        # Moving averages
        sma_50 = np.mean(values[-min(50, n):])
        sma_200 = np.mean(values[-min(200, n):]) if n >= 100 else sma_50
        above_sma50 = bool(values[-1] > sma_50)
        above_sma200 = bool(values[-1] > sma_200)

        # SMA-50 slope (normalised 5-day change)
        if n >= 55:
            sma50_now = np.mean(values[-50:])
            sma50_5ago = np.mean(values[-55:-5])
            sma50_slope = (sma50_now - sma50_5ago) / sma50_5ago if sma50_5ago > 0 else 0
        else:
            sma50_slope = 0

        # Volume
        volume_ratio = 1.0
        avg_volume_20d = 0.0
        if volumes is not None:
            vol_vals = volumes.dropna().values
            if len(vol_vals) >= 20:
                avg_vol_10 = np.mean(vol_vals[-10:]) if len(vol_vals) >= 10 else 0
                avg_vol_60 = np.mean(vol_vals[-60:]) if len(vol_vals) >= 60 else np.mean(vol_vals)
                volume_ratio = avg_vol_10 / max(avg_vol_60, 1)
                avg_volume_20d = float(np.mean(vol_vals[-20:]))

        # Distance from extremes
        high_252 = np.max(values[-min(252, n):])
        low_252 = np.min(values[-min(252, n):])
        pct_from_high = values[-1] / high_252 if high_252 > 0 else 0
        pct_from_low = values[-1] / low_252 if low_252 > 0 else 0

        # Beta vs SPY
        beta = 1.0
        if spy_returns is not None and len(returns_60d) >= 30:
            common_len = min(len(returns_60d), len(spy_returns))
            if common_len >= 30:
                stock_r = returns_60d[-common_len:]
                spy_r = spy_returns[-common_len:]
                cov = np.cov(stock_r, spy_r)
                if cov[1, 1] > 0:
                    beta = float(cov[0, 1] / cov[1, 1])

        last_price = float(values[-1])

        return TickerFeatures(
            ticker=ticker,
            timestamp=timestamp,
            ret_90d=float(ret_90d),
            ret_30d=float(ret_30d),
            ret_10d=float(ret_10d),
            vol_20d=vol_20d,
            vol_60d=vol_60d,
            above_sma50=above_sma50,
            above_sma200=above_sma200,
            sma50_slope=float(sma50_slope),
            volume_ratio=float(volume_ratio),
            avg_volume_20d=avg_volume_20d,
            avg_dollar_volume=avg_volume_20d * last_price,
            pct_from_high_252d=float(pct_from_high),
            pct_from_low_252d=float(pct_from_low),
            beta_90d=beta,
            last_price=last_price,
            returns_60d=list(float(r) for r in returns_60d),
            returns_90d=list(float(r) for r in returns_90d),
        )

    except Exception:
        return None


def _compute_sector_relative_strength(
    results: dict[str, TickerFeatures],
    sector_map: dict[str, str],
) -> None:
    """In-place update of sector relative strength fields."""
    # Group by sector
    sector_returns: dict[str, list[float]] = {}
    for ticker, feat in results.items():
        sector = sector_map.get(ticker, "Unknown")
        feat.sector = sector
        sector_returns.setdefault(sector, []).append(feat.ret_90d)

    # Compute medians
    sector_medians = {
        s: float(np.median(rets)) if rets else 0
        for s, rets in sector_returns.items()
    }

    # Update features
    for ticker, feat in results.items():
        median = sector_medians.get(feat.sector, 0)
        feat.sector_median_90d = median
        feat.relative_strength = feat.ret_90d - median
