"""Benchmark-driven index constituents.

Fetches current constituents for major global benchmarks from Wikipedia and
caches them on disk. Output feeds `reconstitute_universe()` as `extra_tickers`
so that new index entrants graduate into the dynamic universe through the
same liquidity/market-cap gates that guard ETF-decomposed tickers.

Design goals
------------
- Per-benchmark isolation: one failed scrape must not block others.
- Long TTL (default 30 days): indices rebalance quarterly, Wikipedia edits
  are the main source of ticker noise we want to avoid chasing.
- No new runtime dependencies: uses pandas.read_html (lxml already required).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

_logger = logging.getLogger(__name__)

# Wikipedia blocks the default urllib/pandas User-Agent with 403. A realistic
# browser UA is the standard workaround and matches the Wikipedia API guidance
# for low-volume research scrapers.
_USER_AGENT = (
    "Mozilla/5.0 (compatible; TradingDashboard/1.0; +https://local/research) "
    "pandas-read-html"
)

_CACHE_PATH = Path(__file__).parent.parent / "feature_cache" / "benchmark_constituents.json"
_DEFAULT_TTL_DAYS = 30


@dataclass(frozen=True)
class BenchmarkSpec:
    """Wikipedia-derived specification for one index."""

    index: str               # stable key, e.g. "FTSE100"
    wiki_url: str
    # Candidate column names to try for the ticker/symbol. First match wins.
    symbol_columns: tuple[str, ...]
    # Optional suffix to append if the Wikipedia symbol is the local code
    # without a yfinance suffix (e.g. "SHEL" → "SHEL.L").
    yf_suffix: str = ""
    country: str = ""
    # Table index to read when a page has multiple tables. -1 = auto-pick
    # the first table containing any of the candidate symbol columns.
    table_index: int = -1


_SPECS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        index="SP500",
        wiki_url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        symbol_columns=("Symbol", "Ticker symbol"),
        yf_suffix="",
        country="US",
    ),
    BenchmarkSpec(
        index="SPMIDCAP400",
        wiki_url="https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        symbol_columns=("Symbol", "Ticker symbol", "Ticker"),
        yf_suffix="",
        country="US",
    ),
    BenchmarkSpec(
        index="FTSE100",
        wiki_url="https://en.wikipedia.org/wiki/FTSE_100_Index",
        symbol_columns=("EPIC", "Ticker", "Symbol"),
        yf_suffix=".L",
        country="GB",
    ),
    BenchmarkSpec(
        index="FTSE250",
        wiki_url="https://en.wikipedia.org/wiki/FTSE_250_Index",
        symbol_columns=("EPIC", "Ticker", "Symbol"),
        yf_suffix=".L",
        country="GB",
    ),
    BenchmarkSpec(
        index="CAC40",
        wiki_url="https://en.wikipedia.org/wiki/CAC_40",
        symbol_columns=("Ticker", "Symbol"),
        yf_suffix=".PA",
        country="FR",
    ),
    BenchmarkSpec(
        index="DAX",
        wiki_url="https://en.wikipedia.org/wiki/DAX",
        symbol_columns=("Ticker", "Symbol"),
        yf_suffix=".DE",
        country="DE",
    ),
    BenchmarkSpec(
        index="IBEX35",
        wiki_url="https://en.wikipedia.org/wiki/IBEX_35",
        symbol_columns=("Ticker", "Symbol"),
        yf_suffix=".MC",
        country="ES",
    ),
    BenchmarkSpec(
        index="FTSE_MIB",
        wiki_url="https://en.wikipedia.org/wiki/FTSE_MIB",
        symbol_columns=("Ticker", "Symbol", "ISIN"),
        yf_suffix=".MI",
        country="IT",
    ),
    BenchmarkSpec(
        index="AEX",
        wiki_url="https://en.wikipedia.org/wiki/AEX_index",
        symbol_columns=("Ticker symbol", "Ticker", "Symbol"),
        yf_suffix=".AS",
        country="NL",
    ),
    BenchmarkSpec(
        index="SMI",
        wiki_url="https://en.wikipedia.org/wiki/Swiss_Market_Index",
        symbol_columns=("Ticker", "Symbol"),
        yf_suffix=".SW",
        country="CH",
    ),
    BenchmarkSpec(
        index="TSX60",
        wiki_url="https://en.wikipedia.org/wiki/S%26P/TSX_60",
        symbol_columns=("Symbol", "Ticker"),
        yf_suffix=".TO",
        country="CA",
    ),
    BenchmarkSpec(
        index="ASX200",
        wiki_url="https://en.wikipedia.org/wiki/S%26P/ASX_200",
        symbol_columns=("Code", "Ticker", "Symbol"),
        yf_suffix=".AX",
        country="AU",
    ),
    BenchmarkSpec(
        index="NIKKEI225",
        wiki_url="https://en.wikipedia.org/wiki/Nikkei_225",
        symbol_columns=("Code", "Ticker", "Symbol"),
        yf_suffix=".T",
        country="JP",
    ),
)


def _normalize_ticker(raw: str, suffix: str) -> str:
    """Clean a Wikipedia ticker cell into a yfinance-compatible symbol.

    - For US (suffix==""): "BRK.B" → "BRK-B" (yfinance convention).
    - For non-US: strip any pre-existing exchange suffix before appending
      ours, so "AIR.PA" + suffix=".DE" resolves to "AIR.DE" (Wikipedia
      sometimes carries cross-listings in the ticker cell).
    """
    s = str(raw).strip().upper()
    for bad in ("[", "(", " "):
        if bad in s:
            s = s.split(bad, 1)[0]

    if not suffix:
        if "." in s:
            s = s.replace(".", "-")  # BRK.B → BRK-B
        return s

    # Suffix path: if any "." remains in the symbol, treat the first
    # segment as the local code and drop the rest so we don't double-suffix.
    if "." in s:
        s = s.split(".", 1)[0]
    if not s.endswith(suffix):
        s = f"{s}{suffix}"
    return s


def _pick_table(tables: list[pd.DataFrame], symbol_cols: tuple[str, ...]) -> pd.DataFrame | None:
    for tbl in tables:
        cols = {str(c) for c in tbl.columns}
        if any(c in cols for c in symbol_cols):
            return tbl
    return None


def _fetch_one(spec: BenchmarkSpec, timeout: int = 20) -> list[str]:
    """Fetch a single benchmark's constituents. Returns [] on any failure."""
    try:
        resp = requests.get(
            spec.wiki_url,
            headers={"User-Agent": _USER_AGENT, "Accept-Language": "en"},
            timeout=timeout,
        )
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text), flavor="lxml")
    except Exception as e:
        _logger.warning("[%s] fetch/read_html failed: %s", spec.index, e)
        return []

    if not tables:
        _logger.warning("[%s] no tables parsed from %s", spec.index, spec.wiki_url)
        return []

    tbl = None
    if 0 <= spec.table_index < len(tables):
        tbl = tables[spec.table_index]
        cols = {str(c) for c in tbl.columns}
        if not any(c in cols for c in spec.symbol_columns):
            tbl = _pick_table(tables, spec.symbol_columns)
    else:
        tbl = _pick_table(tables, spec.symbol_columns)

    if tbl is None:
        _logger.warning("[%s] no table matched symbol columns %s",
                        spec.index, spec.symbol_columns)
        return []

    col = next((c for c in spec.symbol_columns if c in tbl.columns), None)
    if col is None:
        return []

    symbols: list[str] = []
    for raw in tbl[col].dropna().astype(str):
        tidy = _normalize_ticker(raw, spec.yf_suffix)
        if tidy and len(tidy) <= 12:
            symbols.append(tidy)

    # Dedupe while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    _logger.info("[%s] %d constituents fetched", spec.index, len(out))
    return out


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"benchmarks": {}}


def _save_cache(data: dict) -> None:
    try:
        from utils.atomic_io import atomic_write_json
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(_CACHE_PATH, data, indent=2)
    except Exception as e:
        _logger.debug("benchmark cache write failed: %s", e)


def _is_fresh(entry: dict, ttl_days: int) -> bool:
    ts = entry.get("refreshed_at")
    if not ts:
        return False
    try:
        saved = datetime.fromisoformat(ts)
    except ValueError:
        return False
    return datetime.now() - saved < timedelta(days=ttl_days)


def get_benchmark_tickers(
    ttl_days: int = _DEFAULT_TTL_DAYS,
    force_refresh: bool = False,
) -> list[str]:
    """Return a deduplicated ticker list across all supported benchmarks.

    Uses a per-benchmark cache with `ttl_days` freshness. Failed fetches
    fall back to the last-known good entry for that benchmark if present.
    """
    cache = _load_cache()
    benchmarks: dict = cache.get("benchmarks", {})
    updated = False

    for spec in _SPECS:
        entry = benchmarks.get(spec.index, {})
        if not force_refresh and _is_fresh(entry, ttl_days):
            continue
        fetched = _fetch_one(spec)
        if fetched:
            benchmarks[spec.index] = {
                "refreshed_at": datetime.now().isoformat(),
                "tickers": fetched,
                "country": spec.country,
            }
            updated = True
        elif entry.get("tickers"):
            _logger.info("[%s] fetch empty, reusing cached %d tickers",
                         spec.index, len(entry.get("tickers", [])))

    if updated:
        cache["benchmarks"] = benchmarks
        cache["refreshed_at"] = datetime.now().isoformat()
        _save_cache(cache)

    # Flatten + dedupe
    seen: set[str] = set()
    out: list[str] = []
    for entry in benchmarks.values():
        for t in entry.get("tickers", []):
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def get_supported_benchmark_keys() -> list[str]:
    """Return the benchmark keys the current code expects in the cache."""
    return [spec.index for spec in _SPECS]


def get_benchmark_cache_metadata() -> dict:
    """Return cache freshness metadata used by orchestrator rebuild gating."""
    cache = _load_cache()
    benchmarks = cache.get("benchmarks", {}) or {}
    missing = [
        spec.index
        for spec in _SPECS
        if not (benchmarks.get(spec.index) or {}).get("tickers")
    ]
    latest_refreshed_at = None
    for entry in benchmarks.values():
        refreshed_at = entry.get("refreshed_at")
        if not refreshed_at:
            continue
        try:
            parsed = datetime.fromisoformat(refreshed_at)
        except ValueError:
            continue
        if latest_refreshed_at is None or parsed > latest_refreshed_at:
            latest_refreshed_at = parsed
    return {
        "path": str(_CACHE_PATH),
        "supported": get_supported_benchmark_keys(),
        "missing": missing,
        "latest_refreshed_at": latest_refreshed_at.isoformat() if latest_refreshed_at else None,
    }


def get_benchmark_summary() -> dict:
    """Return cached per-benchmark freshness and counts (no network)."""
    cache = _load_cache()
    out = {}
    for name, entry in cache.get("benchmarks", {}).items():
        out[name] = {
            "count": len(entry.get("tickers", [])),
            "refreshed_at": entry.get("refreshed_at"),
        }
    return out
