"""Point-in-time quarterly fundamental store (roadmap item #9).

Hedge-fund orthodoxy (McLean & Pontiff 2016 "Does Academic Research Destroy
Stock Return Predictability?", JF): ranking today on fundamentals that were
actually reported *after* today is look-ahead bias.  The snapshot store
records each quarterly report with its `report_date` and, when the vendor
provides it, the filing/accepted date so that backtests and rolling IC studies
can use only data that was knowable at the ranking date.

Structure:

    feature_cache/pit_fundamentals.json
    {
      "version": 1,
      "tickers": {
         "AAPL": {
            "2024-09-28": {
                "_report_date": "2024-09-28",
                "_accepted_date": "2024-10-31",
                "net_income": 14736000000,
                "total_assets": 364840000000,
                ...
            },
            "2024-06-29": {...}
         },
         ...
      }
    }

Only additive writes — prior snapshots are never modified.

Callers that want a ranking-time view call `latest_as_of(ticker, date)`.
When `_accepted_date` exists, the snapshot becomes available on that date.
Older snapshots without accepted-date metadata use the conservative
`report_date + lag_days` fallback (default 45 days).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Mapping

from utils.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path("feature_cache") / "pit_fundamentals.json"
_SCHEMA_VERSION = 1
_DEFAULT_REPORT_LAG_DAYS = 45  # SEC 10-Q filing deadline for accelerated filers
_RICH_STATEMENT_SOURCES = {
    "fmp",
    "sec_edgar",
    "esef_ixbrl",
    "yahoo_timeseries",
    "yfinance_quarterly",
    "alpha_vantage_adr",
}
_LIVE_CAPTURE_SOURCES = {"yfinance_info"}
_RICH_SNAPSHOT_MAX_STALENESS_DAYS = 550


def _load_store(path: Path = _DEFAULT_PATH) -> dict:
    if not path.exists():
        return {"version": _SCHEMA_VERSION, "tickers": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("pit_store: corrupt store %s (%s); starting fresh", path, e)
        return {"version": _SCHEMA_VERSION, "tickers": {}}
    if not isinstance(data, dict) or "tickers" not in data:
        return {"version": _SCHEMA_VERSION, "tickers": {}}
    return data


def _save_store(data: dict, path: Path = _DEFAULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(str(path), data)


def record_snapshot(
    ticker: str,
    report_date: str | date | datetime,
    fundamentals: Mapping,
    *,
    accepted_date: str | date | datetime | None = None,
    filing_date: str | date | datetime | None = None,
    source: str = "fmp",
    path: Path = _DEFAULT_PATH,
) -> None:
    """Persist a quarterly snapshot for (ticker, report_date).

    `fundamentals` should be the dict of line items already extracted by
    `engine.enterprise_factors.extract_piotroski_period` (net_income,
    total_assets, etc.) — stored verbatim.  Re-recording the same
    (ticker, report_date) overwrites only that cell.
    """
    if not ticker or not fundamentals:
        return
    rd = _coerce_date(report_date)
    if rd is None:
        logger.warning("pit_store: invalid report_date %r for %s", report_date, ticker)
        return

    store = _load_store(path)
    ticker_key = str(ticker).upper()
    entries = store.setdefault("tickers", {}).setdefault(ticker_key, {})
    payload = {
        k: (float(v) if isinstance(v, (int, float)) and v == v else v)
        for k, v in fundamentals.items()
        if v is not None
    }
    payload["_report_date"] = rd.isoformat()
    accepted = _coerce_date(accepted_date) or _coerce_date(filing_date)
    if accepted is not None:
        payload["_accepted_date"] = accepted.isoformat()
    payload["_source"] = str(source or "unknown")
    entries[rd.isoformat()] = payload
    _save_store(store, path)


def latest_as_of(
    ticker: str,
    as_of: str | date | datetime | None = None,
    *,
    lag_days: int = _DEFAULT_REPORT_LAG_DAYS,
    path: Path = _DEFAULT_PATH,
) -> tuple[dict | None, str | None]:
    """Return (fundamentals, report_date_iso) knowable at `as_of`.

    Applies a reporting-lag buffer so the caller doesn't use filings before
    they would have been public (SEC 10-Q: 45 days).  Returns (None, None)
    when no snapshot predates the cutoff.
    """
    store = _load_store(path)
    t = store.get("tickers", {}).get(str(ticker).upper())
    if not t:
        return None, None

    cutoff = _coerce_date(as_of) or date.today()

    candidates: list[tuple[date, str, dict]] = []
    for rd_str, payload in t.items():
        try:
            rd = date.fromisoformat(rd_str)
        except ValueError:
            continue
        available_date = _available_date(payload, rd, lag_days)
        if available_date <= cutoff:
            candidates.append((rd, rd_str, payload))
    best = _select_best_available_snapshot(candidates)
    if best is None:
        return None, None
    return dict(t[best[1]]), best[1]


def _snapshot_evidence_score(payload: Mapping | None) -> int:
    if not isinstance(payload, Mapping):
        return 0
    source = str(payload.get("_source") or "").lower()
    score = 0
    if payload.get("total_assets") is not None:
        score += 1
    if payload.get("gross_profit") is not None and payload.get("total_assets") is not None:
        score += 4
    if payload.get("net_income") is not None or payload.get("operating_cashflow") is not None:
        score += 1
    if payload.get("revenue") is not None:
        score += 1
    if payload.get("current_assets") is not None or payload.get("current_liabilities") is not None:
        score += 1
    if source in _RICH_STATEMENT_SOURCES and score > 0:
        score += 1
    if source in _LIVE_CAPTURE_SOURCES and payload.get("gross_profit") is None:
        score -= 1
    return score


def _select_best_available_snapshot(
    candidates: list[tuple[date, str, Mapping]],
    *,
    max_staleness_days: int = _RICH_SNAPSHOT_MAX_STALENESS_DAYS,
) -> tuple[date, str] | None:
    """Select the PIT snapshot with enough evidence to be useful.

    Live yfinance captures are dated at the capture date, not the fiscal report
    period.  Once better statement-level sources are available, a balance-only
    live capture should not suppress a recent SEC/FMP/Yahoo statement snapshot.
    The staleness guard keeps very old rich statements from dominating forever.
    """
    if not candidates:
        return None
    latest = max(candidates, key=lambda item: item[0])
    latest_score = _snapshot_evidence_score(latest[2])
    best_rich = max(
        candidates,
        key=lambda item: (_snapshot_evidence_score(item[2]), item[0]),
    )
    best_score = _snapshot_evidence_score(best_rich[2])
    if best_score >= latest_score + 2 and (latest[0] - best_rich[0]).days <= max_staleness_days:
        return best_rich[0], best_rich[1]
    return latest[0], latest[1]


def prior_snapshot(
    ticker: str,
    before_report_date: str | date,
    *,
    path: Path = _DEFAULT_PATH,
) -> tuple[dict | None, str | None]:
    """Return the snapshot immediately preceding `before_report_date`.

    Used to supply `prior` to Piotroski F-score (requires YoY comparison).
    Prefers 4-quarter-prior for "same quarter last year" comparability; falls
    back to the closest prior snapshot if no exact match.
    """
    store = _load_store(path)
    t = store.get("tickers", {}).get(str(ticker).upper())
    if not t:
        return None, None

    target = _coerce_date(before_report_date)
    if target is None:
        return None, None

    dates: list[date] = []
    for rd_str in t.keys():
        try:
            dates.append(date.fromisoformat(rd_str))
        except ValueError:
            continue
    if not dates:
        return None, None
    prior = [d for d in dates if d < target]
    if not prior:
        return None, None
    # Prefer nearest to target-365d for YoY comparison
    target_prior = target - timedelta(days=365)
    prior.sort(key=lambda d: abs((d - target_prior).days))
    best = prior[0]
    return dict(t[best.isoformat()]), best.isoformat()


def all_tickers(*, path: Path = _DEFAULT_PATH) -> list[str]:
    store = _load_store(path)
    return sorted(store.get("tickers", {}).keys())


# ---------------------------------------------------------------------------
# Live-ingest helper — populate PIT history from yfinance `.info` on each run
# ---------------------------------------------------------------------------

_INFO_FIELDS_FOR_PIT = (
    # (pit_key, preferred yfinance .info keys in priority order)
    ("net_income", ("netIncomeToCommon", "netIncome")),
    ("operating_cashflow", ("operatingCashflow",)),
    ("total_assets", ("totalAssets",)),
    ("total_debt", ("totalDebt",)),
    ("current_assets", ("currentAssets", "totalCurrentAssets")),
    ("current_liabilities", ("currentLiabilities", "totalCurrentLiabilities")),
    ("shares_outstanding", ("sharesOutstanding",)),
    ("gross_profit", ("grossProfits",)),
    ("revenue", ("totalRevenue",)),
    ("eps_ttm", ("trailingEps",)),
    ("book_value", ("bookValue",)),
    ("market_cap", ("marketCap",)),
    ("trailing_pe", ("trailingPE",)),
    ("forward_pe", ("forwardPE",)),
    ("return_on_equity", ("returnOnEquity",)),
    ("revenue_growth", ("revenueGrowth",)),
    ("earnings_growth", ("earningsGrowth",)),
)


def record_live_snapshot(
    ticker: str,
    info: Mapping,
    *,
    report_date: str | date | datetime | None = None,
    path: Path = _DEFAULT_PATH,
) -> bool:
    """Persist a PIT snapshot from live ``yfinance .info`` at today's date.

    Builds forward-looking history for backtests: every discovery run captures
    what was knowable today, so a future evaluation over a given ``as_of``
    date can retrieve fundamentals genuinely available at that time.
    """
    if not ticker or not isinstance(info, Mapping):
        return False
    rd = _coerce_date(report_date) if report_date is not None else date.today()
    if rd is None:
        rd = date.today()

    payload = _live_snapshot_payload(info)
    if not payload:
        return False
    try:
        record_snapshot(ticker, rd, payload, source="yfinance_info", path=path)
        return True
    except Exception as e:
        logger.debug("pit_store live snapshot failed for %s: %s", ticker, e)
        return False


def _live_snapshot_payload(info: Mapping) -> dict:
    """Extract the PIT-safe numeric fields from a live info payload."""
    payload: dict = {}
    for pit_key, src_keys in _INFO_FIELDS_FOR_PIT:
        val = None
        for sk in src_keys:
            v = info.get(sk)
            try:
                fv = float(v) if v is not None else None
            except (TypeError, ValueError):
                fv = None
            if fv is not None and fv == fv:  # NaN guard
                val = fv
                break
        if val is not None:
            payload[pit_key] = val
    return payload


def record_live_snapshots(
    items: Mapping[str, Mapping] | Iterable[tuple[str, Mapping]],
    *,
    report_date: str | date | datetime | None = None,
    path: Path = _DEFAULT_PATH,
) -> int:
    """Persist many live PIT snapshots with one load/write cycle.

    ``record_live_snapshot`` is intentionally simple for isolated callers, but
    discovery Stage 5b may touch hundreds of candidates. Rewriting the whole
    PIT JSON for every ticker is extremely slow on synced drives, so batch
    ingestion keeps the ranking loop bounded.
    """
    if not items:
        return 0
    rd = _coerce_date(report_date) if report_date is not None else date.today()
    if rd is None:
        rd = date.today()
    rd_key = rd.isoformat()

    iterable = items.items() if isinstance(items, Mapping) else items
    store = _load_store(path)
    tickers = store.setdefault("tickers", {})
    updated = 0
    for ticker, info in iterable:
        if not ticker or not isinstance(info, Mapping):
            continue
        payload = _live_snapshot_payload(info)
        if not payload:
            continue
        payload["_report_date"] = rd_key
        payload["_source"] = "yfinance_info"
        ticker_key = str(ticker).upper()
        tickers.setdefault(ticker_key, {})[rd_key] = payload
        updated += 1

    if updated:
        try:
            _save_store(store, path)
        except Exception as e:
            logger.debug("pit_store batch live snapshot failed: %s", e)
            return 0
    return updated


def as_of_info(
    ticker: str,
    as_of: str | date | datetime | None = None,
    *,
    lag_days: int = _DEFAULT_REPORT_LAG_DAYS,
    path: Path = _DEFAULT_PATH,
) -> dict | None:
    """Return a dict of fundamentals knowable at ``as_of`` (None=today).

    Thin wrapper around :func:`latest_as_of` for callers that only want the
    fields payload.  Returns None when no eligible snapshot exists.
    """
    fundamentals, _rd = latest_as_of(ticker, as_of, lag_days=lag_days, path=path)
    return fundamentals


def _coerce_date(v) -> date | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return date.fromisoformat(str(v)[:10])
    except ValueError:
        return None


def _available_date(payload: Mapping | None, report_date: date, lag_days: int) -> date:
    """Return the first date a snapshot should be visible to a backtest."""
    payload = payload or {}
    for key in ("_accepted_date", "accepted_date", "acceptedDate", "filing_date", "fillingDate", "filingDate"):
        parsed = _coerce_date(payload.get(key))
        if parsed is not None:
            return parsed
    return report_date + timedelta(days=int(lag_days))
