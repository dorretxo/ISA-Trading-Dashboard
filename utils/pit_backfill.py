"""Point-in-time fundamentals backfill helpers.

This module populates ``feature_cache/pit_fundamentals.json`` with quarterly
statement snapshots.  The snapshots are keyed by fiscal period end date and
consumed through ``pit_store.latest_as_of(..., lag_days=45)`` so replay code
cannot see a filing before it would normally be public.

It intentionally does not run a synthetic discovery replay.  That should only
happen after this store has broad coverage.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import yfinance as yf

import config
from engine.canonical_scores import compute_ev_ebit_score, compute_f_score_score, compute_gpa_score
from engine.enterprise_factors import compute_piotroski_f_score
from engine.factors import compute_factor_scores_from_result
from engine.paper_trading import _connect
from utils.atomic_io import atomic_write_json
from utils import fmp_client
from utils.pit_store import _available_date, _coerce_date, _load_store, record_snapshot
from utils.price_store import download_price_history, get_price_history

logger = logging.getLogger(__name__)
_PIT_STORE_CACHE: dict | None = None


def _float(value):
    try:
        if value is None:
            return None
        result = float(value)
        return result if result == result else None
    except (TypeError, ValueError):
        return None


def _snapshot_from_rows(
    income: Mapping | None,
    balance: Mapping | None,
    cashflow: Mapping | None = None,
) -> dict:
    income = income or {}
    balance = balance or {}
    cashflow = cashflow or {}
    short_debt = _float(balance.get("shortTermDebt"))
    long_debt = _float(balance.get("longTermDebt"))
    cash = _float(balance.get("cashAndCashEquivalents") or balance.get("cashAndShortTermInvestments"))
    fields = {
        "net_income": _float(income.get("netIncome") or income.get("bottomLineNetIncome")),
        "gross_profit": _float(income.get("grossProfit")),
        "revenue": _float(income.get("revenue")),
        "operating_cashflow": _float(
            cashflow.get("netCashProvidedByOperatingActivities")
            or cashflow.get("operatingCashFlow")
        ),
        "capital_expenditure": _float(cashflow.get("capitalExpenditure")),
        "ebit": _float(income.get("operatingIncome") or income.get("ebit")),
        "ebitda": _float(income.get("ebitda")),
        "eps": _float(income.get("eps") or income.get("epsdiluted")),
        "shares_outstanding": _float(
            income.get("weightedAverageShsOutDil")
            or income.get("weightedAverageShsOut")
        ),
        "total_assets": _float(balance.get("totalAssets")),
        "cash": cash,
        "long_term_debt": long_debt,
        "short_term_debt": short_debt,
        "total_debt": (short_debt or 0.0) + (long_debt or 0.0)
        if (short_debt is not None or long_debt is not None)
        else None,
        "current_assets": _float(balance.get("totalCurrentAssets")),
        "current_liabilities": _float(balance.get("totalCurrentLiabilities")),
    }
    return {k: v for k, v in fields.items() if v is not None}


def _row_available_date(*rows: Mapping | None) -> str | None:
    for row in rows:
        if not row:
            continue
        for key in ("acceptedDate", "accepted_date", "fillingDate", "filingDate"):
            value = row.get(key)
            if value:
                return str(value)[:10]
    return None


def backfill_ticker(ticker: str, *, limit: int | None = None) -> int:
    """Fetch quarterly FMP statements and record PIT snapshots for one ticker."""
    ticker = str(ticker or "").upper().strip()
    if not ticker:
        return 0
    limit = int(limit or getattr(config, "PIT_BACKFILL_DEFAULT_QUARTERS", 40))

    income_rows = fmp_client.get_income_statement(ticker, period="quarter", limit=limit) or []
    balance_rows = fmp_client.get_balance_sheet_statement(ticker, period="quarter", limit=limit) or []
    cashflow_rows = fmp_client.get_cash_flow_statement(ticker, period="quarter", limit=limit) or []
    if not income_rows and not balance_rows and not cashflow_rows:
        return 0

    income_by_date = {str(row.get("date", ""))[:10]: row for row in income_rows if row.get("date")}
    balance_by_date = {str(row.get("date", ""))[:10]: row for row in balance_rows if row.get("date")}
    cashflow_by_date = {str(row.get("date", ""))[:10]: row for row in cashflow_rows if row.get("date")}
    dates = sorted(set(income_by_date) | set(balance_by_date) | set(cashflow_by_date))

    written = 0
    for period_date in dates:
        income = income_by_date.get(period_date)
        balance = balance_by_date.get(period_date)
        cashflow = cashflow_by_date.get(period_date)
        snapshot = _snapshot_from_rows(income, balance, cashflow)
        if not snapshot:
            continue
        accepted_date = _row_available_date(income, balance, cashflow)
        record_snapshot(ticker, period_date, snapshot, accepted_date=accepted_date, source="fmp")
        written += 1
    logger.info("PIT backfill: %s wrote %d quarterly snapshots", ticker, written)
    return written


def _yf_statement_value(df: pd.DataFrame | None, period, names: tuple[str, ...]) -> float | None:
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index and period in df.columns:
            val = _float(df.loc[name, period])
            if val is not None:
                return val
    return None


def _snapshot_from_yfinance(
    income: pd.DataFrame | None,
    balance: pd.DataFrame | None,
    cashflow: pd.DataFrame | None,
    period,
) -> dict:
    short_debt = _yf_statement_value(balance, period, ("Current Debt", "Short Long Term Debt", "Short Term Debt"))
    long_debt = _yf_statement_value(balance, period, ("Long Term Debt", "Long Term Debt And Capital Lease Obligation"))
    fields = {
        "net_income": _yf_statement_value(income, period, ("Net Income", "Net Income Common Stockholders")),
        "gross_profit": _yf_statement_value(income, period, ("Gross Profit",)),
        "revenue": _yf_statement_value(income, period, ("Total Revenue", "Operating Revenue")),
        "operating_cashflow": _yf_statement_value(cashflow, period, ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities")),
        "capital_expenditure": _yf_statement_value(cashflow, period, ("Capital Expenditure", "Capital Expenditures")),
        "ebit": _yf_statement_value(income, period, ("EBIT", "Operating Income")),
        "ebitda": _yf_statement_value(income, period, ("EBITDA",)),
        "total_assets": _yf_statement_value(balance, period, ("Total Assets",)),
        "cash": _yf_statement_value(balance, period, ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments")),
        "long_term_debt": long_debt,
        "short_term_debt": short_debt,
        "total_debt": (short_debt or 0.0) + (long_debt or 0.0)
        if (short_debt is not None or long_debt is not None)
        else _yf_statement_value(balance, period, ("Total Debt",)),
        "current_assets": _yf_statement_value(balance, period, ("Current Assets", "Total Current Assets")),
        "current_liabilities": _yf_statement_value(balance, period, ("Current Liabilities", "Total Current Liabilities")),
        "shares_outstanding": _yf_statement_value(income, period, ("Diluted Average Shares", "Basic Average Shares")),
    }
    return {k: v for k, v in fields.items() if v is not None}


def backfill_via_yfinance_quarterly(
    tickers: Iterable[str],
    *,
    limit: int | None = None,
    sleep_seconds: float = 0.0,
) -> dict[str, int]:
    """Best-effort non-US PIT widening from yfinance quarterly statements.

    Snapshots are tagged ``_source='yfinance_quarterly'`` and have no accepted
    filing date, so normal ``report_date + 45d`` lagging still applies.
    """
    limit = int(limit or 16)
    results: dict[str, int] = {}
    for ticker in tickers:
        symbol = str(ticker or "").upper().strip()
        if not symbol:
            continue
        written = 0
        try:
            obj = yf.Ticker(symbol)
            income = getattr(obj, "quarterly_financials", None)
            balance = getattr(obj, "quarterly_balance_sheet", None)
            cashflow = getattr(obj, "quarterly_cashflow", None)
            periods = []
            for df in (income, balance, cashflow):
                if isinstance(df, pd.DataFrame) and not df.empty:
                    periods.extend(list(df.columns))
            unique_periods = sorted(set(periods), reverse=True)[:limit]
            for period in unique_periods:
                try:
                    period_date = pd.to_datetime(period).date().isoformat()
                except Exception:
                    continue
                snapshot = _snapshot_from_yfinance(income, balance, cashflow, period)
                if not snapshot:
                    continue
                record_snapshot(symbol, period_date, snapshot, source="yfinance_quarterly")
                written += 1
        except Exception as exc:
            logger.warning("yfinance quarterly PIT backfill failed for %s: %s", symbol, exc)
        results[symbol] = written
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return results


def backfill_tickers(tickers: Iterable[str], *, limit: int | None = None, sleep_seconds: float = 0.0) -> dict[str, int]:
    results: dict[str, int] = {}
    for ticker in tickers:
        try:
            results[str(ticker).upper()] = backfill_ticker(str(ticker), limit=limit)
        except Exception as exc:
            logger.warning("PIT backfill failed for %s: %s", ticker, exc)
            results[str(ticker).upper()] = 0
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return results


def _is_fmp_statement_candidate(ticker: str) -> bool:
    symbol = str(ticker or "").upper().strip()
    return bool(symbol) and "." not in symbol


def refresh_queue_tickers(
    *,
    queue_path: str | Path | None = None,
    max_tickers: int | None = None,
    limit: int | None = None,
    sleep_seconds: float = 0.0,
    yfinance_fallback: bool = True,
    fmp_only: bool = False,
    write_results: bool = True,
) -> dict:
    """Backfill PIT fundamentals for tickers in the finalist refresh queue.

    FMP is tried for plain US-style tickers. yfinance quarterly statements are
    used for non-US tickers and as a fallback when FMP writes no snapshots.
    """
    path = Path(queue_path or getattr(config, "DISCOVERY_FUNDAMENTAL_REFRESH_QUEUE_PATH", "feature_cache/fundamental_refresh_queue.json"))
    if not path.exists():
        return {"selected": 0, "refreshed": 0, "queue_path": str(path), "error": "queue_missing"}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"selected": 0, "refreshed": 0, "queue_path": str(path), "error": f"queue_read_failed: {exc}"}

    rows = raw.get("items", raw if isinstance(raw, list) else [])
    if not isinstance(rows, list):
        rows = []
    rows = sorted(
        [row for row in rows if isinstance(row, dict) and row.get("ticker")],
        key=lambda row: float(row.get("priority", 0) or 0),
        reverse=True,
    )
    if fmp_only:
        rows = [
            row for row in rows
            if _is_fmp_statement_candidate(str(row.get("ticker") or ""))
        ]
    max_tickers = int(max_tickers or getattr(config, "DISCOVERY_FUNDAMENTAL_REFRESH_MAX_TICKERS", 40))
    selected_rows = rows[:max(0, max_tickers)]
    tickers = [str(row.get("ticker")).upper().strip() for row in selected_rows if row.get("ticker")]

    fmp_results: dict[str, int] = {}
    yf_results: dict[str, int] = {}
    fallback_tickers: list[str] = []
    for ticker in tickers:
        written = 0
        if _is_fmp_statement_candidate(ticker):
            try:
                written = backfill_ticker(ticker, limit=limit)
            except Exception as exc:
                logger.warning("Refresh queue FMP backfill failed for %s: %s", ticker, exc)
                written = 0
            fmp_results[ticker] = written
        if yfinance_fallback and written <= 0:
            fallback_tickers.append(ticker)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if yfinance_fallback and fallback_tickers:
        yf_results = backfill_via_yfinance_quarterly(
            fallback_tickers,
            limit=limit or 16,
            sleep_seconds=sleep_seconds,
        )

    refreshed = {
        ticker: (fmp_results.get(ticker, 0) or 0) + (yf_results.get(ticker, 0) or 0)
        for ticker in tickers
    }
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "queue_path": str(path),
        "fmp_only": bool(fmp_only),
        "yfinance_fallback": bool(yfinance_fallback),
        "selected": len(tickers),
        "refreshed": sum(1 for value in refreshed.values() if value > 0),
        "snapshots_written": sum(refreshed.values()),
        "fmp_results": fmp_results,
        "yfinance_results": yf_results,
        "refreshed_snapshots": refreshed,
        "unrefreshed": [ticker for ticker, value in refreshed.items() if value <= 0],
    }
    if write_results:
        out_path = Path(getattr(config, "DISCOVERY_FUNDAMENTAL_REFRESH_RESULTS_PATH", "feature_cache/fundamental_refresh_results.json"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out_path, payload, indent=2)
    return payload


def signal_backtest_tickers(*, max_tickers: int | None = None) -> list[str]:
    with _connect() as conn:
        sql = "SELECT DISTINCT ticker FROM signal_backtest ORDER BY ticker"
        if max_tickers:
            sql += f" LIMIT {int(max_tickers)}"
        rows = conn.execute(sql).fetchall()
    return [str(row[0]).upper() for row in rows if row[0]]


def _ensure_factor_backfill_columns() -> None:
    columns = {
        "factors_backfilled": "INTEGER",
        "factors_source": "TEXT",
        "factors_backfilled_at": "TEXT",
        "factors_unavailable": "INTEGER",
        "factors_unavailable_reason": "TEXT",
    }
    with _connect() as conn:
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(signal_backtest)").fetchall()}
        for name, col_type in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE signal_backtest ADD COLUMN {name} {col_type}")


def _finite(value) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _pit_store_cached() -> dict:
    global _PIT_STORE_CACHE
    if _PIT_STORE_CACHE is None:
        _PIT_STORE_CACHE = _load_store()
    return _PIT_STORE_CACHE


def _latest_as_of_cached(ticker: str, as_of: str, *, lag_days: int) -> tuple[dict | None, str | None]:
    store = _pit_store_cached()
    entries = store.get("tickers", {}).get(str(ticker).upper())
    if not entries:
        return None, None
    cutoff = _coerce_date(as_of) or date.today()
    best: tuple[date, str] | None = None
    for rd_str, payload in entries.items():
        try:
            rd = date.fromisoformat(str(rd_str)[:10])
        except ValueError:
            continue
        if _available_date(payload, rd, lag_days) <= cutoff and (best is None or rd > best[0]):
            best = (rd, rd_str)
    if best is None:
        return None, None
    return dict(entries[best[1]]), best[1]


def _prior_snapshot_cached(ticker: str, before_report_date: str) -> tuple[dict | None, str | None]:
    store = _pit_store_cached()
    entries = store.get("tickers", {}).get(str(ticker).upper())
    if not entries:
        return None, None
    target = _coerce_date(before_report_date)
    if target is None:
        return None, None
    dates: list[date] = []
    for rd_str in entries.keys():
        try:
            dates.append(date.fromisoformat(str(rd_str)[:10]))
        except ValueError:
            continue
    prior = [d for d in dates if d < target]
    if not prior:
        return None, None
    target_prior = target - timedelta(days=365)
    prior.sort(key=lambda d: abs((d - target_prior).days))
    best = prior[0].isoformat()
    return dict(entries[best]), best


def _pit_factor_snapshot(ticker: str, run_date: str, signal_price: float | None) -> tuple[dict, str | None]:
    latest, report_date = _latest_as_of_cached(
        ticker,
        run_date,
        lag_days=int(getattr(config, "PIT_FUNDAMENTAL_LAG_DAYS", 45)),
    )
    if not latest or not report_date:
        return {}, None
    prior, _prior_date = _prior_snapshot_cached(ticker, report_date)
    f_result = compute_piotroski_f_score(latest, prior)
    fields: dict[str, float | int | None] = {
        "f_score": f_result.get("f_score"),
        "f_score_coverage": f_result.get("f_score_coverage"),
    }
    f_score_score = f_result.get("f_score_score")
    if f_score_score is not None:
        fields["f_score_score"] = f_score_score

    total_assets = _finite(latest.get("total_assets"))
    gross_profit = _finite(latest.get("gross_profit"))
    if total_assets and total_assets > 0 and gross_profit is not None:
        gpa = gross_profit / total_assets
        gpa_score = compute_gpa_score(gpa)
        fields.update({
            "gpa": gpa,
            "gpa_score": gpa_score,
            "gross_profitability": gpa,
        })

    price = _finite(signal_price)
    shares = _finite(latest.get("shares_outstanding"))
    ebit = _finite(latest.get("ebit"))
    operating_cashflow = _finite(latest.get("operating_cashflow"))
    capex = _finite(latest.get("capital_expenditure"))
    total_debt = _finite(latest.get("total_debt")) or 0.0
    cash = _finite(latest.get("cash")) or 0.0
    market_cap = price * shares if price and price > 0 and shares and shares > 0 else None
    # NOTE: do NOT derive pe_ratio = market_cap / net_income here.  The PIT
    # snapshot's net_income is a single-quarter value while live's pe_ratio
    # (yfinance trailing P/E) is TTM-based.  Naive division creates ~4x basis
    # drift in the parity comparator (see PARR/BVS/REPX drilldown).  Correct
    # PIT pe_ratio needs TTM aggregation across 4 quarters — separate change.
    # Until then we leave replay's pe_ratio NULL so the comparator records
    # 'missing' (not 'drift') for value_factor_score recompute.
    if operating_cashflow is not None:
        if capex is None:
            fcf = operating_cashflow
        else:
            fcf = operating_cashflow + capex if capex < 0 else operating_cashflow - capex
        if market_cap and market_cap > 0:
            fields["fcf_yield"] = fcf / market_cap
        if total_assets and total_assets > 0:
            fields["fcf_to_assets"] = fcf / total_assets
    if price and price > 0 and shares and shares > 0 and ebit and ebit > 0:
        ev = price * shares + total_debt - cash
        if ev > 0:
            ev_ebit = ev / ebit
            ebit_yield = ebit / ev
            fields["ev_ebit"] = ev_ebit
            fields["ebit_yield"] = ebit_yield
            fields["ev_ebit_score"] = compute_ev_ebit_score(ebit_yield)

    try:
        factor_scores = compute_factor_scores_from_result(fields)
        if any(fields.get(k) is not None for k in ("quality_score_fundamental", "gpa_score", "f_score_score", "gpa", "f_score")):
            for key in ("quality_factor_score", "gpa_factor_score", "f_score_factor_score"):
                if factor_scores.get(key) is not None:
                    fields[key] = factor_scores[key]
        if any(fields.get(k) is not None for k in ("gpa_score", "gpa", "gross_profitability", "f_score_score", "earnings_stability")):
            for key in ("qmj_factor_score", "qmj_component_count"):
                if factor_scores.get(key) is not None:
                    fields[key] = factor_scores[key]
        if any(fields.get(k) is not None for k in ("pe_ratio", "peg_ratio", "fcf_yield", "ev_ebit_score", "pb_score", "ps_score")):
            fields["value_factor_score"] = factor_scores.get("value_factor_score")
            if factor_scores.get("ev_ebit_factor_score") is not None:
                fields["ev_ebit_factor_score"] = factor_scores["ev_ebit_factor_score"]
    except Exception:
        logger.debug("PIT factor derivation failed for %s %s", ticker, run_date)

    return {k: v for k, v in fields.items() if v is not None}, report_date


def _price_factor_snapshot(
    ticker: str,
    run_date: str,
    signal_price: float | None,
    *,
    frame: pd.DataFrame | None = None,
) -> tuple[dict, str | None]:
    as_of = pd.Timestamp(run_date).tz_localize(None).normalize()
    if frame is None:
        start = (as_of - pd.Timedelta(days=430)).date().isoformat()
        frame = get_price_history(ticker, start=start, end=as_of.date().isoformat())
    if frame is None or frame.empty or "Close" not in frame.columns:
        return {}, None
    hist = frame.loc[frame.index <= as_of].copy()
    close = hist["Close"].astype(float).dropna()
    if len(close) < 200:
        return {}, None
    price = _finite(signal_price) or _finite(close.iloc[-1])
    if not price or price <= 0:
        return {}, None
    sma200 = float(close.tail(200).mean())
    if not math.isfinite(sma200) or sma200 <= 0:
        return {}, None
    out = {
        "sma_200": sma200,
        "price_vs_sma200_stretch": price / sma200 - 1.0,
    }
    if len(close) >= 91:
        ret_90 = price / float(close.iloc[-91]) - 1.0
        out["momentum_factor_score"] = float(np.clip(ret_90 / 0.35, -1.0, 1.0))
    if len(close) >= 21:
        daily = close.pct_change(fill_method=None).dropna()
        if len(daily) >= 20:
            vol = float(daily.tail(20).std() * np.sqrt(252))
            if math.isfinite(vol):
                out["volatility_factor_score"] = float(np.clip((0.30 - vol) / 0.20, -1.0, 1.0))
    return out, "price_history"


def _select_factor_backfill_rows(
    *,
    min_date: str | None,
    max_date: str | None,
    limit: int | None,
) -> list:
    where = [
        "source NOT LIKE 'replay%'",
        "(evaluated_5d = 1 OR evaluated_10d = 1 OR evaluated_30d = 1)",
        "(COALESCE(factors_backfilled, 0) = 0)",
        "("
        "quality_factor_score IS NULL OR f_score IS NULL OR gpa IS NULL "
        "OR ev_ebit IS NULL OR price_vs_sma200_stretch IS NULL"
        ")",
    ]
    params: list = []
    if min_date:
        where.append("date(run_date) >= date(?)")
        params.append(min_date)
    if max_date:
        where.append("date(run_date) <= date(?)")
        params.append(max_date)
    sql = (
        "SELECT * FROM signal_backtest WHERE "
        + " AND ".join(where)
        + " ORDER BY run_date, ticker, id"
    )
    if limit:
        sql += f" LIMIT {int(limit)}"
    with _connect() as conn:
        return conn.execute(sql, params).fetchall()


def _apply_factor_updates(row_id: int, updates: dict, *, source: str, unavailable_reason: str | None = None) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    payload = dict(updates)
    if payload:
        payload["factors_backfilled"] = 1
        payload["factors_source"] = source
        payload["factors_backfilled_at"] = now
        payload["factors_unavailable"] = 0
        payload["factors_unavailable_reason"] = None
    else:
        payload["factors_backfilled"] = 0
        payload["factors_source"] = source
        payload["factors_backfilled_at"] = now
        payload["factors_unavailable"] = 1
        payload["factors_unavailable_reason"] = unavailable_reason or "no_pit_or_price_snapshot"
    assignments = ", ".join(f"{col}=?" for col in payload)
    values = list(payload.values()) + [row_id]
    with _connect() as conn:
        conn.execute(f"UPDATE signal_backtest SET {assignments} WHERE id=?", values)


def backfill_factor_snapshot_pit(
    min_date: str | None = None,
    max_date: str | None = None,
    *,
    dry_run: bool = True,
    limit: int | None = None,
    download_prices: bool = True,
) -> dict:
    """Backfill signal-time factors from PIT fundamentals and historical prices.

    This never recomputes from today's yfinance ``info`` or current feature
    cache. Only already-recorded rows, PIT snapshots as-of ``run_date``, and
    OHLCV sliced to ``run_date`` are used.
    """
    _ensure_factor_backfill_columns()
    rows = _select_factor_backfill_rows(min_date=min_date, max_date=max_date, limit=limit)
    if not rows:
        return {"selected": 0, "updated": 0, "unavailable": 0, "dry_run": dry_run}

    if download_prices:
        dates = [pd.Timestamp(row["run_date"]).normalize() for row in rows if row["run_date"]]
        tickers = sorted({str(row["ticker"]).upper() for row in rows if row["ticker"]})
        if dates and tickers:
            start = (min(dates) - pd.Timedelta(days=430)).date().isoformat()
            end = max(dates).date().isoformat()
            download_price_history(
                tickers,
                start=start,
                end=end,
                batch_size=getattr(config, "PRICE_CACHE_BATCH_SIZE", 75),
            )

    updated = 0
    unavailable = 0
    pit_hits = 0
    price_hits = 0
    price_frame_cache: dict[str, pd.DataFrame] = {}
    for row in rows:
        row_id = int(row["id"])
        ticker = str(row["ticker"]).upper()
        run_date = str(row["run_date"])[:10]
        signal_price = _finite(row["signal_price"])
        pit_fields, report_date = _pit_factor_snapshot(ticker, run_date, signal_price)
        if ticker not in price_frame_cache:
            price_frame_cache[ticker] = get_price_history(ticker)
        price_fields, _price_source = _price_factor_snapshot(
            ticker,
            run_date,
            signal_price,
            frame=price_frame_cache.get(ticker),
        )

        updates: dict = {}
        for col, value in {**pit_fields, **price_fields}.items():
            if col not in row.keys():
                continue
            if row[col] is None and value is not None:
                updates[col] = value
        sources = []
        if pit_fields:
            sources.append(f"pit:{report_date or 'unknown'}")
            pit_hits += 1
        if price_fields:
            sources.append("price_history")
            price_hits += 1
        source = "+".join(sources) if sources else "unavailable"
        if updates:
            updated += 1
            if not dry_run:
                _apply_factor_updates(row_id, updates, source=source)
        else:
            unavailable += 1
            if not dry_run:
                _apply_factor_updates(row_id, {}, source=source)

    return {
        "selected": len(rows),
        "updated": updated,
        "unavailable": unavailable,
        "pit_hits": pit_hits,
        "price_hits": price_hits,
        "dry_run": dry_run,
    }


def _mean_finite(values: Iterable[float | None]) -> float | None:
    vals = [_finite(v) for v in values]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.clip(float(np.mean(vals)), -1.0, 1.0))


def _score_from_f_score(value) -> float | None:
    return compute_f_score_score(value)


def _score_from_pe(value) -> float | None:
    pe = _finite(value)
    if pe is None or pe <= 0:
        return None
    return float(np.clip((20.0 - pe) / 20.0, -1.0, 1.0))


def _score_from_vol(value) -> float | None:
    vol = _finite(value)
    if vol is None or vol <= 0:
        return None
    return float(np.clip((0.35 - vol) / 0.25, -1.0, 1.0))


def _score_from_probability(value) -> float | None:
    prob = _finite(value)
    if prob is None:
        return None
    return float(np.clip(2.0 * prob - 1.0, -1.0, 1.0))


def _score_from_rr(value) -> float | None:
    rr = _finite(value)
    if rr is None:
        return None
    return float(np.clip((rr - 1.5) / 1.5, -1.0, 1.0))


def _score_from_stretch(value) -> float | None:
    stretch = _finite(value)
    if stretch is None:
        return None
    # Reward names close to trend support; penalise extreme upside extension.
    return float(np.clip((0.50 - max(stretch, 0.0)) / 0.50, -1.0, 1.0))


def _row_sleeve_snapshot(row) -> dict:
    """Assemble sleeve scalars from signal-time columns already in the DB.

    This is intentionally row-local. It does not call yfinance ``info`` or read
    current feature caches, so replay rows keep their point-in-time integrity.
    """
    quality = _mean_finite([
        row["quality_factor_score"] if "quality_factor_score" in row.keys() else None,
        row["qmj_factor_score"] if "qmj_factor_score" in row.keys() else None,
        row["gpa_score"] if "gpa_score" in row.keys() else None,
        _score_from_f_score(row["f_score"] if "f_score" in row.keys() else None),
    ])
    value = _mean_finite([
        row["value_factor_score"] if "value_factor_score" in row.keys() else None,
        row["ev_ebit_score"] if "ev_ebit_score" in row.keys() else None,
        _score_from_pe(row["pe_ratio"] if "pe_ratio" in row.keys() else None),
    ])
    momentum = _mean_finite([
        row["momentum_factor_score"] if "momentum_factor_score" in row.keys() else None,
        row["momentum_score"] if "momentum_score" in row.keys() else None,
    ])
    low_risk = _mean_finite([
        row["bab_factor_score"] if "bab_factor_score" in row.keys() else None,
        row["volatility_factor_score"] if "volatility_factor_score" in row.keys() else None,
        _score_from_vol(row["vol_20d"] if "vol_20d" in row.keys() else None),
    ])
    pead = _mean_finite([
        row["pead_factor_score"] if "pead_factor_score" in row.keys() else None,
        row["sue_score"] if "sue_score" in row.keys() else None,
        row["revision_momentum_3m"] if "revision_momentum_3m" in row.keys() else None,
    ])
    ready = _mean_finite([
        _score_from_probability(row["fill_probability"] if "fill_probability" in row.keys() else None),
        _score_from_rr(row["r_r_ratio"] if "r_r_ratio" in row.keys() else None),
        _score_from_stretch(row["price_vs_sma200_stretch"] if "price_vs_sma200_stretch" in row.keys() else None),
    ])

    updates = {
        "sleeve_momentum": momentum,
        "sleeve_quality": quality,
        "sleeve_value": value if value is not None else 0.0,
        "sleeve_low_risk": low_risk,
        "sleeve_pead": pead if pead is not None else 0.0,
        "sleeve_ready": ready if ready is not None else 0.0,
    }
    return {k: v for k, v in updates.items() if v is not None}


def backfill_replay_sleeves(
    *,
    source_like: str = "replay%",
    dry_run: bool = True,
    limit: int | None = None,
    recompute_stats: bool = True,
) -> dict:
    """Backfill sleeve scalar columns for replay/live research rows.

    The source rows already contain signal-time factor columns. This function
    only maps those columns into the newer sleeve schema so the ML ranker and
    pillar-effectiveness tables stop seeing all-NULL sleeve features.
    """
    _ensure_factor_backfill_columns()
    where = [
        "source LIKE ?",
        "("
        "sleeve_momentum IS NULL OR sleeve_quality IS NULL OR sleeve_value IS NULL "
        "OR sleeve_low_risk IS NULL OR sleeve_pead IS NULL OR sleeve_ready IS NULL"
        ")",
    ]
    params: list = [source_like]
    sql = (
        "SELECT * FROM signal_backtest WHERE "
        + " AND ".join(where)
        + " ORDER BY run_date, ticker, id"
    )
    if limit:
        sql += f" LIMIT {int(limit)}"
    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    selected = len(rows)
    updated = 0
    skipped = 0
    now = datetime.now().isoformat(timespec="seconds")
    pending_updates: list[tuple[dict, int]] = []
    for row in rows:
        updates = _row_sleeve_snapshot(row)
        missing_only = {
            k: v for k, v in updates.items()
            if k in row.keys() and row[k] is None and v is not None
        }
        if not missing_only:
            skipped += 1
            continue
        updated += 1
        if dry_run:
            continue
        payload = dict(missing_only)
        payload.update({
            "factors_backfilled": 1,
            "factors_source": "row_factor_sleeves",
            "factors_backfilled_at": now,
            "factors_unavailable": 0,
            "factors_unavailable_reason": None,
        })
        pending_updates.append((payload, int(row["id"])))

    if pending_updates and not dry_run:
        with _connect() as conn:
            for payload, row_id in pending_updates:
                assignments = ", ".join(f"{col}=?" for col in payload)
                values = list(payload.values()) + [row_id]
                conn.execute(f"UPDATE signal_backtest SET {assignments} WHERE id=?", values)

    if updated and not dry_run and recompute_stats:
        try:
            from engine.discovery_backtest import _recompute_all_stats
            _recompute_all_stats()
        except Exception as exc:
            logger.warning("Replay sleeve backfill updated rows but stats recompute failed: %s", exc)

    return {
        "selected": selected,
        "updated": updated,
        "skipped": skipped,
        "source_like": source_like,
        "dry_run": dry_run,
    }


def _parse_tickers(value: str) -> list[str]:
    return [part.strip().upper() for part in value.split(",") if part.strip()]


def _main() -> None:
    parser = argparse.ArgumentParser(description="Backfill PIT quarterly fundamentals from FMP")
    parser.add_argument("--tickers", default="", help="Comma-separated ticker list")
    parser.add_argument("--from-signal-db", action="store_true", help="Use distinct tickers in signal_backtest")
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--yfinance-quarterly", action="store_true", help="Use yfinance quarterly statements and tag snapshots as yfinance_quarterly")
    parser.add_argument("--backfill-factor-snapshot", action="store_true", help="Backfill PIT-safe factor snapshots into signal_backtest")
    parser.add_argument("--backfill-replay-sleeves", action="store_true", help="Backfill sleeve scalar columns from signal-time factor rows")
    parser.add_argument("--refresh-queue", action="store_true", help="Backfill PIT fundamentals for feature_cache/fundamental_refresh_queue.json")
    parser.add_argument("--queue-path", default=None, help="Override fundamental refresh queue path")
    parser.add_argument("--no-yfinance-fallback", action="store_true", help="Disable yfinance quarterly fallback for refresh queue")
    parser.add_argument("--source-like", default="replay%", help="SQL LIKE pattern for sleeve backfill source rows")
    parser.add_argument("--min-date", default=None)
    parser.add_argument("--max-date", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--apply", action="store_true", help="Apply factor backfill updates. Default is dry-run.")
    parser.add_argument("--no-price-download", action="store_true", help="Use only already-cached OHLCV for factor backfill")
    parser.add_argument(
        "--limit",
        type=int,
        default=getattr(config, "PIT_BACKFILL_DEFAULT_QUARTERS", 40),
        help="Quarterly statements per ticker",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional pause between tickers")
    args = parser.parse_args()

    if args.backfill_factor_snapshot:
        stats = backfill_factor_snapshot_pit(
            min_date=args.min_date,
            max_date=args.max_date,
            dry_run=not args.apply,
            limit=args.max_rows,
            download_prices=not args.no_price_download,
        )
        print(json.dumps(stats, indent=2, sort_keys=True))
        return

    if args.backfill_replay_sleeves:
        stats = backfill_replay_sleeves(
            source_like=args.source_like,
            dry_run=not args.apply,
            limit=args.max_rows,
            recompute_stats=True,
        )
        print(json.dumps(stats, indent=2, sort_keys=True))
        return

    if args.refresh_queue:
        stats = refresh_queue_tickers(
            queue_path=args.queue_path,
            max_tickers=args.max_tickers,
            limit=args.limit,
            sleep_seconds=args.sleep,
            yfinance_fallback=not args.no_yfinance_fallback,
            fmp_only=args.no_yfinance_fallback,
            write_results=True,
        )
        print(json.dumps(stats, indent=2, sort_keys=True))
        return

    tickers = _parse_tickers(args.tickers)
    if args.from_signal_db:
        tickers.extend(signal_backtest_tickers(max_tickers=args.max_tickers))
    tickers = sorted(set(tickers))
    if not tickers:
        raise SystemExit("No tickers supplied")

    if args.yfinance_quarterly:
        results = backfill_via_yfinance_quarterly(tickers, limit=args.limit, sleep_seconds=args.sleep)
    else:
        results = backfill_tickers(tickers, limit=args.limit, sleep_seconds=args.sleep)
    total = sum(results.values())
    print(f"pit_snapshots_written={total} tickers={len(results)}")


if __name__ == "__main__":
    _main()
