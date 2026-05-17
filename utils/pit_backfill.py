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
import csv
import json
import logging
import math
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import config
from engine.canonical_scores import compute_ev_ebit_score, compute_f_score_score, compute_gpa_score
from engine.enterprise_factors import compute_piotroski_f_score
from engine.factors import compute_factor_scores_from_result
from engine.paper_trading import _connect
from utils.atomic_io import atomic_write_json
from utils import fmp_client
from utils.pit_store import (
    _available_date,
    _coerce_date,
    _load_store,
    _select_best_available_snapshot,
    record_snapshot,
)
from utils.price_store import download_price_history, get_price_history

logger = logging.getLogger(__name__)
_PIT_STORE_CACHE: dict | None = None

_SEC_COMPANY_TICKERS_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
_SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
_ALPHA_VANTAGE_QUERY_URL = "https://www.alphavantage.co/query"
_ADR_MAPPING_PATH = Path(getattr(config, "ADR_MAPPING_TABLE_PATH", "data/adr_mappings.csv"))
_ALPHA_ATTEMPT_LEDGER_PATH = Path(
    getattr(config, "ALPHA_VANTAGE_ADR_ATTEMPT_LEDGER_PATH", "feature_cache/alpha_vantage_adr_attempts.json")
)
_SEC_COMPANY_TICKER_CACHE: dict[str, dict] | None = None
_SEC_COMPANYFACTS_CACHE: dict[int, dict] = {}
_ADR_MAPPING_TABLE_CACHE: dict[str, dict] | None = None

# Local-listing -> ADR mappings used only for fundamentals enrichment.  Keeping
# these local avoids changing trading/cross-listing behaviour elsewhere.
_SEC_EDGAR_ADR_OVERRIDES: dict[str, str] = {
    "ASML.AS": "ASML",
    "DGE.L": "DEO",
    "NOKIA.HE": "NOK",
}

_ALPHA_FIELD_MAPS: dict[str, dict[str, tuple[str, ...]]] = {
    "income": {
        "net_income": ("netIncome", "netIncomeFromContinuingOperations"),
        "gross_profit": ("grossProfit",),
        "revenue": ("totalRevenue", "reportedCurrencyTotalRevenue"),
        "cost_of_revenue": ("costOfRevenue", "costofGoodsAndServicesSold"),
        "ebit": ("ebit", "operatingIncome"),
        "ebitda": ("ebitda",),
    },
    "balance": {
        "total_assets": ("totalAssets",),
        "cash": ("cashAndCashEquivalentsAtCarryingValue", "cashAndShortTermInvestments"),
        "long_term_debt": ("longTermDebt", "longTermDebtNoncurrent"),
        "short_term_debt": ("shortTermDebt", "currentDebt"),
        "total_debt": ("shortLongTermDebtTotal", "totalDebt"),
        "current_assets": ("totalCurrentAssets",),
        "current_liabilities": ("totalCurrentLiabilities",),
        "shares_outstanding": ("commonStockSharesOutstanding",),
    },
    "cashflow": {
        "operating_cashflow": ("operatingCashflow", "netCashProvidedByOperatingActivities"),
        "capital_expenditure": ("capitalExpenditures",),
    },
}

_SEC_FORMS = {"10-K", "10-Q", "20-F", "40-F", "6-K"}
_SEC_TAXONOMIES = ("ifrs-full", "us-gaap")
_SEC_CURRENCY_UNITS_EXCLUDE = {"shares", "pure"}

_SEC_FIELD_TAGS: dict[str, tuple[str, ...]] = {
    "gross_profit": ("GrossProfit", "GrossProfitLoss"),
    "revenue": (
        "Revenue",
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ),
    "cost_of_revenue": ("CostOfSales", "CostOfRevenue", "CostOfGoodsAndServicesSold"),
    "net_income": ("ProfitLoss", "NetIncomeLoss"),
    "operating_cashflow": (
        "CashFlowsFromUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "capital_expenditure": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquirePropertyPlantAndEquipmentClassifiedAsInvestingActivities",
        "PaymentsToAcquireProductiveAssets",
    ),
    "ebit": ("ProfitLossFromOperatingActivities", "OperatingIncomeLoss"),
    "ebitda": ("EarningsBeforeInterestTaxesDepreciationAndAmortization",),
    "total_assets": ("Assets",),
    "cash": ("CashAndCashEquivalents", "CashAndCashEquivalentsAtCarryingValue"),
    "total_debt": (
        "Borrowings",
        "DebtCurrentAndNoncurrent",
        "LongTermDebtAndFinanceLeaseObligationsCurrentAndNoncurrent",
    ),
    "current_assets": ("AssetsCurrent",),
    "current_liabilities": ("LiabilitiesCurrent",),
    "shares_outstanding": ("EntityCommonStockSharesOutstanding",),
}

_SEC_INSTANT_FIELDS = {
    "total_assets",
    "cash",
    "total_debt",
    "current_assets",
    "current_liabilities",
    "shares_outstanding",
}

_YAHOO_TIMESERIES_TYPES: tuple[str, ...] = (
    "quarterlyGrossProfit",
    "quarterlyTotalAssets",
    "quarterlyNetIncome",
    "quarterlyOperatingCashFlow",
    "quarterlyCapitalExpenditure",
    "quarterlyTotalDebt",
    "quarterlyCashAndCashEquivalents",
    "quarterlyTotalRevenue",
    "quarterlyEBIT",
    "quarterlyEBITDA",
    "trailingGrossProfit",
    "trailingTotalAssets",
    "trailingNetIncome",
    "trailingOperatingCashFlow",
    "trailingTotalRevenue",
)

_YAHOO_TIMESERIES_FIELD_MAP: dict[str, str] = {
    "quarterlyGrossProfit": "gross_profit",
    "trailingGrossProfit": "gross_profit",
    "quarterlyTotalAssets": "total_assets",
    "trailingTotalAssets": "total_assets",
    "quarterlyNetIncome": "net_income",
    "trailingNetIncome": "net_income",
    "quarterlyOperatingCashFlow": "operating_cashflow",
    "trailingOperatingCashFlow": "operating_cashflow",
    "quarterlyCapitalExpenditure": "capital_expenditure",
    "quarterlyTotalDebt": "total_debt",
    "quarterlyCashAndCashEquivalents": "cash",
    "quarterlyTotalRevenue": "revenue",
    "trailingTotalRevenue": "revenue",
    "quarterlyEBIT": "ebit",
    "quarterlyEBITDA": "ebitda",
}


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


def _http_user_agent() -> str:
    return str(
        getattr(
            config,
            "SEC_EDGAR_USER_AGENT",
            "TradingApp/1.0 contact@example.com",
        )
    )


def _sec_get_json(url: str, *, timeout: int = 20) -> dict | None:
    headers = {
        "User-Agent": _http_user_agent(),
        "Accept-Encoding": "gzip, deflate",
    }
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            if response.status_code != 200:
                logger.debug("SEC EDGAR request %s returned status %s", url, response.status_code)
                return None
            return response.json()
        except Exception as exc:
            if attempt >= 2:
                logger.debug("SEC EDGAR request failed for %s: %s", url, exc)
                return None
            time.sleep(0.5 * (attempt + 1))
    return None


def _sec_company_ticker_map() -> dict[str, dict]:
    global _SEC_COMPANY_TICKER_CACHE
    if _SEC_COMPANY_TICKER_CACHE is not None:
        return _SEC_COMPANY_TICKER_CACHE
    payload = _sec_get_json(_SEC_COMPANY_TICKERS_EXCHANGE_URL) or {}
    fields = payload.get("fields") or []
    rows = payload.get("data") or []
    try:
        indexes = {name: fields.index(name) for name in ("cik", "name", "ticker", "exchange")}
    except ValueError:
        _SEC_COMPANY_TICKER_CACHE = {}
        return _SEC_COMPANY_TICKER_CACHE
    out: dict[str, dict] = {}
    for row in rows:
        try:
            ticker = str(row[indexes["ticker"]]).upper().strip()
            if not ticker:
                continue
            out[ticker] = {
                "cik": int(row[indexes["cik"]]),
                "name": str(row[indexes["name"]]),
                "ticker": ticker,
                "exchange": str(row[indexes["exchange"]]),
            }
        except Exception:
            continue
    _SEC_COMPANY_TICKER_CACHE = out
    return out


def _sec_companyfacts(cik: int) -> dict | None:
    cik = int(cik)
    if cik in _SEC_COMPANYFACTS_CACHE:
        return _SEC_COMPANYFACTS_CACHE[cik]
    payload = _sec_get_json(_SEC_COMPANYFACTS_URL.format(cik=cik))
    if payload:
        _SEC_COMPANYFACTS_CACHE[cik] = payload
    return payload


def _load_adr_mapping_table(path: Path | None = None) -> dict[str, dict]:
    """Load local-listing -> ADR metadata from the maintainable CSV artifact."""
    global _ADR_MAPPING_TABLE_CACHE
    target = Path(path or _ADR_MAPPING_PATH)
    if path is None and _ADR_MAPPING_TABLE_CACHE is not None:
        return _ADR_MAPPING_TABLE_CACHE
    if not target.exists():
        if path is None:
            _ADR_MAPPING_TABLE_CACHE = {}
        return {}
    out: dict[str, dict] = {}
    try:
        with open(target, "r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                local = str(row.get("local_symbol") or "").upper().strip()
                adr = str(row.get("adr_symbol") or "").upper().strip()
                if not local or not adr:
                    continue
                out[local] = {
                    "local_symbol": local,
                    "adr_symbol": adr,
                    "adr_exchange": str(row.get("adr_exchange") or "").upper().strip(),
                    "sec_cik": str(row.get("sec_cik") or "").strip(),
                    "notes": str(row.get("notes") or "").strip(),
                }
    except Exception as exc:
        logger.warning("ADR mapping table read failed for %s: %s", target, exc)
        out = {}
    if path is None:
        _ADR_MAPPING_TABLE_CACHE = out
    return out


def _adr_mapping_row_for_ticker(ticker: str) -> dict | None:
    symbol = str(ticker or "").upper().strip()
    if not symbol or "." not in symbol:
        return None
    mapping: dict[str, dict] = {}
    try:
        from utils.global_universe import ADR_MAPPING

        mapping.update(
            {
                str(k).upper(): {"local_symbol": str(k).upper(), "adr_symbol": str(v).upper()}
                for k, v in ADR_MAPPING.items()
            }
        )
    except Exception:
        pass
    mapping.update(_load_adr_mapping_table())
    mapping.update(
        {
            str(k).upper(): {"local_symbol": str(k).upper(), "adr_symbol": str(v).upper()}
            for k, v in _SEC_EDGAR_ADR_OVERRIDES.items()
        }
    )
    config_overrides = getattr(config, "SEC_EDGAR_ADR_MAPPING_OVERRIDES", {}) or {}
    if isinstance(config_overrides, Mapping):
        mapping.update(
            {
                str(k).upper(): {"local_symbol": str(k).upper(), "adr_symbol": str(v).upper()}
                for k, v in config_overrides.items()
            }
        )
    row = mapping.get(symbol)
    if not row:
        return None
    adr = str(row.get("adr_symbol") or "").upper().strip()
    return dict(row, local_symbol=symbol, adr_symbol=adr) if adr else None


def _sec_adr_symbol_for_ticker(ticker: str) -> str | None:
    row = _adr_mapping_row_for_ticker(ticker)
    return str(row.get("adr_symbol") or "").upper() if row else None


def _snapshot_has_qmj_minimum(snapshot: Mapping | None) -> bool:
    if not isinstance(snapshot, Mapping):
        return False
    gross_profit = _finite(snapshot.get("gross_profit"))
    total_assets = _finite(snapshot.get("total_assets"))
    return gross_profit is not None and total_assets is not None and total_assets > 0


def _latest_snapshot_has_qmj_minimum(ticker: str, *, as_of: str | date | datetime | None = None) -> bool:
    store = _load_store()
    entries = store.get("tickers", {}).get(str(ticker or "").upper().strip())
    if not entries:
        return False
    cutoff = _coerce_date(as_of) or date.today()
    candidates: list[tuple[date, str, dict]] = []
    lag_days = int(getattr(config, "PIT_FUNDAMENTAL_LAG_DAYS", 45))
    for rd_str, payload in entries.items():
        try:
            rd = date.fromisoformat(str(rd_str)[:10])
        except ValueError:
            continue
        if _available_date(payload, rd, lag_days) <= cutoff:
            candidates.append((rd, rd_str, payload))
    best = _select_best_available_snapshot(candidates)
    if best is None:
        return False
    return _snapshot_has_qmj_minimum(entries.get(best[1]))


def _existing_snapshot_for_report_date(ticker: str, report_date: str) -> dict | None:
    store = _load_store()
    payload = (
        store.get("tickers", {})
        .get(str(ticker or "").upper().strip(), {})
        .get(str(report_date or "")[:10])
    )
    return dict(payload) if isinstance(payload, Mapping) else None



def _sec_fact_value(record: Mapping | None) -> float | None:
    if not isinstance(record, Mapping):
        return None
    return _float(record.get("val"))


def _sec_fact_duration_days(record: Mapping | None) -> int | None:
    if not isinstance(record, Mapping):
        return None
    start = _coerce_date(record.get("start"))
    end = _coerce_date(record.get("end"))
    if start is None or end is None:
        return None
    return (end - start).days


def _sec_fact_score(record: Mapping, *, instant: bool) -> tuple:
    form = str(record.get("form") or "").upper()
    fp = str(record.get("fp") or "").upper()
    filed = str(record.get("filed") or "")
    duration = _sec_fact_duration_days(record)
    form_score = {
        "20-F": 5,
        "40-F": 5,
        "10-K": 4,
        "10-Q": 3,
        "6-K": 2,
    }.get(form, 0)
    if instant:
        duration_score = 0
    elif duration is None:
        duration_score = -1
    elif 250 <= duration <= 460:
        duration_score = 3
    elif 70 <= duration <= 120:
        duration_score = 2
    else:
        duration_score = 0
    return (
        form_score,
        1 if fp == "FY" else 0,
        duration_score,
        filed,
    )


def _iter_sec_concept_facts(companyfacts: Mapping, field: str) -> Iterable[Mapping]:
    facts = companyfacts.get("facts") if isinstance(companyfacts, Mapping) else None
    if not isinstance(facts, Mapping):
        return []
    tags = _SEC_FIELD_TAGS.get(field, ())
    unit_kind = "shares" if field == "shares_outstanding" else "currency"
    records: list[Mapping] = []
    for taxonomy in _SEC_TAXONOMIES:
        taxonomy_facts = facts.get(taxonomy) or {}
        if not isinstance(taxonomy_facts, Mapping):
            continue
        for tag in tags:
            concept = taxonomy_facts.get(tag)
            if not isinstance(concept, Mapping):
                continue
            units = concept.get("units") or {}
            if not isinstance(units, Mapping):
                continue
            for unit, values in units.items():
                unit_norm = str(unit or "").lower()
                if unit_kind == "shares":
                    if unit_norm != "shares":
                        continue
                elif unit_norm in _SEC_CURRENCY_UNITS_EXCLUDE:
                    continue
                if not isinstance(values, list):
                    continue
                for record in values:
                    if not isinstance(record, Mapping):
                        continue
                    if str(record.get("form") or "").upper() not in _SEC_FORMS:
                        continue
                    if _sec_fact_value(record) is None:
                        continue
                    if not record.get("end"):
                        continue
                    records.append(record)
    return records


def _sec_best_facts_by_end(companyfacts: Mapping, field: str) -> dict[str, Mapping]:
    instant = field in _SEC_INSTANT_FIELDS
    best: dict[str, Mapping] = {}
    for record in _iter_sec_concept_facts(companyfacts, field):
        end = str(record.get("end") or "")[:10]
        if not end:
            continue
        existing = best.get(end)
        if existing is None or _sec_fact_score(record, instant=instant) > _sec_fact_score(existing, instant=instant):
            best[end] = record
    return best


def _derive_gross_profit(snapshot: dict) -> None:
    if snapshot.get("gross_profit") is not None:
        snapshot["_gross_profit_source"] = "direct_sec_xbrl"
        return
    revenue = _finite(snapshot.get("revenue"))
    cost = _finite(snapshot.get("cost_of_revenue"))
    if revenue is None or cost is None:
        return
    snapshot["gross_profit"] = revenue + cost if cost < 0 else revenue - cost
    snapshot["_gross_profit_source"] = "revenue_minus_cost_of_revenue"


def _snapshots_from_sec_companyfacts(
    companyfacts: Mapping,
    *,
    adr_symbol: str,
    cik: int | None = None,
    entity_name: str | None = None,
    limit: int | None = None,
) -> list[tuple[str, dict, str | None]]:
    """Normalize SEC companyfacts XBRL into PIT snapshots.

    Returns ``(period_end, snapshot, filed_date)`` rows.  The snapshot is
    source-tagged by ``record_snapshot(..., source='sec_edgar')`` at write time.
    """
    by_date: dict[str, dict] = {}
    filed_dates: dict[str, list[str]] = {}
    for field in _SEC_FIELD_TAGS:
        best = _sec_best_facts_by_end(companyfacts, field)
        for period_date, record in best.items():
            value = _sec_fact_value(record)
            if value is None:
                continue
            row = by_date.setdefault(period_date, {})
            row[field] = value
            filed = str(record.get("filed") or "")[:10]
            if filed:
                filed_dates.setdefault(period_date, []).append(filed)

    rows: list[tuple[str, dict, str | None]] = []
    for period_date in sorted(by_date, reverse=True):
        snapshot = dict(by_date[period_date])
        _derive_gross_profit(snapshot)
        snapshot.pop("cost_of_revenue", None)
        if not snapshot:
            continue
        snapshot["_sec_adr_symbol"] = str(adr_symbol).upper()
        if cik is not None:
            snapshot["_sec_cik"] = str(int(cik))
        if entity_name:
            snapshot["_sec_entity_name"] = str(entity_name)
        accepted = max(filed_dates.get(period_date) or []) if filed_dates.get(period_date) else None
        rows.append((period_date, snapshot, accepted))
    max_rows = int(limit or 0)
    return rows[:max_rows] if max_rows > 0 else rows


def backfill_via_sec_edgar(
    tickers: Iterable[str],
    *,
    limit: int | None = None,
    sleep_seconds: float = 0.1,
) -> dict[str, int]:
    """Backfill local non-US tickers from SEC companyfacts through ADR mapping.

    Snapshots are written under the local ticker and tagged ``source='sec_edgar'``.
    This is intentionally mapping-gated: plain US tickers continue to use FMP,
    while local listings only use SEC when we have an explicit ADR relationship.
    """
    limit = int(limit or 16)
    company_map = _sec_company_ticker_map()
    results: dict[str, int] = {}
    for ticker in tickers:
        symbol = str(ticker or "").upper().strip()
        if not symbol:
            continue
        adr = _sec_adr_symbol_for_ticker(symbol)
        if not adr:
            results[symbol] = 0
            continue
        listing = company_map.get(adr.upper())
        if not listing:
            results[symbol] = 0
            continue
        cik = int(listing["cik"])
        payload = _sec_companyfacts(cik)
        if not payload:
            results[symbol] = 0
            continue
        written = 0
        rows = _snapshots_from_sec_companyfacts(
            payload,
            adr_symbol=adr,
            cik=cik,
            entity_name=payload.get("entityName") or listing.get("name"),
            limit=limit,
        )
        for period_date, snapshot, accepted_date in rows:
            # Avoid writing SEC rows that cannot at least anchor a balance sheet.
            if _finite(snapshot.get("total_assets")) is None:
                continue
            record_snapshot(
                symbol,
                period_date,
                snapshot,
                accepted_date=accepted_date,
                source="sec_edgar",
            )
            written += 1
        results[symbol] = written
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return results


def _yahoo_timeseries_url(ticker: str) -> str:
    symbol = str(ticker or "").upper().strip()
    return f"https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{symbol}"


def _fetch_yahoo_timeseries(ticker: str, *, years: int = 6) -> dict | None:
    end = int(time.time())
    start = end - int(max(1, years) * 365.25 * 24 * 3600)
    params = {
        "symbol": str(ticker or "").upper().strip(),
        "type": ",".join(_YAHOO_TIMESERIES_TYPES),
        "period1": start,
        "period2": end,
        "lang": "en-US",
        "region": "US",
    }
    headers = {"User-Agent": "TradingApp/1.0 fundamentals-timeseries probe"}
    for attempt in range(2):
        try:
            response = requests.get(_yahoo_timeseries_url(ticker), params=params, headers=headers, timeout=15)
            if response.status_code in {429, 500, 502, 503, 504} and attempt == 0:
                time.sleep(0.5)
                continue
            if response.status_code != 200:
                logger.debug("Yahoo timeseries %s returned status %s", ticker, response.status_code)
                return None
            return response.json()
        except Exception as exc:
            if attempt:
                logger.debug("Yahoo timeseries failed for %s: %s", ticker, exc)
                return None
            time.sleep(0.5)
    return None


def _reported_value(entry: Mapping) -> float | None:
    reported = entry.get("reportedValue") if isinstance(entry, Mapping) else None
    if isinstance(reported, Mapping):
        return _float(reported.get("raw"))
    return None


def _snapshots_from_yahoo_timeseries_payload(
    payload: Mapping,
    *,
    limit: int | None = None,
) -> list[tuple[str, dict]]:
    result = ((payload.get("timeseries") or {}).get("result") or []) if isinstance(payload, Mapping) else []
    by_date: dict[str, dict] = {}
    type_by_field: dict[str, set[str]] = {}
    if not isinstance(result, list):
        return []
    for block in result:
        if not isinstance(block, Mapping):
            continue
        typelist = (block.get("meta") or {}).get("type") or []
        if isinstance(typelist, str):
            typelist = [typelist]
        for yahoo_type in typelist:
            field = _YAHOO_TIMESERIES_FIELD_MAP.get(str(yahoo_type))
            if not field:
                continue
            values = block.get(str(yahoo_type)) or []
            if not isinstance(values, list):
                continue
            for entry in values:
                if not isinstance(entry, Mapping):
                    continue
                period_date = str(entry.get("asOfDate") or "")[:10]
                value = _reported_value(entry)
                if not period_date or value is None:
                    continue
                row = by_date.setdefault(period_date, {})
                # Prefer trailing profitability/flow fields when available:
                # annualized gross profit over point-in-time assets is the
                # Novy-Marx-style GPA measure we need for QMJ.
                field_key = f"{period_date}:{field}"
                existing_types = type_by_field.get(field_key, set())
                has_trailing = any(str(item).startswith("trailing") for item in existing_types)
                if has_trailing and not str(yahoo_type).startswith("trailing"):
                    continue
                row[field] = value
                type_by_field.setdefault(field_key, set()).add(str(yahoo_type))

    instant_fields = {"total_assets", "cash", "total_debt"}
    last_instants: dict[str, tuple[float, str]] = {}
    for period_date in sorted(by_date):
        row = by_date[period_date]
        carried_from: set[str] = set()
        for field in instant_fields:
            if field not in row and field in last_instants:
                value, source_date = last_instants[field]
                row[field] = value
                carried_from.add(f"{field}:{source_date}")
        if carried_from:
            row["_yahoo_instant_fields_carried_forward_from"] = ",".join(sorted(carried_from))
        for field in instant_fields:
            value = _finite(row.get(field))
            if value is not None:
                last_instants[field] = (value, period_date)

    rows: list[tuple[str, dict]] = []
    for period_date in sorted(by_date, reverse=True):
        snapshot = dict(by_date[period_date])
        if not snapshot:
            continue
        snapshot["_yahoo_timeseries_types"] = ",".join(
            sorted(
                {
                    typ
                    for key, types in type_by_field.items()
                    if key.startswith(f"{period_date}:")
                    for typ in types
                }
            )
        )
        rows.append((period_date, snapshot))
    max_rows = int(limit or 0)
    return rows[:max_rows] if max_rows > 0 else rows


def backfill_via_yahoo_timeseries(
    tickers: Iterable[str],
    *,
    limit: int | None = None,
    sleep_seconds: float = 0.0,
) -> dict[str, int]:
    """Best-effort non-US PIT widening from Yahoo fundamentals-timeseries.

    Yahoo does not provide accepted filing dates through this endpoint, so
    snapshots intentionally rely on the PIT store's ``report_date + lag`` rule.
    """
    limit = int(limit or 16)
    results: dict[str, int] = {}
    for ticker in tickers:
        symbol = str(ticker or "").upper().strip()
        if not symbol:
            continue
        payload = _fetch_yahoo_timeseries(symbol)
        written = 0
        if payload:
            for period_date, snapshot in _snapshots_from_yahoo_timeseries_payload(payload, limit=limit):
                if not snapshot:
                    continue
                record_snapshot(symbol, period_date, snapshot, source="yahoo_timeseries")
                written += 1
        results[symbol] = written
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return results


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


def _alpha_get_json(
    function: str,
    symbol: str,
    *,
    api_key: str | None = None,
    session=None,
) -> dict | None:
    token = api_key if api_key is not None else str(getattr(config, "AV_TOKEN", "") or "")
    if not token:
        return {"_error": "missing_api_key"}
    client = session or requests
    try:
        response = client.get(
            _ALPHA_VANTAGE_QUERY_URL,
            params={"function": function, "symbol": symbol, "apikey": token},
            timeout=25,
        )
        if response.status_code != 200:
            return {"_error": f"http_{response.status_code}", "_http_status": response.status_code}
        payload = response.json()
    except Exception as exc:
        return {"_error": f"request_failed: {exc}"}
    return payload if isinstance(payload, dict) else {"_error": "non_object_payload"}


def _alpha_payload_status(payload: Mapping | None) -> str | None:
    if not isinstance(payload, Mapping):
        return "missing_payload"
    if payload.get("_error"):
        return str(payload.get("_error"))
    if payload.get("Information") or payload.get("Note"):
        return "rate_limited"
    if payload.get("Error Message"):
        return "symbol_error"
    return None


def _alpha_reports_by_date(payload: Mapping | None) -> dict[str, Mapping]:
    if not isinstance(payload, Mapping):
        return {}
    rows: dict[str, Mapping] = {}
    for section in ("quarterlyReports", "annualReports"):
        reports = payload.get(section) or []
        if not isinstance(reports, list):
            continue
        for report in reports:
            if not isinstance(report, Mapping):
                continue
            period_date = str(report.get("fiscalDateEnding") or "")[:10]
            if not period_date:
                continue
            enriched = dict(report)
            enriched["_alpha_report_section"] = section
            rows.setdefault(period_date, enriched)
    return rows


def _alpha_report_value(row: Mapping | None, names: tuple[str, ...]) -> float | None:
    if not isinstance(row, Mapping):
        return None
    for name in names:
        value = _float(row.get(name))
        if value is not None:
            return value
    return None


def _snapshot_from_alpha_reports(
    income: Mapping | None,
    balance: Mapping | None = None,
    cashflow: Mapping | None = None,
) -> dict:
    snapshot: dict[str, float | str] = {}
    for section, row in (("income", income), ("balance", balance), ("cashflow", cashflow)):
        for field, names in _ALPHA_FIELD_MAPS[section].items():
            value = _alpha_report_value(row, names)
            if value is not None:
                snapshot[field] = value
    short_debt = _finite(snapshot.get("short_term_debt"))
    long_debt = _finite(snapshot.get("long_term_debt"))
    if snapshot.get("total_debt") is None and (short_debt is not None or long_debt is not None):
        snapshot["total_debt"] = (short_debt or 0.0) + (long_debt or 0.0)
    revenue = _finite(snapshot.get("revenue"))
    cost = _finite(snapshot.get("cost_of_revenue"))
    if snapshot.get("gross_profit") is None and revenue is not None and cost is not None:
        snapshot["gross_profit"] = revenue + cost if cost < 0 else revenue - cost
        snapshot["_gross_profit_source"] = "alpha_revenue_minus_cost_of_revenue"
    elif snapshot.get("gross_profit") is not None:
        snapshot["_gross_profit_source"] = "direct_alpha_vantage"
    sections = sorted(
        {
            str(row.get("_alpha_report_section"))
            for row in (income, balance, cashflow)
            if isinstance(row, Mapping) and row.get("_alpha_report_section")
        }
    )
    if sections:
        snapshot["_alpha_vantage_report_sections"] = ",".join(sections)
    return {k: v for k, v in snapshot.items() if v is not None}


def _snapshots_from_alpha_vantage_payloads(
    income_payload: Mapping | None,
    balance_payload: Mapping | None = None,
    cashflow_payload: Mapping | None = None,
    *,
    adr_symbol: str,
    limit: int | None = None,
) -> list[tuple[str, dict]]:
    income_by_date = _alpha_reports_by_date(income_payload)
    balance_by_date = _alpha_reports_by_date(balance_payload)
    cashflow_by_date = _alpha_reports_by_date(cashflow_payload)
    dates = sorted(
        set(income_by_date) | set(balance_by_date) | set(cashflow_by_date),
        reverse=True,
    )
    rows: list[tuple[str, dict]] = []
    for period_date in dates:
        snapshot = _snapshot_from_alpha_reports(
            income_by_date.get(period_date),
            balance_by_date.get(period_date),
            cashflow_by_date.get(period_date),
        )
        if not snapshot:
            continue
        snapshot["_alpha_vantage_adr_symbol"] = str(adr_symbol or "").upper()
        rows.append((period_date, snapshot))
    max_rows = int(limit or 0)
    return rows[:max_rows] if max_rows > 0 else rows


def _merge_alpha_snapshot(base: Mapping | None, supplement: Mapping) -> tuple[dict, int]:
    merged = dict(base or {})
    additions = 0
    for key, value in supplement.items():
        if key.startswith("_"):
            continue
        if value is not None and merged.get(key) is None:
            merged[key] = value
            additions += 1
    metadata = {
        "_alpha_vantage_adr_symbol": supplement.get("_alpha_vantage_adr_symbol"),
        "_alpha_vantage_report_sections": supplement.get("_alpha_vantage_report_sections"),
        "_gross_profit_source": supplement.get("_gross_profit_source") or merged.get("_gross_profit_source"),
        "_alpha_vantage_base_source": (base or {}).get("_source") if isinstance(base, Mapping) else None,
    }
    for key, value in metadata.items():
        if value:
            merged[key] = value
    return merged, additions


def _load_alpha_attempt_ledger(path: str | Path | None = None) -> dict:
    target = Path(path or _ALPHA_ATTEMPT_LEDGER_PATH)
    if not target.exists():
        return {"version": 1, "daily_calls": {}, "tickers": {}}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "daily_calls": {}, "tickers": {}}
    if not isinstance(data, dict):
        return {"version": 1, "daily_calls": {}, "tickers": {}}
    data.setdefault("version", 1)
    data.setdefault("daily_calls", {})
    data.setdefault("tickers", {})
    return data


def _write_alpha_attempt_ledger(ledger: Mapping, path: str | Path | None = None) -> None:
    target = Path(path or _ALPHA_ATTEMPT_LEDGER_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, dict(ledger), indent=2)


def _alpha_cooldown_active(record: Mapping | None, now: datetime) -> bool:
    if not isinstance(record, Mapping):
        return False
    cooldown_until = _coerce_date(record.get("cooldown_until"))
    return bool(cooldown_until and cooldown_until > now.date())


def _alpha_cooldown_days(status: str) -> int:
    if status in {"rate_limited", "missing_api_key"}:
        return 1
    if status.startswith("http_") or status in {"symbol_error", "no_income_reports", "no_usable_reports"}:
        return 30
    return 7


def backfill_via_alpha_vantage_adr(
    tickers: Iterable[str],
    *,
    limit: int | None = None,
    sleep_seconds: float | None = None,
    daily_call_budget: int | None = None,
    attempt_ledger_path: str | Path | None = None,
    api_key: str | None = None,
    session=None,
    fetch_cashflow: bool | None = None,
) -> dict[str, int]:
    """Last-resort ADR supplement for residual non-US QMJ concept gaps.

    Alpha Vantage is intentionally not a broad non-US provider here.  The free
    tier is too small for that.  This path only runs for explicit local->ADR
    mappings, only when the current PIT record still lacks the QMJ minimum, and
    it records every attempt in a daily-call ledger.
    """
    limit = int(limit or 16)
    budget = int(
        daily_call_budget
        if daily_call_budget is not None
        else getattr(config, "ALPHA_VANTAGE_ADR_DAILY_CALL_BUDGET", 12)
    )
    sleep = (
        float(sleep_seconds)
        if sleep_seconds is not None
        else float(getattr(config, "ALPHA_VANTAGE_ADR_SLEEP_SECONDS", 13.0))
    )
    should_fetch_cashflow = (
        bool(fetch_cashflow)
        if fetch_cashflow is not None
        else bool(getattr(config, "ALPHA_VANTAGE_ADR_FETCH_CASHFLOW", True))
    )
    ledger = _load_alpha_attempt_ledger(attempt_ledger_path)
    calls_by_day = ledger.setdefault("daily_calls", {})
    records = ledger.setdefault("tickers", {})
    today_key = date.today().isoformat()
    calls_used = int(calls_by_day.get(today_key, 0) or 0)
    now = datetime.now()
    results: dict[str, int] = {}

    for ticker in tickers:
        symbol = str(ticker or "").upper().strip()
        if not symbol:
            continue
        results.setdefault(symbol, 0)
        mapping = _adr_mapping_row_for_ticker(symbol)
        if not mapping:
            continue
        adr_symbol = str(mapping.get("adr_symbol") or "").upper().strip()
        if not adr_symbol or _latest_snapshot_has_qmj_minimum(symbol):
            continue
        record = records.get(symbol, {}) if isinstance(records.get(symbol), Mapping) else {}
        if _alpha_cooldown_active(record, now):
            continue
        if calls_used >= budget:
            break

        payloads: dict[str, Mapping | None] = {}
        status: str | None = None
        calls_for_ticker = 0
        for function, key in (
            ("INCOME_STATEMENT", "income"),
            ("BALANCE_SHEET", "balance"),
            ("CASH_FLOW", "cashflow"),
        ):
            if key == "cashflow" and not should_fetch_cashflow:
                continue
            if calls_used >= budget:
                status = "budget_exhausted"
                break
            payload = _alpha_get_json(function, adr_symbol, api_key=api_key, session=session)
            calls_used += 1
            calls_for_ticker += 1
            payloads[key] = payload
            status = _alpha_payload_status(payload)
            if status:
                break
            if key == "income" and not _alpha_reports_by_date(payload):
                status = "no_income_reports"
                break
            if sleep > 0 and calls_used < budget:
                time.sleep(sleep)

        written = 0
        if not status or status == "budget_exhausted":
            rows = _snapshots_from_alpha_vantage_payloads(
                payloads.get("income"),
                payloads.get("balance"),
                payloads.get("cashflow"),
                adr_symbol=adr_symbol,
                limit=limit,
            )
            for period_date, supplement in rows:
                base = _existing_snapshot_for_report_date(symbol, period_date)
                merged, additions = _merge_alpha_snapshot(base, supplement)
                merged["_alpha_vantage_local_symbol"] = symbol
                merged["_alpha_vantage_supplemented_fields"] = additions
                if not base and not _snapshot_has_qmj_minimum(merged):
                    continue
                if base and additions <= 0 and not _snapshot_has_qmj_minimum(merged):
                    continue
                record_snapshot(
                    symbol,
                    period_date,
                    merged,
                    accepted_date=(base or {}).get("_accepted_date") if isinstance(base, Mapping) else None,
                    source="alpha_vantage_adr",
                )
                written += 1
            if written <= 0:
                status = "no_usable_reports"
            else:
                status = "written"

        results[symbol] = written
        cooldown_until = (now.date() + timedelta(days=_alpha_cooldown_days(status))).isoformat()
        records[symbol] = {
            **dict(record),
            "ticker": symbol,
            "adr_symbol": adr_symbol,
            "last_attempted_at": now.isoformat(timespec="seconds"),
            "last_status": status,
            "last_snapshots_written": written,
            "last_calls_used": calls_for_ticker,
            "cooldown_until": cooldown_until,
        }
        calls_by_day[today_key] = calls_used
        if status == "rate_limited":
            break

    calls_by_day[today_key] = calls_used
    _write_alpha_attempt_ledger(ledger, attempt_ledger_path)
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
    sec_edgar_fallback: bool = True,
    yahoo_timeseries_fallback: bool = True,
    yfinance_fallback: bool = True,
    alpha_vantage_adr_fallback: bool | None = None,
    alpha_vantage_adr_daily_call_budget: int | None = None,
    fmp_only: bool = False,
    write_results: bool = True,
) -> dict:
    """Backfill PIT fundamentals for tickers in the finalist refresh queue.

    Provider order is intentionally conservative:

    * FMP for plain US-style tickers.
    * SEC EDGAR for explicit local->ADR mappings.
    * Yahoo fundamentals-timeseries for broad non-US coverage.
    * yfinance quarterly statements as the final best-effort fallback.
    * Alpha Vantage ADR supplement only for mapped residual QMJ gaps.
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
    sec_edgar_results: dict[str, int] = {}
    yahoo_timeseries_results: dict[str, int] = {}
    yf_results: dict[str, int] = {}
    alpha_vantage_adr_results: dict[str, int] = {}
    fallback_tickers: list[str] = []
    alpha_enabled = (
        bool(getattr(config, "ALPHA_VANTAGE_ADR_FALLBACK_ENABLED", False))
        if alpha_vantage_adr_fallback is None
        else bool(alpha_vantage_adr_fallback)
    )
    for ticker in tickers:
        written = 0
        if _is_fmp_statement_candidate(ticker):
            try:
                written = backfill_ticker(ticker, limit=limit)
            except Exception as exc:
                logger.warning("Refresh queue FMP backfill failed for %s: %s", ticker, exc)
                written = 0
            fmp_results[ticker] = written

        if (
            not fmp_only
            and sec_edgar_fallback
            and written <= 0
            and _sec_adr_symbol_for_ticker(ticker)
        ):
            try:
                sec_written = backfill_via_sec_edgar([ticker], limit=limit or 16, sleep_seconds=0.0).get(ticker, 0)
            except Exception as exc:
                logger.warning("Refresh queue SEC EDGAR backfill failed for %s: %s", ticker, exc)
                sec_written = 0
            sec_edgar_results[ticker] = sec_written
            written += sec_written

        if (
            not fmp_only
            and yahoo_timeseries_fallback
            and written <= 0
            and "." in ticker
        ):
            try:
                yahoo_written = backfill_via_yahoo_timeseries([ticker], limit=limit or 16, sleep_seconds=0.0).get(ticker, 0)
            except Exception as exc:
                logger.warning("Refresh queue Yahoo timeseries backfill failed for %s: %s", ticker, exc)
                yahoo_written = 0
            yahoo_timeseries_results[ticker] = yahoo_written
            written += yahoo_written

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

    if alpha_enabled and not fmp_only:
        alpha_candidates = [
            ticker for ticker in tickers
            if _adr_mapping_row_for_ticker(ticker) and not _latest_snapshot_has_qmj_minimum(ticker)
        ]
        if alpha_candidates:
            alpha_vantage_adr_results = backfill_via_alpha_vantage_adr(
                alpha_candidates,
                limit=limit or 16,
                daily_call_budget=alpha_vantage_adr_daily_call_budget,
            )

    refreshed = {
        ticker: (
            (fmp_results.get(ticker, 0) or 0)
            + (sec_edgar_results.get(ticker, 0) or 0)
            + (yahoo_timeseries_results.get(ticker, 0) or 0)
            + (yf_results.get(ticker, 0) or 0)
            + (alpha_vantage_adr_results.get(ticker, 0) or 0)
        )
        for ticker in tickers
    }
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "queue_path": str(path),
        "fmp_only": bool(fmp_only),
        "sec_edgar_fallback": bool(sec_edgar_fallback and not fmp_only),
        "yahoo_timeseries_fallback": bool(yahoo_timeseries_fallback and not fmp_only),
        "yfinance_fallback": bool(yfinance_fallback),
        "alpha_vantage_adr_fallback": bool(alpha_enabled and not fmp_only),
        "provider_order": [
            "fmp",
            "sec_edgar",
            "yahoo_timeseries",
            "yfinance_quarterly",
            "alpha_vantage_adr",
        ],
        "selected": len(tickers),
        "refreshed": sum(1 for value in refreshed.values() if value > 0),
        "snapshots_written": sum(refreshed.values()),
        "fmp_results": fmp_results,
        "sec_edgar_results": sec_edgar_results,
        "yahoo_timeseries_results": yahoo_timeseries_results,
        "yfinance_results": yf_results,
        "alpha_vantage_adr_results": alpha_vantage_adr_results,
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
    candidates: list[tuple[date, str, dict]] = []
    for rd_str, payload in entries.items():
        try:
            rd = date.fromisoformat(str(rd_str)[:10])
        except ValueError:
            continue
        if _available_date(payload, rd, lag_days) <= cutoff:
            candidates.append((rd, rd_str, payload))
    best = _select_best_available_snapshot(candidates)
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
    parser.add_argument("--sec-edgar", action="store_true", help="Use SEC EDGAR companyfacts via local->ADR mapping")
    parser.add_argument("--yahoo-timeseries", action="store_true", help="Use Yahoo fundamentals-timeseries and tag snapshots as yahoo_timeseries")
    parser.add_argument("--alpha-vantage-adr", action="store_true", help="Use quota-budgeted Alpha Vantage ADR supplement")
    parser.add_argument("--backfill-factor-snapshot", action="store_true", help="Backfill PIT-safe factor snapshots into signal_backtest")
    parser.add_argument("--backfill-replay-sleeves", action="store_true", help="Backfill sleeve scalar columns from signal-time factor rows")
    parser.add_argument("--refresh-queue", action="store_true", help="Backfill PIT fundamentals for feature_cache/fundamental_refresh_queue.json")
    parser.add_argument("--queue-path", default=None, help="Override fundamental refresh queue path")
    parser.add_argument("--no-sec-edgar-fallback", action="store_true", help="Disable SEC EDGAR ADR fallback for refresh queue")
    parser.add_argument("--no-yahoo-timeseries-fallback", action="store_true", help="Disable Yahoo fundamentals-timeseries fallback for refresh queue")
    parser.add_argument("--no-yfinance-fallback", action="store_true", help="Disable yfinance quarterly fallback for refresh queue")
    parser.add_argument("--alpha-vantage-adr-fallback", action="store_true", help="Enable Alpha Vantage ADR supplement in refresh queue")
    parser.add_argument("--alpha-vantage-budget", type=int, default=None, help="Max Alpha Vantage calls to spend today")
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
            sec_edgar_fallback=not args.no_sec_edgar_fallback,
            yahoo_timeseries_fallback=not args.no_yahoo_timeseries_fallback,
            yfinance_fallback=not args.no_yfinance_fallback,
            alpha_vantage_adr_fallback=args.alpha_vantage_adr_fallback,
            alpha_vantage_adr_daily_call_budget=args.alpha_vantage_budget,
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
    elif args.sec_edgar:
        results = backfill_via_sec_edgar(tickers, limit=args.limit, sleep_seconds=args.sleep)
    elif args.yahoo_timeseries:
        results = backfill_via_yahoo_timeseries(tickers, limit=args.limit, sleep_seconds=args.sleep)
    elif args.alpha_vantage_adr:
        results = backfill_via_alpha_vantage_adr(
            tickers,
            limit=args.limit,
            sleep_seconds=args.sleep if args.sleep > 0 else None,
            daily_call_budget=args.alpha_vantage_budget,
        )
    else:
        results = backfill_tickers(tickers, limit=args.limit, sleep_seconds=args.sleep)
    total = sum(results.values())
    print(f"pit_snapshots_written={total} tickers={len(results)}")


if __name__ == "__main__":
    _main()
