"""Read-only diagnostics for missing non-US QMJ fundamentals.

The daily screener already records when a candidate is capped by thin
fundamental evidence.  This module turns those caps into an actionable routing
table: which names are ADR-addressable, which need official UK/ESEF filings,
which need a sector-specific quality contract, and which are likely paid-data
long tail.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from xml.etree import ElementTree as ET

import requests

import config
from utils.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)

OPENFIGI_MAPPING_URL = "https://api.openfigi.com/v3/mapping"
FILINGS_XBRL_API_BASE = "https://filings.xbrl.org/api"
FILINGS_XBRL_BASE = "https://filings.xbrl.org"
USER_AGENT = "TradingDashboardCoverageAudit/1.0"

XBRL_NS = "{http://www.xbrl.org/2003/instance}"
IX_NS = "{http://www.xbrl.org/2013/inlineXBRL}"

SUFFIX_COUNTRY: dict[str, str] = {
    ".L": "GB",
    ".AX": "AU",
    ".HK": "HK",
    ".SI": "SG",
    ".TO": "CA",
    ".T": "JP",
    ".DE": "DE",
    ".PA": "FR",
    ".AS": "NL",
    ".SW": "CH",
    ".ST": "SE",
    ".CO": "DK",
    ".MI": "IT",
    ".MC": "ES",
    ".HE": "FI",
    ".OL": "NO",
}

OPENFIGI_EXCH_CODE: dict[str, str] = {
    ".L": "LN",
    ".AX": "AU",
    ".HK": "HK",
    ".SI": "SP",
    ".TO": "CN",
    ".T": "JP",
    ".DE": "GY",
    ".PA": "FP",
    ".AS": "NA",
    ".SW": "SW",
    ".ST": "SS",
    ".CO": "DC",
    ".MI": "IM",
    ".MC": "SM",
    ".HE": "FH",
    ".OL": "NO",
}

ESEF_SUFFIXES = {".L", ".DE", ".PA", ".AS", ".SW", ".ST", ".CO", ".MI", ".MC", ".HE", ".OL"}
OFFICIAL_ROUTE_BY_SUFFIX: dict[str, str] = {
    ".L": "openfigi_to_esef_ixbrl_then_companies_house",
    ".DE": "openfigi_to_esef_ixbrl",
    ".PA": "openfigi_to_esef_ixbrl",
    ".AS": "openfigi_to_esef_ixbrl",
    ".SW": "openfigi_to_esef_ixbrl",
    ".ST": "openfigi_to_esef_ixbrl",
    ".CO": "openfigi_to_esef_ixbrl",
    ".MI": "openfigi_to_esef_ixbrl",
    ".MC": "openfigi_to_esef_ixbrl",
    ".HE": "openfigi_to_esef_ixbrl",
    ".OL": "openfigi_to_esef_ixbrl",
    ".T": "edinet_xbrl",
    ".AX": "asx_or_paid_global",
    ".HK": "hkex_or_paid_global",
    ".SI": "sgx_or_paid_global",
    ".TO": "sedar_or_adr_or_paid_global",
}

FUND_KEYWORDS = (
    "investment trust",
    "closed end",
    "closed-end",
    "fund",
    "ucits",
    "etf",
    "nav",
    "infrastructure",
)
BANK_KEYWORDS = ("bank", "banc", "banco", "bancorp", "banking")
INSURANCE_KEYWORDS = ("insurance", "insurer", "assurance", "reinsurance")
REIT_KEYWORDS = ("reit", "real estate investment")

ESEF_FIELD_TAGS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "ifrs-full:Revenue",
        "ifrs-full:RevenueFromContractsWithCustomers",
    ),
    "gross_profit": ("ifrs-full:GrossProfit",),
    "net_income": (
        "ifrs-full:ProfitLoss",
        "core:ProfitLoss",
    ),
    "operating_cashflow": (
        "ifrs-full:CashFlowsFromUsedInOperatingActivities",
        "ifrs-full:CashFlowsFromUsedInOperations",
        "ifrs-full:NetCashFlowsFromUsedInOperatingActivities",
    ),
    "capital_expenditure": (
        "ifrs-full:PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
        "ifrs-full:PaymentsToAcquirePropertyPlantAndEquipment",
        "ifrs-full:PaymentsToAcquirePropertyPlantAndEquipmentClassifiedAsInvestingActivities",
    ),
    "total_assets": ("ifrs-full:Assets",),
    "cash": ("ifrs-full:CashAndCashEquivalents",),
    "current_assets": ("ifrs-full:CurrentAssets",),
    "current_liabilities": ("ifrs-full:CurrentLiabilities",),
}

INSTANT_FIELDS = {"total_assets", "cash", "current_assets", "current_liabilities"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if out == out else None
    except (TypeError, ValueError):
        return None


def ticker_suffix(ticker: str) -> str:
    match = re.search(r"(\.[A-Z]{1,4})$", str(ticker or "").upper().strip())
    return match.group(1) if match else "US"


def ticker_root(ticker: str) -> str:
    symbol = str(ticker or "").upper().strip()
    suffix = ticker_suffix(symbol)
    return symbol[: -len(suffix)] if suffix != "US" else symbol


def load_cached_candidates(path: str | Path = "orchestrator_state.json") -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("cached_discovery") or []
    return [row for row in rows if isinstance(row, dict)]


def load_adr_mapping(path: str | Path | None = None) -> dict[str, dict[str, str]]:
    target = Path(path or getattr(config, "ADR_MAPPING_TABLE_PATH", "data/adr_mappings.csv"))
    out: dict[str, dict[str, str]] = {}
    if target.exists():
        with target.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                local = str(row.get("local_symbol") or "").upper().strip()
                adr = str(row.get("adr_symbol") or "").upper().strip()
                if local and adr:
                    out[local] = {
                        "adr_symbol": adr,
                        "adr_exchange": str(row.get("adr_exchange") or "").upper().strip(),
                        "sec_cik": str(row.get("sec_cik") or "").strip(),
                    }
    try:
        from utils.global_universe import ADR_MAPPING

        for local, adr in ADR_MAPPING.items():
            local_symbol = str(local or "").upper().strip()
            out.setdefault(
                local_symbol,
                {"adr_symbol": str(adr or "").upper().strip(), "adr_exchange": "", "sec_cik": ""},
            )
    except Exception:
        logger.debug("Unable to merge ADR mapping from global universe", exc_info=True)
    return out


def classify_instrument(row: Mapping[str, Any]) -> str:
    text = " ".join(
        str(row.get(key) or "")
        for key in ("ticker", "name", "sector", "industry", "exchange")
    ).lower()
    sector = str(row.get("sector") or "").lower()
    if any(token in text for token in REIT_KEYWORDS) or "real estate" in sector:
        return "reit_or_property"
    if any(token in text for token in FUND_KEYWORDS):
        return "fund_or_investment_trust"
    if any(token in text for token in INSURANCE_KEYWORDS):
        return "insurer"
    if any(token in text for token in BANK_KEYWORDS):
        return "bank"
    if "financial" in sector:
        return "financial_operating_or_unclassified"
    return "operating_company"


def qmj_contract_family(instrument_class: str) -> str:
    if instrument_class == "operating_company":
        return "standard_qmj"
    if instrument_class in {"bank", "insurer", "reit_or_property", "fund_or_investment_trust"}:
        return "sector_specific_quality"
    return "needs_name_or_industry_resolution"


def refined_instrument_class(row: Mapping[str, Any], openfigi_row: Mapping[str, Any]) -> str:
    """Classify with OpenFIGI name/security type when the cached row is sparse."""
    merged = dict(row)
    name = str(openfigi_row.get("name") or "").strip()
    security_type = str(openfigi_row.get("security_type") or "").strip()
    if name:
        merged["name"] = name
    if security_type:
        merged["industry"] = " ".join([str(merged.get("industry") or ""), security_type])
    return classify_instrument(merged)


def recommended_route(row: Mapping[str, Any], adr_mapping: Mapping[str, Mapping[str, str]]) -> str:
    ticker = str(row.get("ticker") or "").upper().strip()
    if _safe_float(row.get("qmj_factor_score")) is not None:
        return "covered"
    if ticker in adr_mapping:
        return "sec_edgar_then_alpha_vantage_adr"
    instrument = classify_instrument(row)
    if qmj_contract_family(instrument) == "sector_specific_quality":
        return "sector_specific_quality_contract"
    suffix = ticker_suffix(ticker)
    return OFFICIAL_ROUTE_BY_SUFFIX.get(suffix, "paid_global_or_manual_review")


def _priority(row: Mapping[str, Any]) -> float:
    sb_score = _safe_float(row.get("sb_score")) or 0.0
    final_rank = _safe_float(row.get("final_rank")) or 0.0
    prior_coverage = _safe_float(row.get("institutional_prior_coverage")) or 0.0
    action = str(row.get("action") or "").upper()
    ceiling = str(row.get("action_gate_ceiling") or "").upper()
    ready = str(row.get("ready_contract_status") or "").upper()
    reasons = " ".join(str(item) for item in (row.get("action_gate_reasons") or []))

    score = max(sb_score, 0.0) * 5.0 + final_rank * 2.0 + prior_coverage
    if action == "BUY":
        score += 1.0
    if ceiling == "STRONG BUY":
        score += 1.5
    elif ceiling == "BUY":
        score += 0.75
    if ready == "PASS":
        score += 1.0
    if "Thin yfinance quality evidence" in reasons:
        score += 0.75
    return round(score, 4)


def build_coverage_audit(
    candidates: Iterable[Mapping[str, Any]],
    *,
    adr_mapping: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    adr_mapping = adr_mapping or load_adr_mapping()
    items: list[dict[str, Any]] = []

    summary = {
        "total_candidates": 0,
        "non_us_candidates": 0,
        "non_us_qmj_null": 0,
        "non_us_thin_yfinance_quality": 0,
    }
    by_suffix: Counter[str] = Counter()
    qmj_null_by_suffix: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    contract_counts: Counter[str] = Counter()
    pit_source_counts: Counter[str] = Counter()

    for raw in candidates:
        ticker = str(raw.get("ticker") or raw.get("symbol") or "").upper().strip()
        if not ticker:
            continue
        suffix = ticker_suffix(ticker)
        summary["total_candidates"] += 1
        if suffix == "US":
            continue

        qmj = _safe_float(raw.get("qmj_factor_score"))
        qmj_missing = qmj is None
        reasons = [str(item) for item in (raw.get("action_gate_reasons") or [])]
        thin_quality = any("Thin yfinance quality evidence" in reason for reason in reasons)
        instrument_class = classify_instrument(raw)
        contract = qmj_contract_family(instrument_class)
        route = recommended_route(raw, adr_mapping)
        adr = adr_mapping.get(ticker) or {}
        pit_source = str(raw.get("pit_source") or raw.get("_pit_source") or "unknown")

        summary["non_us_candidates"] += 1
        by_suffix[suffix] += 1
        pit_source_counts[pit_source] += 1
        class_counts[instrument_class] += 1
        contract_counts[contract] += 1
        route_counts[route] += 1
        if qmj_missing:
            summary["non_us_qmj_null"] += 1
            qmj_null_by_suffix[suffix] += 1
        if thin_quality:
            summary["non_us_thin_yfinance_quality"] += 1

        items.append(
            {
                "ticker": ticker,
                "suffix": suffix,
                "country": SUFFIX_COUNTRY.get(suffix, ""),
                "sector": raw.get("sector") or "",
                "industry": raw.get("industry") or "",
                "exchange": raw.get("exchange") or "",
                "pit_source": pit_source,
                "qmj_factor_score": qmj,
                "qmj_component_count": raw.get("qmj_component_count"),
                "qmj_missing": qmj_missing,
                "thin_yfinance_quality_evidence": thin_quality,
                "instrument_class": instrument_class,
                "qmj_contract_family": contract,
                "recommended_route": route,
                "adr_symbol": adr.get("adr_symbol"),
                "adr_exchange": adr.get("adr_exchange"),
                "sec_cik": adr.get("sec_cik"),
                "action": raw.get("action"),
                "action_gate_ceiling": raw.get("action_gate_ceiling"),
                "ready_contract_status": raw.get("ready_contract_status"),
                "sb_score": _safe_float(raw.get("sb_score")),
                "final_rank": _safe_float(raw.get("final_rank")),
                "institutional_prior_coverage": _safe_float(raw.get("institutional_prior_coverage")),
                "action_gate_reasons": reasons,
                "priority": _priority(raw),
            }
        )

    items.sort(key=lambda item: (item["qmj_missing"], item["priority"]), reverse=True)
    summary["non_us_qmj_null_rate"] = round(
        summary["non_us_qmj_null"] / max(summary["non_us_candidates"], 1),
        4,
    )
    return {
        "generated_at": _now_iso(),
        "schema_version": 1,
        "purpose": "route missing non-US QMJ fundamentals to official/free/provider lanes",
        "summary": summary,
        "by_suffix": _counter_table(by_suffix, qmj_null_by_suffix),
        "route_counts": dict(route_counts.most_common()),
        "instrument_class_counts": dict(class_counts.most_common()),
        "qmj_contract_family_counts": dict(contract_counts.most_common()),
        "pit_source_counts": dict(pit_source_counts.most_common()),
        "items": items,
    }


def _counter_table(total: Counter[str], nulls: Counter[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, count in total.most_common():
        missing = nulls.get(key, 0)
        out[key] = {
            "total": count,
            "qmj_null": missing,
            "qmj_null_rate": round(missing / max(count, 1), 4),
        }
    return out


def write_coverage_audit(
    candidates: Iterable[Mapping[str, Any]],
    *,
    output_path: str | Path | None = None,
    adr_mapping: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    payload = build_coverage_audit(candidates, adr_mapping=adr_mapping)
    target = output_path or getattr(
        config,
        "NON_US_FUNDAMENTAL_COVERAGE_AUDIT_PATH",
        "feature_cache/non_us_fundamental_coverage_audit.json",
    )
    atomic_write_json(target, payload, indent=2)
    return payload


def _openfigi_jobs(items: list[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], dict[str, str] | None]]:
    jobs: list[tuple[Mapping[str, Any], dict[str, str] | None]] = []
    for item in items:
        suffix = str(item.get("suffix") or "")
        exch = OPENFIGI_EXCH_CODE.get(suffix)
        if not exch:
            jobs.append((item, None))
            continue
        jobs.append((item, {"idType": "TICKER", "idValue": ticker_root(str(item.get("ticker") or "")), "exchCode": exch}))
    return jobs


def _request_json(method: str, url: str, *, timeout: float = 20.0, **kwargs) -> tuple[int, Any]:
    headers = dict(kwargs.pop("headers", {}) or {})
    headers.setdefault("User-Agent", USER_AGENT)
    response = requests.request(method, url, timeout=timeout, headers=headers, **kwargs)
    status = int(response.status_code)
    try:
        return status, response.json()
    except ValueError:
        return status, {"raw_text": response.text[:500]}


def map_openfigi(items: list[Mapping[str, Any]], *, batch_size: int = 10, sleep_seconds: float = 0.0) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for start in range(0, len(items), batch_size):
        batch_items = items[start : start + batch_size]
        job_pairs = _openfigi_jobs(batch_items)
        supported_pairs = [(item, job) for item, job in job_pairs if job is not None]
        for item, job in job_pairs:
            if job is None:
                out.append({"ticker": item.get("ticker"), "status": "unsupported_suffix"})
        jobs = [job for _, job in supported_pairs if job is not None]
        if not jobs:
            continue
        try:
            status, payload = _request_json("POST", OPENFIGI_MAPPING_URL, json=jobs)
        except requests.RequestException as exc:
            for item, _job in supported_pairs:
                out.append({"ticker": item.get("ticker"), "status": "request_error", "error": str(exc)})
            continue

        if status != 200 or not isinstance(payload, list):
            for item, _job in supported_pairs:
                out.append({"ticker": item.get("ticker"), "status": f"http_{status}"})
            continue
        for (item, _job), result in zip(supported_pairs, payload):
            data = result.get("data") if isinstance(result, dict) else None
            first = data[0] if isinstance(data, list) and data else {}
            out.append(
                {
                    "ticker": item.get("ticker"),
                    "status": "mapped" if first else "not_found",
                    "figi": first.get("figi"),
                    "name": first.get("name"),
                    "openfigi_ticker": first.get("ticker"),
                    "exch_code": first.get("exchCode"),
                    "security_type": first.get("securityType"),
                }
            )
        if sleep_seconds > 0 and start + batch_size < len(items):
            time.sleep(sleep_seconds)
    return out


def _absolute_filings_url(path: str | None) -> str | None:
    if not path:
        return None
    return path if path.startswith("http") else f"{FILINGS_XBRL_BASE}{path}"


def probe_filings_xbrl_entity(name: str) -> dict[str, Any]:
    found = find_filings_xbrl_filings(name, limit=1)
    if found.get("status") != "entity_found":
        return found
    result: dict[str, Any] = {
        "status": "entity_found",
        "query_name": found.get("query_name"),
        "entity_name": found.get("entity_name"),
        "entity_identifier": found.get("entity_identifier"),
    }
    filings = found.get("filings") or []
    if not filings:
        result["latest_filing_status"] = found.get("filing_status") or "filing_not_found"
        return result
    result["latest_filing_status"] = "filing_found"
    result["latest_filing"] = filings[0]
    return result


def find_filings_xbrl_filings(name: str, *, limit: int = 4) -> dict[str, Any]:
    if not name:
        return {"status": "missing_name"}
    try:
        status, payload = _request_json(
            "GET",
            f"{FILINGS_XBRL_API_BASE}/entities",
            params={"filter[name]": name, "page[size]": 5},
        )
    except requests.RequestException as exc:
        return {"status": "request_error", "error": str(exc)}
    if status != 200 or not isinstance(payload, dict):
        return {"status": f"http_{status}"}
    data = payload.get("data") or []
    if not data:
        return {"status": "entity_not_found", "query_name": name}

    entity = data[0]
    attrs = entity.get("attributes") or {}
    identifier = attrs.get("identifier")
    result: dict[str, Any] = {
        "status": "entity_found",
        "query_name": name,
        "entity_name": attrs.get("name"),
        "entity_identifier": identifier,
        "filings": [],
    }
    if not identifier:
        return result
    try:
        filing_status, filing_payload = _request_json(
            "GET",
            f"{FILINGS_XBRL_API_BASE}/entities/{identifier}/filings",
            params={"page[size]": max(1, int(limit or 1)), "sort": "-processed"},
        )
    except requests.RequestException as exc:
        result.update({"filing_status": "request_error", "filing_error": str(exc)})
        return result
    if filing_status != 200 or not isinstance(filing_payload, dict):
        result["filing_status"] = f"http_{filing_status}"
        return result
    filings = filing_payload.get("data") or []
    if not filings:
        result["filing_status"] = "filing_not_found"
        return result
    result["filing_status"] = "filing_found"
    result["filings"] = []
    for filing in filings:
        filing_attrs = filing.get("attributes") or {}
        result["filings"].append(
            {
                "period_end": filing_attrs.get("period_end"),
                "processed": filing_attrs.get("processed"),
                "country": filing_attrs.get("country"),
                "error_count": filing_attrs.get("error_count"),
                "warning_count": filing_attrs.get("warning_count"),
                "report_url": _absolute_filings_url(filing_attrs.get("report_url")),
                "json_url": _absolute_filings_url(filing_attrs.get("json_url")),
                "package_url": _absolute_filings_url(filing_attrs.get("package_url")),
            }
        )
    return result


def probe_public_filing_routes(
    audit_payload: Mapping[str, Any],
    *,
    limit: int = 12,
    suffixes: set[str] | None = None,
    sleep_seconds: float = 0.2,
) -> dict[str, Any]:
    suffixes = suffixes or {".L"}
    candidates = [
        item
        for item in (audit_payload.get("items") or [])
        if item.get("qmj_missing")
        and item.get("suffix") in suffixes
        and "esef" in str(item.get("recommended_route") or "")
    ]
    candidates = sorted(candidates, key=lambda item: item.get("priority") or 0.0, reverse=True)[:limit]
    figi_rows = {row.get("ticker"): row for row in map_openfigi(candidates, sleep_seconds=sleep_seconds)}

    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for item in candidates:
        ticker = item.get("ticker")
        figi = figi_rows.get(ticker) or {"status": "not_probed"}
        refined_class = refined_instrument_class(item, figi) if figi.get("status") == "mapped" else item.get("instrument_class")
        refined_contract = qmj_contract_family(str(refined_class or ""))
        if figi.get("status") == "mapped":
            if refined_contract == "sector_specific_quality":
                filing = {"status": "skipped_sector_specific_quality_contract"}
            else:
                filing = probe_filings_xbrl_entity(str(figi.get("name") or ""))
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
        else:
            filing = {"status": "skipped_no_openfigi_name"}
        status = filing.get("latest_filing_status") or filing.get("status") or figi.get("status")
        status_counts[str(status)] += 1
        rows.append(
            {
                "ticker": ticker,
                "priority": item.get("priority"),
                "suffix": item.get("suffix"),
                "sector": item.get("sector"),
                "instrument_class": item.get("instrument_class"),
                "refined_instrument_class": refined_class,
                "refined_qmj_contract_family": refined_contract,
                "recommended_route": item.get("recommended_route"),
                "openfigi": figi,
                "filings_xbrl": filing,
            }
        )

    return {
        "generated_at": _now_iso(),
        "schema_version": 1,
        "purpose": "read-only OpenFIGI to filings.xbrl.org route probe for missing non-US QMJ",
        "probe_limit": limit,
        "suffixes": sorted(suffixes),
        "probed": len(rows),
        "status_counts": dict(status_counts.most_common()),
        "items": rows,
    }


def write_public_filing_probe(
    audit_payload: Mapping[str, Any],
    *,
    output_path: str | Path | None = None,
    limit: int = 12,
    suffixes: set[str] | None = None,
    sleep_seconds: float = 0.2,
) -> dict[str, Any]:
    payload = probe_public_filing_routes(
        audit_payload,
        limit=limit,
        suffixes=suffixes,
        sleep_seconds=sleep_seconds,
    )
    target = output_path or getattr(
        config,
        "NON_US_FUNDAMENTAL_FILING_PROBE_PATH",
        "feature_cache/non_us_fundamental_filing_probe.json",
    )
    atomic_write_json(target, payload, indent=2)
    return payload


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _child_text(parent: ET.Element, local_name: str) -> str | None:
    for child in parent.iter():
        if _local_name(child.tag) == local_name:
            return child.text
    return None


def _context_table(root: ET.Element) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    for elem in root.iter():
        if _local_name(elem.tag) != "context":
            continue
        context_id = elem.attrib.get("id")
        if not context_id:
            continue
        period = next((child for child in elem if _local_name(child.tag) == "period"), None)
        start_date = _child_text(period, "startDate") if period is not None else None
        end_date = _child_text(period, "endDate") if period is not None else None
        instant = _child_text(period, "instant") if period is not None else None
        has_dimensions = any(_local_name(child.tag) in {"explicitMember", "typedMember"} for child in elem.iter())
        contexts[context_id] = {
            "start_date": start_date,
            "end_date": end_date or instant,
            "instant": instant,
            "is_instant": bool(instant),
            "has_dimensions": has_dimensions,
        }
    return contexts


def _parse_ix_number(elem: ET.Element) -> float | None:
    text = "".join(elem.itertext()).strip()
    text = (
        text.replace("\u00a0", "")
        .replace(" ", "")
        .replace(",", "")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
    )
    if not text or text in {"-", "--", "."}:
        return None
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]
    try:
        value = float(text)
    except ValueError:
        return None
    if negative or elem.attrib.get("sign") == "-":
        value = -value
    scale = elem.attrib.get("scale")
    if scale not in (None, ""):
        try:
            value *= 10 ** int(scale)
        except ValueError:
            pass
    return value


def _fact_candidates(root: ET.Element, contexts: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for elem in root.iter():
        if _local_name(elem.tag) != "nonFraction":
            continue
        concept = elem.attrib.get("name")
        context_ref = elem.attrib.get("contextRef")
        value = _parse_ix_number(elem)
        if not concept or context_ref not in contexts or value is None:
            continue
        ctx = contexts[context_ref]
        rows.append(
            {
                "concept": concept,
                "context_ref": context_ref,
                "value": value,
                "start_date": ctx.get("start_date"),
                "end_date": ctx.get("end_date"),
                "is_instant": ctx.get("is_instant"),
                "has_dimensions": ctx.get("has_dimensions"),
            }
        )
    return rows


def _select_fact(
    facts: list[Mapping[str, Any]],
    concepts: Iterable[str],
    *,
    period_end: str | None,
    instant: bool,
) -> Mapping[str, Any] | None:
    concept_set = set(concepts)
    candidates = [fact for fact in facts if fact.get("concept") in concept_set and bool(fact.get("is_instant")) == instant]
    if period_end:
        candidates = [fact for fact in candidates if str(fact.get("end_date") or "")[:10] == period_end[:10]]
    if not candidates:
        return None
    candidates.sort(
        key=lambda fact: (
            bool(fact.get("has_dimensions")),
            -abs(_safe_float(fact.get("value")) or 0.0),
            str(fact.get("context_ref") or ""),
        )
    )
    return candidates[0]


def extract_esef_ixbrl_snapshot(package_bytes: bytes, *, period_end: str | None = None) -> dict[str, Any]:
    """Extract a conservative standard-field snapshot from an ESEF report package.

    This is intentionally narrow: it only accepts standard IFRS concepts from
    inline XBRL facts and prefers non-dimensional facts matching the filing
    period end.  Extension-taxonomy mapping belongs in a later, audited layer.
    """
    with zipfile.ZipFile(io.BytesIO(package_bytes)) as archive:
        report_names = [
            name
            for name in archive.namelist()
            if "/reports/" in name and name.lower().endswith((".xhtml", ".html", ".htm"))
        ]
        if not report_names:
            return {"status": "report_not_found", "fields": {}, "field_sources": {}}
        report_name = report_names[0]
        root = ET.fromstring(archive.read(report_name))

    contexts = _context_table(root)
    facts = _fact_candidates(root, contexts)
    fields: dict[str, float] = {}
    sources: dict[str, dict[str, Any]] = {}
    for field, tags in ESEF_FIELD_TAGS.items():
        fact = _select_fact(facts, tags, period_end=period_end, instant=field in INSTANT_FIELDS)
        if fact is None:
            continue
        fields[field] = float(fact["value"])
        sources[field] = {
            "concept": fact.get("concept"),
            "context_ref": fact.get("context_ref"),
            "start_date": fact.get("start_date"),
            "end_date": fact.get("end_date"),
            "has_dimensions": fact.get("has_dimensions"),
        }

    qmj_fields = {"gross_profit", "total_assets", "net_income", "operating_cashflow"}
    return {
        "status": "extracted",
        "report_name": report_name,
        "fact_count": len(facts),
        "fields": fields,
        "field_sources": sources,
        "has_qmj_minimum": len(qmj_fields.intersection(fields)) >= 2,
        "missing_qmj_fields": sorted(qmj_fields.difference(fields)),
    }


def probe_esef_extraction(
    filing_probe_payload: Mapping[str, Any],
    *,
    limit: int = 5,
    sleep_seconds: float = 0.2,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    qmj_hits = 0
    candidates = [
        item
        for item in (filing_probe_payload.get("items") or [])
        if (item.get("filings_xbrl") or {}).get("latest_filing_status") == "filing_found"
        and item.get("refined_qmj_contract_family") != "sector_specific_quality"
    ][:limit]
    for item in candidates:
        latest = ((item.get("filings_xbrl") or {}).get("latest_filing") or {})
        package_url = latest.get("package_url")
        if not package_url:
            extracted = {"status": "missing_package_url"}
        else:
            try:
                response = requests.get(package_url, timeout=30.0, headers={"User-Agent": USER_AGENT})
                if response.status_code == 200:
                    extracted = extract_esef_ixbrl_snapshot(response.content, period_end=latest.get("period_end"))
                else:
                    extracted = {"status": f"http_{response.status_code}"}
            except (requests.RequestException, zipfile.BadZipFile, ET.ParseError) as exc:
                extracted = {"status": "extract_error", "error": str(exc)}
        status_counts[str(extracted.get("status"))] += 1
        if extracted.get("has_qmj_minimum"):
            qmj_hits += 1
        rows.append(
            {
                "ticker": item.get("ticker"),
                "openfigi_name": (item.get("openfigi") or {}).get("name"),
                "period_end": latest.get("period_end"),
                "package_url": package_url,
                "extract": extracted,
            }
        )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return {
        "generated_at": _now_iso(),
        "schema_version": 1,
        "purpose": "read-only ESEF package extraction probe for standard QMJ fields",
        "probe_limit": limit,
        "probed": len(rows),
        "qmj_minimum_hits": qmj_hits,
        "status_counts": dict(status_counts.most_common()),
        "items": rows,
    }


def write_esef_extraction_probe(
    filing_probe_payload: Mapping[str, Any],
    *,
    output_path: str | Path | None = None,
    limit: int = 5,
    sleep_seconds: float = 0.2,
) -> dict[str, Any]:
    payload = probe_esef_extraction(filing_probe_payload, limit=limit, sleep_seconds=sleep_seconds)
    target = output_path or getattr(
        config,
        "NON_US_FUNDAMENTAL_ESEF_EXTRACT_PROBE_PATH",
        "feature_cache/non_us_fundamental_esef_extract_probe.json",
    )
    atomic_write_json(target, payload, indent=2)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-path", default="orchestrator_state.json")
    parser.add_argument("--audit-output", default=getattr(config, "NON_US_FUNDAMENTAL_COVERAGE_AUDIT_PATH", "feature_cache/non_us_fundamental_coverage_audit.json"))
    parser.add_argument("--probe-output", default=getattr(config, "NON_US_FUNDAMENTAL_FILING_PROBE_PATH", "feature_cache/non_us_fundamental_filing_probe.json"))
    parser.add_argument("--extract-output", default=getattr(config, "NON_US_FUNDAMENTAL_ESEF_EXTRACT_PROBE_PATH", "feature_cache/non_us_fundamental_esef_extract_probe.json"))
    parser.add_argument("--probe-esef", action="store_true", help="Probe OpenFIGI plus filings.xbrl.org for priority ESEF-route names")
    parser.add_argument("--extract-esef", action="store_true", help="Download matched ESEF packages and test standard IFRS field extraction")
    parser.add_argument("--probe-limit", type=int, default=12)
    parser.add_argument("--extract-limit", type=int, default=5)
    parser.add_argument("--probe-suffixes", default=".L", help="Comma-separated suffixes to probe, e.g. .L,.PA,.DE")
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args(argv)

    candidates = load_cached_candidates(args.state_path)
    audit = write_coverage_audit(candidates, output_path=args.audit_output)
    print(
        json.dumps(
            {
                "audit_output": args.audit_output,
                "summary": audit.get("summary"),
                "route_counts": audit.get("route_counts"),
            },
            indent=2,
        )
    )
    probe = None
    if args.probe_esef or args.extract_esef:
        suffixes = {value.strip().upper() for value in args.probe_suffixes.split(",") if value.strip()}
        probe = write_public_filing_probe(
            audit,
            output_path=args.probe_output,
            limit=args.probe_limit,
            suffixes=suffixes,
            sleep_seconds=args.sleep,
        )
        print(
            json.dumps(
                {
                    "probe_output": args.probe_output,
                    "probed": probe.get("probed"),
                    "status_counts": probe.get("status_counts"),
                },
                indent=2,
            )
        )
    if args.extract_esef:
        if probe is None:
            probe = json.loads(Path(args.probe_output).read_text(encoding="utf-8"))
        extract = write_esef_extraction_probe(
            probe,
            output_path=args.extract_output,
            limit=args.extract_limit,
            sleep_seconds=args.sleep,
        )
        print(
            json.dumps(
                {
                    "extract_output": args.extract_output,
                    "probed": extract.get("probed"),
                    "qmj_minimum_hits": extract.get("qmj_minimum_hits"),
                    "status_counts": extract.get("status_counts"),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
