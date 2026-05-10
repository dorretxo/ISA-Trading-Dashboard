"""Shared helpers for source-aware fundamental coverage evidence.

The action gates and replay/live diagnostics both need to answer the same
question: does this row have enough usable quality evidence, or is it merely
showing a thin balance-sheet fragment?  Keeping that logic in one module
prevents the gate and the report from drifting apart.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from engine.fscore_utils import is_f_score_actionable


YFINANCE_PIT_SOURCES = {"yfinance_info", "yfinance_quarterly", "yfinance"}


def _finite(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _get(row_or_candidate: Any, field: str) -> Any:
    if isinstance(row_or_candidate, Mapping):
        return row_or_candidate.get(field)
    return getattr(row_or_candidate, field, None)


def is_fmp_statement_candidate(ticker: str) -> bool:
    """Return True for plain US/FMP-style tickers."""
    symbol = str(ticker or "").upper().strip()
    return bool(symbol) and "." not in symbol


def is_non_us(ticker: str) -> bool:
    """Return True for dotted global tickers that generally use non-FMP data."""
    symbol = str(ticker or "").upper().strip()
    return "." in symbol


def qmj_usable_components(row_or_candidate: Any, *, config_module=None) -> int:
    """Count non-overlapping QMJ-lite dimensions with usable evidence."""
    if not row_or_candidate:
        return 0
    count = 0
    if any(
        _finite(_get(row_or_candidate, field)) is not None
        for field in ("gpa_score", "gpa", "gross_profitability")
    ):
        count += 1
    if is_f_score_actionable(
        _get(row_or_candidate, "f_score"),
        _get(row_or_candidate, "f_score_coverage"),
        config_module=config_module,
    ):
        count += 1
    if any(
        _finite(_get(row_or_candidate, field)) is not None
        for field in ("earnings_stability", "leverage_factor_score")
    ):
        count += 1
    return count


def classify_evidence(
    *,
    ticker: str,
    pit_source: str | None,
    qmj_components: int | None,
) -> str:
    """Classify statement evidence into operational buckets.

    ``yfinance_info`` is intentionally treated like yfinance quarterly data:
    in the latest audit it often supplied only balance-sheet fragments, which
    is the precise case that must not unlock STRONG BUY.
    """
    symbol = str(ticker or "").upper().strip()
    source = str(pit_source or "").strip().lower()
    components = int(qmj_components or 0)

    fmp_like = source == "fmp" or (not source and is_fmp_statement_candidate(symbol))
    yf_like = source in YFINANCE_PIT_SOURCES or (not source and is_non_us(symbol))

    if fmp_like:
        return "fmp_full" if components >= 3 else "fmp_partial"
    if yf_like:
        if components <= 1:
            return "yfinance_balance_only"
        return "yfinance_partial"
    if components <= 0:
        return "no_data"
    return "unknown_partial"
