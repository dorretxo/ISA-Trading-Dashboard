"""Safe numeric utilities — single source of truth for NaN/None handling.

Every numeric value that flows from yfinance, FMP, or cached state should
pass through safe_float() before arithmetic or display.  Every currency
display should use format_currency() or format_pct() instead of raw f-strings.
"""

import math


def safe_float(val, default: float = 0.0) -> float:
    """Convert val to a finite float, returning default if None/NaN/inf/non-numeric.

    This is the ONE rule for "valid number" across the entire codebase.
    """
    if val is None:
        return default
    try:
        f = float(val)
        if math.isfinite(f):
            return f
        return default
    except (TypeError, ValueError):
        return default


def is_valid_number(val) -> bool:
    """Return True if val is a finite number (not None, NaN, inf)."""
    if val is None:
        return False
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def format_currency(val, currency: str = "GBP", decimals: int = 2, default: str = "N/A") -> str:
    """Format a numeric value as a currency string, safe against NaN/None.

    Args:
        val: The numeric value (can be None, NaN, inf)
        currency: Currency code (GBP, GBX, USD, EUR, etc.)
        decimals: Decimal places (0 for whole numbers, 2 for prices)
        default: String to return for invalid values
    """
    if not is_valid_number(val):
        return default

    val = float(val)
    symbol = "£" if currency in ("GBP", "GBX") else "$" if currency == "USD" else "€"

    if currency == "GBX":
        return f"{val:,.0f}p"

    if decimals == 0:
        return f"{symbol}{val:,.0f}"
    return f"{symbol}{val:,.{decimals}f}"


def format_pct(val, decimals: int = 1, plus_sign: bool = True, default: str = "N/A") -> str:
    """Format a numeric value as a percentage string, safe against NaN/None.

    Args:
        val: The numeric value (can be None, NaN, inf) — already a percentage (e.g. 5.2 not 0.052)
        decimals: Decimal places
        plus_sign: Whether to prepend '+' for positive values
        default: String to return for invalid values
    """
    if not is_valid_number(val):
        return default

    val = float(val)
    sign = "+" if plus_sign and val >= 0 else ""
    return f"{sign}{val:.{decimals}f}%"


def format_score(val, decimals: int = 3, default: str = "N/A") -> str:
    """Format a score value, safe against NaN/None."""
    if not is_valid_number(val):
        return default
    val = float(val)
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.{decimals}f}"
