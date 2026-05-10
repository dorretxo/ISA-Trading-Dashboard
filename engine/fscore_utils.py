"""Shared Piotroski F-score coverage semantics.

The F-score itself can be genuinely computed at zero, while its coverage can
also be unknown because the upstream computation never ran.  Keep those states
separate so gates and diagnostics do not silently treat unknown as computed.
"""

from __future__ import annotations

import math
from typing import Any


def finite_float(value: Any) -> float | None:
    """Return a finite float, or None for missing / non-numeric values."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def f_score_min_coverage(config_module=None) -> float:
    """Configured minimum coverage for an actionable F-score."""
    cfg = config_module
    if cfg is None:
        import config as cfg  # type: ignore[no-redef]
    return float(getattr(cfg, "F_SCORE_GATE_MIN_COVERAGE", 6.0 / 9.0))


def normalize_f_score_coverage(value: Any) -> float | None:
    """Normalize F-score coverage while preserving unknown as None."""
    cov = finite_float(value)
    if cov is None:
        return None
    return max(0.0, min(1.0, cov))


def is_f_score_actionable(
    f_score: Any,
    f_score_coverage: Any,
    *,
    config_module=None,
) -> bool:
    """True when the F-score and its coverage are both usable for gates."""
    return (
        finite_float(f_score) is not None
        and (normalize_f_score_coverage(f_score_coverage) or 0.0) >= f_score_min_coverage(config_module)
    )
