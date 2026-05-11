"""Canonical scalar transforms for factor inputs.

These helpers keep live scoring, replay, and parity canonicalization on the
same numeric contract.  They intentionally do not apply source-confidence
dampeners; source quality belongs in confidence/gating layers rather than in
the raw factor score itself.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _finite(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _clip(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def compute_f_score_score(f_score: Any) -> float | None:
    value = _finite(f_score)
    if value is None:
        return None
    return _clip((value - 4.5) / 3.0)


def compute_gpa_score(gpa: Any) -> float | None:
    value = _finite(gpa)
    if value is None:
        return None
    return _clip((value - 0.30) / 0.20)


def compute_ev_ebit_score(ebit_yield: Any) -> float | None:
    value = _finite(ebit_yield)
    if value is None:
        return None
    return _clip((value - 0.10) / 0.10)


def compute_fcf_yield_score(fcf_yield: Any) -> float | None:
    value = _finite(fcf_yield)
    if value is None:
        return None
    return _clip((value - 0.03) / 0.08)
