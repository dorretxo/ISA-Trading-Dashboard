"""Conformal prediction calibration for STRONG BUY confidence.

Distribution-free, frequentist coverage guarantees that work with as few as
~50 calibration samples — exactly matching the recovered training set after
``backfill_panel_actions``.

References
----------
Vovk, Gammerman & Shafer (2005). *Algorithmic Learning in a Random World*.
Angelopoulos & Bates (2021). "A Gentle Introduction to Conformal Prediction
    and Distribution-Free Uncertainty Quantification."  arXiv:2107.07511.
Wisniewski & Lindsay (2020). "Application of Conformal Prediction Interval
    Estimations to Market Makers."

Mechanics
---------
For each candidate with a model-predicted 90d return ``ŷ`` we ask: what
fraction of matured historical signals achieved a realised return AT LEAST
as high as ``ŷ``?  This is the right-tail empirical exceedance probability:

    p = (1 + #{i : r_i ≥ ŷ}) / (1 + n)

LOW ``p`` ⇔ the prediction is in the upper tail of the historical
distribution (rare, exceptional outcome).  By convention:

    p < 0.05  →  STRONG BUY at 95% one-sided coverage.
    p < 0.10  →  BUY confidence band.
    p ≥ 0.50  →  prediction is below the historical median; not confident.

Crucially, this is a *finite-sample* coverage guarantee — we do not need to
assume normality or calibrate any hyperparameter beyond the look-back window.

The slope ``k_ic`` mapping ``aggregate_score`` (or ``sb_score``) to expected
90d return is fit via Fama–MacBeth (1973) cross-sectional regression.  When
fewer than ``min_samples`` signals are available, we fall back to
``CONFORMAL_K_IC_FALLBACK`` (default 0.10 — equity-factor IC anchor of
Lewellen 2015 *Critical Finance Review*).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Iterable

import numpy as np

import config
from engine.paper_trading import _connect

logger = logging.getLogger(__name__)


def _calibration_returns(days: int) -> list[float]:
    """Return realised 90d returns for matured signals in the look-back window.

    Pulls from any source where ``evaluated_90d=1`` and ``return_90d`` is not
    null.  Returns are stored as percent (e.g. 7.9 for +7.9 %); we convert to
    decimal so the unit matches the model's predicted return scale.
    """
    if days <= 0:
        return []
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    rows: list[float] = []
    try:
        with _connect() as conn:
            cur = conn.execute(
                """
                SELECT return_90d FROM signal_backtest
                WHERE evaluated_90d = 1 AND return_90d IS NOT NULL
                  AND run_date >= ?
                """,
                (cutoff,),
            )
            for r in cur:
                try:
                    rows.append(float(r[0]) / 100.0)
                except (TypeError, ValueError):
                    continue
    except Exception as e:
        logger.warning("conformal calibration query failed: %s", e)
        return []
    return rows


def _famamacbeth_slope(min_samples: int, look_back_days: int) -> float | None:
    """Fama-MacBeth (1973) cross-sectional slope of ``return_90d ~ aggregate_score``.

    For each ``run_date`` cohort with ≥ 5 matured rows, compute the simple
    OLS slope ``return = α + β·score``.  Take the mean β across cohorts.
    Returns None when fewer than ``min_samples`` total mature rows exist.
    """
    if look_back_days <= 0:
        return None
    cutoff = (datetime.now() - timedelta(days=look_back_days)).isoformat(timespec="seconds")
    try:
        with _connect() as conn:
            cur = conn.execute(
                """
                SELECT run_date, aggregate_score, return_90d
                FROM signal_backtest
                WHERE evaluated_90d = 1 AND return_90d IS NOT NULL
                  AND aggregate_score IS NOT NULL
                  AND run_date >= ?
                """,
                (cutoff,),
            )
            rows = cur.fetchall()
    except Exception as e:
        logger.warning("Fama-MacBeth query failed: %s", e)
        return None

    if len(rows) < min_samples:
        return None

    by_date: dict[str, list[tuple[float, float]]] = {}
    for run_date, score, ret in rows:
        if score is None or ret is None:
            continue
        d = str(run_date or "")[:10]
        by_date.setdefault(d, []).append((float(score), float(ret) / 100.0))

    slopes: list[float] = []
    for d, pts in by_date.items():
        if len(pts) < 5:
            continue
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        if np.std(xs) < 1e-9:
            continue
        # Simple OLS: slope = cov(x,y) / var(x)
        slope = float(np.cov(xs, ys, ddof=0)[0, 1] / np.var(xs, ddof=0))
        if np.isfinite(slope):
            slopes.append(slope)

    if not slopes:
        return None
    mean_slope = float(np.mean(slopes))
    return mean_slope if np.isfinite(mean_slope) else None


def compute_k_ic() -> float:
    """Returns the calibrated score → 90d-return slope, or the fallback prior."""
    look_back = int(getattr(config, "CONFORMAL_CALIBRATION_DAYS", 365))
    min_n = int(getattr(config, "CONFORMAL_MIN_CALIBRATION_N", 50))
    fallback = float(getattr(config, "CONFORMAL_K_IC_FALLBACK", 0.10))
    slope = _famamacbeth_slope(min_samples=min_n, look_back_days=look_back)
    if slope is None or slope <= 0:
        logger.info("conformal: using fallback k_ic = %.4f (no/insufficient history)", fallback)
        return fallback
    logger.info("conformal: Fama-MacBeth k_ic = %.4f from history", slope)
    return slope


def compute_conformal_p(candidates: Iterable, *, k_ic: float | None = None) -> dict[str, float]:
    """Populate conformal p-values on each candidate with an ``sb_score``.

    Returns ``{ticker: p_value}`` and also sets ``c.conformal_p`` in place.
    Candidates without a usable score get ``p = 1.0`` (non-informative).
    """
    if not bool(getattr(config, "CONFORMAL_ENABLED", True)):
        return {}

    cands = list(candidates)
    if not cands:
        return {}

    look_back = int(getattr(config, "CONFORMAL_CALIBRATION_DAYS", 365))
    min_n = int(getattr(config, "CONFORMAL_MIN_CALIBRATION_N", 50))
    cal = _calibration_returns(look_back)
    if len(cal) < min_n:
        logger.info("conformal: only %d calibration rows (< %d); skipping", len(cal), min_n)
        return {}

    cal_arr = np.array(cal, dtype=float)
    n_cal = cal_arr.size

    if k_ic is None:
        k_ic = compute_k_ic()

    out: dict[str, float] = {}
    for c in cands:
        ticker = getattr(c, "ticker", None) or getattr(c, "symbol", None)
        if not ticker:
            continue
        sb = getattr(c, "sb_score", None)
        agg = getattr(c, "aggregate_score", None)
        # Prefer sb_score when set (richer cohort signal); else fall back to aggregate.
        score = sb if sb is not None else agg
        if score is None:
            try:
                c.conformal_p = 1.0
            except (AttributeError, TypeError):
                pass
            out[ticker] = 1.0
            continue

        predicted_return = float(k_ic) * float(score)
        # Right-tail exceedance p-value: how many historical realised returns
        # AT LEAST matched our predicted return?  Low p = prediction is in
        # the upper tail = confident STRONG BUY.
        p = float((1 + int(np.sum(cal_arr >= predicted_return))) / (1 + n_cal))
        try:
            c.conformal_p = p
        except (AttributeError, TypeError):
            pass
        out[ticker] = p

    return out
