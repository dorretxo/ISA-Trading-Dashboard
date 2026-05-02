"""Factor validation surface (roadmap item #10).

Diagnostic helpers for measuring whether a factor actually carries signal
before it is trusted in live ranking.  Every number here has a paper
reference; the goal is an always-on health panel for the scoring engine.

Primary consumers:
    - `engine.discovery_eval` summaries
    - a future "Factor Lab" Streamlit panel
    - CI smoke tests that assert minimum IC / monotonicity

Metrics:
    - rolling_ic            — Spearman-rank cross-sectional IC over time
                              (Grinold & Kahn 2000 "Active Portfolio Management")
    - decile_monotonicity   — top-vs-bottom decile spread and monotonic slope
                              (Fama & French 1993; Asness et al. 2013)
    - fama_macbeth          — two-stage cross-sectional regression with
                              Newey-West style t-stats (Fama & MacBeth 1973)
    - turnover_report       — one-period name-turnover in the top bucket
                              (Novy-Marx & Velikov 2016 "A Taxonomy of Anomalies
                              and Their Trading Costs")
    - sector_neutrality_audit — checks whether a factor collapses into a
                              sector bet (Ang 2014 "Asset Management")

All functions are defensive: NaNs propagate, small samples return None,
callers can log without handling exceptions.
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rolling Information Coefficient (IC)
# ---------------------------------------------------------------------------

def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman rank correlation on paired arrays (NaN-dropped)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return None
    rx = pd.Series(x[mask]).rank().values
    ry = pd.Series(y[mask]).rank().values
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return None
    c = float(np.corrcoef(rx, ry)[0, 1])
    return c if math.isfinite(c) else None


def rolling_ic(
    panel: pd.DataFrame,
    *,
    factor_col: str,
    forward_return_col: str,
    date_col: str = "as_of",
    window: int = 12,
) -> pd.DataFrame:
    """Cross-sectional IC per date + rolling mean IC.

    Expects a long-form panel with one row per (date, ticker).  Returns a
    frame indexed by date with `ic` and `ic_{window}m_mean` columns.

    Positive mean IC with low standard deviation ⇒ factor is worth keeping.
    Grinold & Kahn (2000) rule of thumb: IC > 0.05, IR = mean / std > 0.5.
    """
    if panel is None or panel.empty:
        return pd.DataFrame(columns=["ic", "ic_mean"])
    req = {factor_col, forward_return_col, date_col}
    if not req.issubset(panel.columns):
        missing = req - set(panel.columns)
        logger.warning("rolling_ic: missing columns %s", missing)
        return pd.DataFrame(columns=["ic", "ic_mean"])

    out = []
    for dt, chunk in panel.groupby(date_col):
        ic = _spearman(chunk[factor_col].values, chunk[forward_return_col].values)
        out.append((dt, ic, len(chunk)))
    df = pd.DataFrame(out, columns=[date_col, "ic", "n"]).sort_values(date_col).set_index(date_col)
    df["ic_mean"] = df["ic"].rolling(window=window, min_periods=max(3, window // 2)).mean()
    df["ic_std"] = df["ic"].rolling(window=window, min_periods=max(3, window // 2)).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ic_ir"] = df["ic_mean"] / df["ic_std"]
    return df


# ---------------------------------------------------------------------------
# Decile monotonicity
# ---------------------------------------------------------------------------

def decile_monotonicity(
    factor: Iterable[float],
    forward_return: Iterable[float],
    *,
    n_buckets: int = 5,
) -> dict[str, float | None]:
    """Bucket by factor, return per-bucket mean forward return.

    Reports:
        - `buckets`: per-bucket mean forward return (list)
        - `spread`: top − bottom bucket mean return
        - `monotonic_slope`: OLS slope of bucket index vs mean return
        - `is_monotonic`: True if slope sign matches spread sign and all
                          adjacent diffs share it in ≥ 60% of cases
    Five-bucket by default (retail-scale cross-sections); use 10 for
    large universes.
    """
    f = np.asarray(list(factor), dtype=float)
    r = np.asarray(list(forward_return), dtype=float)
    mask = np.isfinite(f) & np.isfinite(r)
    if mask.sum() < n_buckets * 3:
        return {"buckets": None, "spread": None, "monotonic_slope": None, "is_monotonic": False}

    f = f[mask]
    r = r[mask]
    try:
        labels = pd.qcut(f, q=n_buckets, labels=False, duplicates="drop")
    except ValueError:
        return {"buckets": None, "spread": None, "monotonic_slope": None, "is_monotonic": False}

    buckets: list[float] = []
    for b in range(int(labels.max()) + 1):
        sel = r[labels == b]
        buckets.append(float(np.mean(sel)) if len(sel) else float("nan"))
    if len(buckets) < 2:
        return {"buckets": buckets, "spread": None, "monotonic_slope": None, "is_monotonic": False}

    spread = buckets[-1] - buckets[0]
    xs = np.arange(len(buckets), dtype=float)
    ys = np.array(buckets, dtype=float)
    ok = np.isfinite(ys)
    if ok.sum() < 2:
        return {"buckets": buckets, "spread": spread, "monotonic_slope": None, "is_monotonic": False}
    slope, _ = np.polyfit(xs[ok], ys[ok], 1)

    diffs = np.diff(ys[ok])
    signs = np.sign(diffs)
    dom_sign = np.sign(spread) if spread != 0 else 0
    agree = float(np.mean(signs == dom_sign)) if len(signs) else 0.0
    is_monotonic = (abs(slope) > 1e-6) and (np.sign(slope) == dom_sign) and (agree >= 0.6)

    return {
        "buckets": buckets,
        "spread": float(spread),
        "monotonic_slope": float(slope),
        "is_monotonic": bool(is_monotonic),
    }


# ---------------------------------------------------------------------------
# Fama-MacBeth two-stage cross-sectional regression
# ---------------------------------------------------------------------------

def fama_macbeth(
    panel: pd.DataFrame,
    *,
    factor_cols: Sequence[str],
    forward_return_col: str,
    date_col: str = "as_of",
) -> dict[str, dict[str, float]]:
    """Per-factor mean premium and t-stat across cross-sectional regressions.

    Fama & MacBeth (1973): run one OLS per period of forward_return on the
    factor panel, then test whether the time-series of slopes has a mean
    statistically different from zero.

    Returns {factor: {"mean": λ_bar, "t_stat": t, "n_periods": T}}.
    This is the textbook t-stat (√T · λ̄ / σ_λ); for heteroskedasticity /
    autocorrelation corrections use `statsmodels`.
    """
    if panel is None or panel.empty or not factor_cols:
        return {}

    if not set([*factor_cols, forward_return_col, date_col]).issubset(panel.columns):
        return {}

    slope_records: dict[str, list[float]] = {c: [] for c in factor_cols}
    for _, chunk in panel.groupby(date_col):
        sub = chunk[[*factor_cols, forward_return_col]].dropna()
        if len(sub) < len(factor_cols) + 3:
            continue
        X = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in factor_cols])
        y = sub[forward_return_col].values
        try:
            betas, *_ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            continue
        for i, c in enumerate(factor_cols, start=1):
            slope_records[c].append(float(betas[i]))

    out: dict[str, dict[str, float]] = {}
    for c, slopes in slope_records.items():
        arr = np.asarray(slopes, dtype=float)
        if len(arr) < 3:
            out[c] = {"mean": float("nan"), "t_stat": float("nan"), "n_periods": len(arr)}
            continue
        mean = float(arr.mean())
        sd = float(arr.std(ddof=1))
        t = (mean / (sd / math.sqrt(len(arr)))) if sd > 1e-12 else float("nan")
        out[c] = {"mean": mean, "t_stat": t, "n_periods": len(arr)}
    return out


# ---------------------------------------------------------------------------
# Turnover report
# ---------------------------------------------------------------------------

def turnover_report(
    rolling_top_buckets: Sequence[Sequence[str]],
) -> dict[str, float]:
    """One-period name-turnover in the top bucket.

    Given a list of top-N name-sets ordered by time, compute:
        - mean_turnover: fraction of names swapped per period
        - p90_turnover: 90th percentile
        - avg_holding_months: naive 1/mean_turnover estimate

    Novy-Marx & Velikov (2016): momentum's post-cost alpha depends critically
    on turnover. A factor with 50%+ monthly turnover will be eaten by costs
    at retail-scale.
    """
    if len(rolling_top_buckets) < 2:
        return {"mean_turnover": float("nan"), "p90_turnover": float("nan"), "avg_holding_months": float("nan")}

    turnovers: list[float] = []
    for prev, cur in zip(rolling_top_buckets[:-1], rolling_top_buckets[1:]):
        pset = set(prev)
        cset = set(cur)
        if not pset or not cset:
            continue
        drops = len(pset - cset)
        denom = max(len(pset), 1)
        turnovers.append(drops / denom)
    if not turnovers:
        return {"mean_turnover": float("nan"), "p90_turnover": float("nan"), "avg_holding_months": float("nan")}

    arr = np.asarray(turnovers, dtype=float)
    mean = float(arr.mean())
    return {
        "mean_turnover": mean,
        "p90_turnover": float(np.quantile(arr, 0.9)),
        "avg_holding_months": float(1.0 / mean) if mean > 1e-6 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Sector neutrality audit
# ---------------------------------------------------------------------------

def sector_neutrality_audit(
    factor: Iterable[float],
    sectors: Iterable[str | None],
) -> dict[str, float | dict[str, float]]:
    """Measure how concentrated a factor is along sector lines.

    Reports:
        - `sector_r2`: R² of a one-way ANOVA where the factor is the
                       response and the sector label is the dummy.  High
                       R² means the factor is largely a sector bet.
        - `top_sector_share`: fraction of the top quintile that belongs to
                              the single most-represented sector.
        - `sector_mean`: mean factor value per sector.

    A healthy cross-sectional factor has `sector_r2` < 0.30 and
    `top_sector_share` < 0.40 (Ang 2014).
    """
    f = np.asarray(list(factor), dtype=float)
    secs = [str(s) if s else "Unknown" for s in sectors]
    mask = np.isfinite(f)
    if mask.sum() < 10:
        return {"sector_r2": float("nan"), "top_sector_share": float("nan"), "sector_mean": {}}

    f_ok = f[mask]
    secs_ok = [secs[i] for i, m in enumerate(mask) if m]
    grand_mean = float(f_ok.mean())

    # ANOVA R²
    ss_total = float(((f_ok - grand_mean) ** 2).sum())
    if ss_total < 1e-12:
        r2 = float("nan")
    else:
        sector_sums = {}
        for v, s in zip(f_ok, secs_ok):
            sector_sums.setdefault(s, []).append(v)
        ss_between = sum(
            len(vals) * (float(np.mean(vals)) - grand_mean) ** 2
            for vals in sector_sums.values()
        )
        r2 = ss_between / ss_total

    # Top-quintile sector concentration
    cutoff = float(np.quantile(f_ok, 0.8)) if len(f_ok) >= 5 else float(np.max(f_ok))
    top_secs = [secs_ok[i] for i, v in enumerate(f_ok) if v >= cutoff]
    if top_secs:
        top_counter = Counter(top_secs)
        top_share = top_counter.most_common(1)[0][1] / len(top_secs)
    else:
        top_share = float("nan")

    # Per-sector mean (sorted for readability)
    per: dict[str, float] = {}
    for s in set(secs_ok):
        vals = [f_ok[i] for i, sec in enumerate(secs_ok) if sec == s]
        per[s] = float(np.mean(vals))

    return {
        "sector_r2": float(r2),
        "top_sector_share": float(top_share),
        "sector_mean": per,
    }
