"""Discovery diagnostics — bucketed hit-rate / trap-rate analyzer.

Purpose
-------
Surface *where* discovery is producing edge and *where* it is producing
traps, by bucketing evaluated signals along the dimensions the ranker
cares about:

    * price-vs-200DMA stretch (late-cycle risk)
    * fundamental_score tier
    * Piotroski F-score tier (when persisted)
    * GPA tier (when persisted)
    * sector
    * market regime

For each bucket we emit:

    n                : sample size
    hit_rate         : fraction with fwd return > +5% (5d/10d/30d/60d/90d horizon)
    trap_rate        : fraction with fwd return < −5%
    avg_return       : mean fwd return
    median_return    : robust central tendency
    beat_spy_rate    : fraction with 90d return > SPY return (when available)

Usage
-----
    python -m engine.discovery_diagnostics --horizon 10
    python -m engine.discovery_diagnostics --horizon 30
    python -m engine.discovery_diagnostics --horizon 90 --min-n 20 --source discovery

Statistical honesty
-------------------
Buckets with fewer than ``min-n`` observations are dropped — the
script will tell you when that's every bucket (i.e. you don't yet have
enough evaluated data to draw conclusions).  **Thresholds that inform
pipeline gates should not be fit from buckets below n=30.**
"""
from __future__ import annotations

import argparse
import logging
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Reuse the same connection helper the backtest store uses
from engine.paper_trading import _connect
from engine.discovery_backtest import init_backtest_db


# ---------------------------------------------------------------------------
# Bucket definitions — tweakable from CLI later if needed
# ---------------------------------------------------------------------------

# Stretch tiers: price / sma_200 − 1.  Negative = below 200DMA (downtrend).
STRETCH_BUCKETS = [
    ("below_200dma",   -math.inf, 0.0),
    ("near_trend",     0.0,       0.15),
    ("elevated",       0.15,      0.35),
    ("extended",       0.35,      0.60),
    ("parabolic",      0.60,      math.inf),
]

# Piotroski tiers — Piotroski 2000 cuts: ≤3 bearish, 4–5 mid, ≥6 bullish
FSCORE_BUCKETS = [
    ("bearish_0_3", -0.5, 3.5),
    ("mid_4_5",     3.5,  5.5),
    ("bullish_6_9", 5.5,  9.5),
]

# GPA tiers — Novy-Marx 2013 uses sector-relative, but absolute tiers
# are a defensible first cut for this diagnostic.
GPA_BUCKETS = [
    ("weak_lt_015",    -math.inf, 0.15),
    ("mid_015_033",    0.15,      0.33),
    ("strong_ge_033",  0.33,      math.inf),
]

FUNDAMENTAL_BUCKETS = [
    ("negative", -math.inf, -0.1),
    ("neutral",  -0.1,      0.2),
    ("positive", 0.2,       math.inf),
]


@dataclass
class BucketStats:
    bucket: str
    n: int
    hit_rate: float | None
    trap_rate: float | None
    avg_return: float | None
    median_return: float | None
    beat_spy_rate: float | None

    def as_row(self) -> str:
        def fmt_pct(v):
            return f"{v*100:6.1f}%" if v is not None else "    —  "

        def fmt_num(v):
            return f"{v:+7.2f}%" if v is not None else "    —  "

        return (
            f"{self.bucket:<20} n={self.n:>4} "
            f"hit={fmt_pct(self.hit_rate)} trap={fmt_pct(self.trap_rate)} "
            f"avg={fmt_num(self.avg_return)} med={fmt_num(self.median_return)} "
            f"vs_spy={fmt_pct(self.beat_spy_rate)}"
        )


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

def _load_evaluated_signals(
    horizon: int,
    source: str | None,
    lookback_days: int | None,
) -> list[sqlite3.Row]:
    """Pull evaluated signals from paper_trading.db.

    Only rows with non-null return at the requested horizon are returned.
    """
    init_backtest_db()
    ret_col = f"return_{horizon}d"
    eval_col = f"evaluated_{horizon}d"

    where = [f"{ret_col} IS NOT NULL", f"{eval_col} = 1"]
    params: list = []
    if source:
        where.append("source = ?")
        params.append(source)
    if lookback_days:
        where.append(
            "julianday('now') - julianday(run_date) <= ?"
        )
        params.append(lookback_days)

    sql = (
        "SELECT ticker, sector, regime, source, run_date, signal_price, "
        "technical_score, fundamental_score, sentiment_score, forecast_score, "
        "momentum_score, final_rank, "
        "gross_profitability, quality_score_fundamental, "
        "f_score, f_score_coverage, ev_ebit, ev_ebit_score, gpa, gpa_score, "
        "sma_200, price_vs_sma200_stretch, analyst_upside, "
        "gate_v2_status, trap_safeguard_triggered, "
        f"{ret_col} AS fwd_return, spy_return_90d "
        "FROM signal_backtest WHERE " + " AND ".join(where)
    )

    with _connect() as conn:
        return conn.execute(sql, params).fetchall()


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

def _tier_for(value: float | None, buckets: list[tuple[str, float, float]]) -> str | None:
    """Return the bucket name, or None if value is missing/unbucketable."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    for name, lo, hi in buckets:
        if lo <= v < hi:
            return name
    return None


def _stats_for_group(rows: list[sqlite3.Row], hit_thresh: float = 5.0,
                     trap_thresh: float = -5.0) -> BucketStats | None:
    """Compute summary stats for a group of evaluated signals."""
    returns = []
    beat_spy_flags = []
    for r in rows:
        ret = r["fwd_return"]
        if ret is None:
            continue
        returns.append(float(ret))
        spy = r["spy_return_90d"]
        if spy is not None:
            beat_spy_flags.append(1 if float(ret) > float(spy) else 0)

    n = len(returns)
    if n == 0:
        return None

    returns_sorted = sorted(returns)
    mid = n // 2
    median = (returns_sorted[mid] if n % 2 == 1
              else (returns_sorted[mid - 1] + returns_sorted[mid]) / 2.0)

    hits = sum(1 for r in returns if r > hit_thresh)
    traps = sum(1 for r in returns if r < trap_thresh)

    return BucketStats(
        bucket="",  # caller fills in
        n=n,
        hit_rate=hits / n,
        trap_rate=traps / n,
        avg_return=sum(returns) / n,
        median_return=median,
        beat_spy_rate=(sum(beat_spy_flags) / len(beat_spy_flags)) if beat_spy_flags else None,
    )


def _bucket_by(rows: list[sqlite3.Row], field: str,
               buckets: list[tuple[str, float, float]] | None = None,
               min_n: int = 10) -> list[BucketStats]:
    """Group rows by ``field`` (or by pre-defined bucket tiers) and summarise."""
    groups: dict[str, list[sqlite3.Row]] = {}
    for r in rows:
        if buckets is not None:
            key = _tier_for(r[field], buckets)
        else:
            key = r[field]
        if key is None or key == "":
            key = "unknown"
        groups.setdefault(str(key), []).append(r)

    out: list[BucketStats] = []
    # Order: if tier list provided, preserve its order; else alpha
    if buckets is not None:
        ordering = [b[0] for b in buckets] + ["unknown"]
        for name in ordering:
            if name in groups:
                stats = _stats_for_group(groups[name])
                if stats and stats.n >= min_n:
                    stats.bucket = name
                    out.append(stats)
    else:
        for name in sorted(groups.keys()):
            stats = _stats_for_group(groups[name])
            if stats and stats.n >= min_n:
                stats.bucket = name
                out.append(stats)
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def run_report(horizon: int = 30, source: str = "discovery",
               min_n: int = 10, lookback_days: int | None = None) -> dict:
    """Print the bucketed hit-rate report and return structured results."""
    rows = _load_evaluated_signals(horizon, source, lookback_days)
    total = len(rows)

    print()
    print("=" * 88)
    print(f"Discovery diagnostics — horizon={horizon}d source={source or 'all'} "
          f"min_n={min_n} lookback={lookback_days or 'all'}")
    print(f"Evaluated signals found: {total}")
    print("=" * 88)

    if total < min_n:
        print()
        print(f"[WARN] Only {total} evaluated signals — below min_n={min_n}.")
        print("      Any bucketed statistics would be noise.  Let the evaluator")
        print("      accumulate more matured signals (daily orchestrator calls")
        print("      evaluate_matured_signals on every run).")
        print()
        return {"total": total, "buckets": {}}

    results: dict[str, list[BucketStats]] = {}

    dims = [
        ("stretch vs 200DMA", "price_vs_sma200_stretch", STRETCH_BUCKETS),
        ("Piotroski F-score", "f_score", FSCORE_BUCKETS),
        ("GPA (Novy-Marx)", "gpa", GPA_BUCKETS),
        ("fundamental_score", "fundamental_score", FUNDAMENTAL_BUCKETS),
        ("gate v2 shadow", "gate_v2_status", None),
        ("trap safeguard", "trap_safeguard_triggered", None),
        ("sector", "sector", None),
        ("regime", "regime", None),
    ]

    for label, field, buckets in dims:
        stats_list = _bucket_by(rows, field, buckets, min_n=min_n)
        if not stats_list:
            continue
        print()
        print(f"-- by {label} " + "-" * (82 - len(label) - 6))
        for s in stats_list:
            print(s.as_row())
        results[label] = stats_list

    print()
    print("Legend: hit = fwd_return > +5%   trap = fwd_return < -5%   "
          "vs_spy = beats SPY over 90d")
    print("=" * 88)
    print()

    return {"total": total, "buckets": results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Discovery bucketed hit-rate analyzer")
    p.add_argument("--horizon", type=int, default=30, choices=[5, 10, 30, 60, 90])
    p.add_argument("--source", type=str, default="discovery",
                   help="signal source filter; use 'all' to disable")
    p.add_argument("--min-n", type=int, default=10,
                   help="minimum sample size per bucket (buckets below this are dropped)")
    p.add_argument("--lookback-days", type=int, default=None,
                   help="only include signals with run_date within the last N days")
    args = p.parse_args()

    src = None if args.source == "all" else args.source
    run_report(
        horizon=args.horizon,
        source=src,
        min_n=args.min_n,
        lookback_days=args.lookback_days,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
