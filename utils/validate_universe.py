"""Monthly universe validation script.

Checks every ticker in the structured universe for:
  1. yfinance resolvability (does it return any data?)
  2. Price history freshness (last trade within 30 days?)
  3. All-NaN price data (delisted / broken suffix)
  4. Duplicate detection

Produces a report of:
  - DEAD: ticker returns no data at all
  - STALE: last price > 30 days old
  - DUPLICATE: appears more than once (case-insensitive)

Usage:
    python -m utils.validate_universe [--fix]

    --fix  writes a cleaned version with dead tickers marked inactive
           (not yet implemented — for now, just prints the report)
"""

from __future__ import annotations

import datetime as dt
import sys
import time

import pandas as pd


def validate(verbose: bool = True) -> dict:
    """Run full validation and return results dict."""
    from utils.global_universe import get_full_universe, get_universe_stats

    universe = get_full_universe()
    stats = get_universe_stats()

    if verbose:
        print(f"Universe validation — {stats['total']} tickers "
              f"(T1: {stats['tier1']}, T2: {stats['tier2']})")
        print(f"Last refreshed: {stats['last_refreshed']}")
        print("-" * 60)

    # Check for duplicates first (no network needed)
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for entry in universe:
        key = entry.ticker.upper()
        if key in seen:
            duplicates.append(f"{entry.ticker} (dup of {seen[key]})")
        else:
            seen[key] = entry.ticker

    if duplicates and verbose:
        print(f"\n⚠ DUPLICATES ({len(duplicates)}):")
        for d in duplicates:
            print(f"  {d}")

    # Batch download price data (faster than per-ticker)
    import yfinance as yf

    tickers_list = [e.ticker for e in universe]
    batch_size = 100
    dead: list[str] = []
    stale: list[str] = []
    ok: list[str] = []
    cutoff = dt.datetime.now() - dt.timedelta(days=30)

    for i in range(0, len(tickers_list), batch_size):
        batch = tickers_list[i:i + batch_size]
        if verbose:
            print(f"\rValidating {i+1}-{min(i+batch_size, len(tickers_list))} "
                  f"of {len(tickers_list)}...", end="", flush=True)

        try:
            data = yf.download(
                batch,
                period="5d",
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception as e:
            if verbose:
                print(f"\n  ✗ Batch download failed: {e}")
            dead.extend(batch)
            continue

        for ticker in batch:
            try:
                if len(batch) == 1:
                    closes = data["Close"]
                else:
                    closes = data[ticker]["Close"] if ticker in data.columns.get_level_values(0) else pd.Series(dtype=float)

                if closes.dropna().empty:
                    dead.append(ticker)
                else:
                    last_date = closes.dropna().index[-1]
                    if hasattr(last_date, 'tz') and last_date.tz is not None:
                        last_date = last_date.tz_localize(None)
                    if last_date < pd.Timestamp(cutoff):
                        stale.append(ticker)
                    else:
                        ok.append(ticker)
            except Exception:
                dead.append(ticker)

        # Brief pause to avoid rate-limiting
        time.sleep(0.5)

    if verbose:
        print(f"\n\n{'='*60}")
        print(f"RESULTS:")
        print(f"  ✓ OK:    {len(ok)}")
        print(f"  ⚠ STALE: {len(stale)} (no trade in 30 days)")
        print(f"  ✗ DEAD:  {len(dead)} (no data returned)")
        print(f"  ⊘ DUPS:  {len(duplicates)}")

        if stale:
            print(f"\nSTALE tickers (review needed):")
            for t in sorted(stale):
                print(f"  {t}")

        if dead:
            print(f"\nDEAD tickers (consider removing):")
            for t in sorted(dead):
                print(f"  {t}")

        print(f"\n{'='*60}")

    return {
        "total": len(tickers_list),
        "ok": ok,
        "stale": stale,
        "dead": dead,
        "duplicates": duplicates,
    }


if __name__ == "__main__":
    validate(verbose=True)
