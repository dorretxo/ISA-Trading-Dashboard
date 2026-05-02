"""Performance attribution vs. a market benchmark.

Two components:

1. **Time-weighted return (TWR)** — Brinson, Hood & Beebower (1986).  The
   standard GIPS-compliant metric for comparing a portfolio's return to a
   benchmark when external cash flows are present.  We compute a simple
   single-period TWR from daily NAV-weighted returns (no cash-flow adjustment
   — the dashboard does not track deposits yet).

2. **Brinson-Fachler attribution** (Brinson & Fachler 1985).  Decomposes
   excess return vs. the benchmark into:
       * **Allocation**  = Σ (w_p,s − w_b,s) · (r_b,s − r_b)
       * **Selection**   = Σ  w_b,s       · (r_p,s − r_b,s)
       * **Interaction** = Σ (w_p,s − w_b,s) · (r_p,s − r_b,s)
   where s indexes GICS sectors, r_b is the total benchmark return, and
   the subscripts p/b denote portfolio/benchmark weights and returns.

Benchmark: SPY total, with sector returns proxied by SPDR sector ETFs.
If a sector ETF or price series is missing we drop that sector and flag
reduced coverage in the return payload.
"""
from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd
import yfinance as yf

import config
from utils.data_fetch import get_price_history, get_ticker_info

logger = logging.getLogger(__name__)

BENCHMARK_TICKER = getattr(config, "PORTFOLIO_BENCHMARK", "SPY")

# GICS-sector → SPDR sector ETF (benchmark proxies).
SECTOR_ETF = {
    "Information Technology": "XLK",
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Discretionary": "XLY",
    "Consumer Defensive": "XLP",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Basic Materials": "XLB",
    "Materials": "XLB",
}


def _compound(returns: pd.Series) -> float:
    """Return compound return Π(1+r) − 1, NaN-safe."""
    clean = returns.dropna().astype(float)
    if clean.empty:
        return float("nan")
    return float(np.prod(1.0 + clean.values) - 1.0)


def compute_twr(
    results: list[dict],
    holdings: list[dict],
    lookback_days: int = 90,
    benchmark: str = BENCHMARK_TICKER,
) -> dict:
    """Time-weighted total return over the trailing ``lookback_days`` window.

    Returns per-ticker compound returns, NAV-weighted portfolio return, and
    the benchmark return over the same calendar window.
    """
    empty = {
        "portfolio_twr": None, "benchmark_twr": None,
        "excess": None, "per_ticker": {}, "lookback_days": lookback_days,
        "benchmark": benchmark,
    }
    if not results or not holdings:
        return empty

    weights: dict[str, float] = {}
    total = 0.0
    for r, h in zip(results, holdings):
        t = r.get("ticker")
        if not t:
            continue
        price = float(r.get("current_price") or 0)
        qty = float(h.get("quantity") or 0)
        factor = 0.01 if (h.get("currency") == "GBX") else 1.0
        val = price * qty * factor
        weights[t] = weights.get(t, 0.0) + val
        total += val
    if total <= 0:
        return empty
    weights = {t: v / total for t, v in weights.items()}

    per_ticker: dict[str, float] = {}
    port_return = 0.0
    covered_w = 0.0
    for t, w in weights.items():
        df = get_price_history(t)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        closes = df["Close"].tail(lookback_days + 5).dropna()
        if len(closes) < 5:
            continue
        r = float(closes.iloc[-1] / closes.iloc[0] - 1.0)
        per_ticker[t] = round(r, 4)
        port_return += w * r
        covered_w += w
    if covered_w <= 0:
        return empty
    port_twr = port_return / covered_w

    bench_return: float | None = None
    try:
        bdf = get_price_history(benchmark)
        if bdf is not None and not bdf.empty and "Close" in bdf.columns:
            bc = bdf["Close"].tail(lookback_days + 5).dropna()
            if len(bc) >= 5:
                bench_return = float(bc.iloc[-1] / bc.iloc[0] - 1.0)
    except Exception as e:
        logger.debug("benchmark TWR fetch failed: %s", e)

    return {
        "portfolio_twr": round(port_twr, 4),
        "benchmark_twr": round(bench_return, 4) if bench_return is not None else None,
        "excess": round(port_twr - bench_return, 4) if bench_return is not None else None,
        "per_ticker": per_ticker,
        "coverage": round(covered_w, 3),
        "lookback_days": lookback_days,
        "benchmark": benchmark,
    }


def _sector_weights_and_returns(
    weights: Mapping[str, float],
    lookback_days: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    """Roll per-ticker weights up to sectors and compute sector returns."""
    sector_w: dict[str, float] = {}
    sector_ret: dict[str, list[tuple[float, float]]] = {}  # sector → [(w, r)]
    sector_map: dict[str, str] = {}
    for t, w in weights.items():
        info = get_ticker_info(t) or {}
        sector = info.get("sector") or "Unknown"
        sector_map[t] = sector
        df = get_price_history(t)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        closes = df["Close"].tail(lookback_days + 5).dropna()
        if len(closes) < 5:
            continue
        r = float(closes.iloc[-1] / closes.iloc[0] - 1.0)
        sector_w[sector] = sector_w.get(sector, 0.0) + w
        sector_ret.setdefault(sector, []).append((w, r))
    # Weighted sector return
    sector_r: dict[str, float] = {}
    for s, pairs in sector_ret.items():
        tot_w = sum(w for w, _ in pairs)
        if tot_w <= 0:
            continue
        sector_r[s] = sum(w * r for w, r in pairs) / tot_w
    return sector_w, sector_r, sector_map


def compute_brinson_attribution(
    results: list[dict],
    holdings: list[dict],
    lookback_days: int = 90,
) -> dict:
    """Brinson-Fachler sector attribution vs. SPDR sector-ETF benchmarks."""
    empty = {
        "allocation": None, "selection": None, "interaction": None,
        "total_active": None, "per_sector": [], "lookback_days": lookback_days,
    }
    if not results or not holdings:
        return empty

    weights: dict[str, float] = {}
    total = 0.0
    for r, h in zip(results, holdings):
        t = r.get("ticker")
        if not t:
            continue
        price = float(r.get("current_price") or 0)
        qty = float(h.get("quantity") or 0)
        factor = 0.01 if (h.get("currency") == "GBX") else 1.0
        val = price * qty * factor
        weights[t] = weights.get(t, 0.0) + val
        total += val
    if total <= 0:
        return empty
    weights = {t: v / total for t, v in weights.items()}

    port_w, port_r, _sector_map = _sector_weights_and_returns(weights, lookback_days)
    if not port_w:
        return empty

    # Benchmark: download each SPDR sector ETF, compute return; benchmark
    # weights come from the SPY sector composition heuristic — we use
    # simple-average ETF weights if precise SPY weights aren't available.
    etfs = sorted({SECTOR_ETF[s] for s in port_w.keys() if s in SECTOR_ETF})
    bench_sector_r: dict[str, float] = {}
    if etfs:
        try:
            bx = yf.download(
                etfs, period=f"{lookback_days + 5}d",
                progress=False, auto_adjust=True, group_by="ticker",
            )
        except Exception as e:
            logger.debug("Brinson ETF download failed: %s", e)
            bx = None
        if bx is not None and len(bx) >= 5:
            for sector, etf in SECTOR_ETF.items():
                if etf not in etfs:
                    continue
                try:
                    s = bx[etf]["Close"].dropna() if etf in bx.columns.get_level_values(0) else None
                except Exception:
                    s = None
                if s is None or len(s) < 5:
                    continue
                bench_sector_r[sector] = float(s.iloc[-1] / s.iloc[0] - 1.0)

    # Benchmark weights: equal among covered sectors (proxy — SPY weights
    # would need FMP / sector-holdings data).
    covered_sectors = [s for s in port_w if s in bench_sector_r]
    if not covered_sectors:
        return empty
    bench_w = {s: 1.0 / len(covered_sectors) for s in covered_sectors}
    bench_total = sum(bench_w[s] * bench_sector_r[s] for s in covered_sectors)

    alloc = sel = inter = 0.0
    per_sector: list[dict] = []
    for s in covered_sectors:
        wp = port_w.get(s, 0.0)
        wb = bench_w[s]
        rp = port_r.get(s, 0.0)
        rb = bench_sector_r[s]
        a = (wp - wb) * (rb - bench_total)
        se = wb * (rp - rb)
        i = (wp - wb) * (rp - rb)
        alloc += a; sel += se; inter += i
        per_sector.append({
            "sector": s,
            "port_w": round(wp, 3),
            "bench_w": round(wb, 3),
            "port_r": round(rp, 4),
            "bench_r": round(rb, 4),
            "allocation": round(a, 4),
            "selection": round(se, 4),
            "interaction": round(i, 4),
        })
    per_sector.sort(key=lambda d: abs(d["allocation"] + d["selection"]), reverse=True)

    return {
        "allocation": round(alloc, 4),
        "selection": round(sel, 4),
        "interaction": round(inter, 4),
        "total_active": round(alloc + sel + inter, 4),
        "per_sector": per_sector,
        "lookback_days": lookback_days,
        "n_sectors": len(covered_sectors),
    }
