"""Point-in-time synthetic replay for the screener.

This is the safe way to warm-start the self-learning loop. For every replay
date it builds features from prices up to that date and fundamentals available
through ``pit_store.latest_as_of`` only, then writes labelled rows into
``signal_backtest`` with source ``replay_pit_v1``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
import logging
import math
from types import SimpleNamespace
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

import config
from engine.canonical_scores import compute_ev_ebit_score, compute_gpa_score
from engine.discovery_backtest import _connect, init_backtest_db
from engine.enterprise_factors import compute_piotroski_f_score
from engine.factors import compute_factor_scores_from_result, compute_fundamental_quality_metrics
from engine.institutional_prior import neutral_prior, score_universe
from engine.labeling import triple_barrier_from_frame
from engine.technical import analyse_from_df
from utils.pit_store import _available_date, _coerce_date, _load_store
from utils.pit_store import all_tickers as pit_tickers
from utils.price_store import download_price_history
from utils.global_universe import is_excluded_ticker, resolve_yahoo_ticker

logger = logging.getLogger(__name__)

REPLAY_SOURCE = "replay_pit_v1"
_PIT_STORE_CACHE: dict | None = None
_SIGNAL_COLUMNS_CACHE: set[str] | None = None


def _finite(value) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, value)))


def _table_columns(conn) -> set[str]:
    global _SIGNAL_COLUMNS_CACHE
    if _SIGNAL_COLUMNS_CACHE is None:
        _SIGNAL_COLUMNS_CACHE = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(signal_backtest)").fetchall()
        }
    return _SIGNAL_COLUMNS_CACHE


def _pit_store_cached() -> dict:
    global _PIT_STORE_CACHE
    if _PIT_STORE_CACHE is None:
        _PIT_STORE_CACHE = _load_store()
    return _PIT_STORE_CACHE


def reset_pit_cache() -> None:
    """Clear cached PIT fundamentals after an in-process backfill."""
    global _PIT_STORE_CACHE
    _PIT_STORE_CACHE = None


def _latest_as_of_cached(ticker: str, as_of, *, lag_days: int) -> tuple[dict | None, str | None]:
    store = _pit_store_cached()
    entries = store.get("tickers", {}).get(str(ticker).upper())
    if not entries:
        return None, None
    cutoff = _coerce_date(as_of) or date.today()
    best: tuple[date, str] | None = None
    for rd_str, payload in entries.items():
        try:
            rd = date.fromisoformat(str(rd_str)[:10])
        except ValueError:
            continue
        if _available_date(payload, rd, lag_days) <= cutoff and (best is None or rd > best[0]):
            best = (rd, rd_str)
    if best is None:
        return None, None
    return dict(entries[best[1]]), best[1]


def _available_snapshots_cached(
    ticker: str,
    as_of,
    *,
    lag_days: int,
) -> list[tuple[date, str, dict]]:
    """Return PIT snapshots available at as_of, newest first."""
    store = _pit_store_cached()
    entries = store.get("tickers", {}).get(str(ticker).upper())
    if not entries:
        return []
    cutoff = _coerce_date(as_of) or date.today()
    available: list[tuple[date, str, dict]] = []
    for rd_str, payload in entries.items():
        if not isinstance(payload, dict):
            continue
        try:
            rd = date.fromisoformat(str(rd_str)[:10])
        except ValueError:
            continue
        if _available_date(payload, rd, lag_days) <= cutoff:
            available.append((rd, rd_str, dict(payload)))
    available.sort(key=lambda item: item[0], reverse=True)
    return available


def _prior_snapshot_cached(ticker: str, before_report_date: str) -> tuple[dict | None, str | None]:
    store = _pit_store_cached()
    entries = store.get("tickers", {}).get(str(ticker).upper())
    if not entries:
        return None, None
    target = _coerce_date(before_report_date)
    if target is None:
        return None, None
    dates: list[date] = []
    for rd_str in entries.keys():
        try:
            dates.append(date.fromisoformat(str(rd_str)[:10]))
        except ValueError:
            continue
    prior = [d for d in dates if d < target]
    if not prior:
        return None, None
    target_prior = target - timedelta(days=365)
    prior.sort(key=lambda d: abs((d - target_prior).days))
    best = prior[0].isoformat()
    return dict(entries[best]), best


def _sum_snapshots(rows: list[dict], key: str) -> float | None:
    total = 0.0
    found = False
    for row in rows:
        value = _finite(row.get(key))
        if value is None:
            return None
        total += value
        found = True
    return total if found else None


def _snapshots_look_quarterly(rows: list[tuple[date, str, dict]]) -> bool:
    if len(rows) < 4:
        return False
    min_gap = int(getattr(config, "HISTORICAL_REPLAY_TTM_MIN_QUARTER_GAP_DAYS", 45))
    max_gap = int(getattr(config, "HISTORICAL_REPLAY_TTM_MAX_QUARTER_GAP_DAYS", 130))
    dates = [rd for rd, _, _ in rows[:4]]
    gaps = [(dates[i] - dates[i + 1]).days for i in range(len(dates) - 1)]
    return all(min_gap <= gap <= max_gap for gap in gaps)


def _ttm_snapshot_features(
    ticker: str,
    as_of: pd.Timestamp,
    *,
    lag_days: int,
) -> dict:
    """Build PIT-safe TTM flow features from cached quarterly snapshots."""
    available = _available_snapshots_cached(ticker, as_of.date(), lag_days=lag_days)
    if not available:
        return {}
    # Use dated quarterly/PIT snapshots. Daily yfinance_info snapshots lack
    # accepted-date metadata and are only fallback evidence, not TTM quarters.
    quarterly = [
        (rd, key, payload)
        for rd, key, payload in available
        if str(payload.get("_source") or "").lower() != "yfinance_info"
    ]
    if len(quarterly) < 4:
        return {}
    if not _snapshots_look_quarterly(quarterly[:4]):
        return {}

    latest_rows = [payload for _, _, payload in quarterly[:4]]
    latest = latest_rows[0]
    prior_rows = [payload for _, _, payload in quarterly[4:8]]
    prior_balance = quarterly[4][2] if len(quarterly) >= 5 else None

    total_assets = _finite(latest.get("total_assets"))
    gross_profit_ttm = _sum_snapshots(latest_rows, "gross_profit")
    if total_assets is None or total_assets <= 0:
        return {}
    if gross_profit_ttm is not None:
        gpa_ttm = gross_profit_ttm / total_assets
        max_gpa = float(getattr(config, "HISTORICAL_REPLAY_TTM_GPA_MAX", 1.50))
        if gpa_ttm > max_gpa:
            return {}
    net_income_ttm = _sum_snapshots(latest_rows, "net_income")
    revenue_ttm = _sum_snapshots(latest_rows, "revenue")
    operating_cashflow_ttm = _sum_snapshots(latest_rows, "operating_cashflow")
    capex_ttm = _sum_snapshots(latest_rows, "capital_expenditure")
    ebit_ttm = _sum_snapshots(latest_rows, "ebit")
    ebitda_ttm = _sum_snapshots(latest_rows, "ebitda")
    eps_ttm = _sum_snapshots(latest_rows, "eps")

    out: dict = {"ttm_source": "pit_quarterly"}
    if total_assets:
        out["total_assets"] = total_assets
    if gross_profit_ttm is not None:
        out["gross_profit"] = gross_profit_ttm
    if net_income_ttm is not None:
        out["net_income"] = net_income_ttm
    if revenue_ttm is not None:
        out["revenue"] = revenue_ttm
    if operating_cashflow_ttm is not None:
        out["operating_cashflow"] = operating_cashflow_ttm
    if capex_ttm is not None:
        out["capital_expenditure"] = capex_ttm
    if ebit_ttm is not None:
        out["ebit"] = ebit_ttm
    if ebitda_ttm is not None:
        out["ebitda"] = ebitda_ttm
    if eps_ttm is not None:
        out["eps"] = eps_ttm

    for key in (
        "shares_outstanding",
        "total_debt",
        "long_term_debt",
        "short_term_debt",
        "cash",
        "current_assets",
        "current_liabilities",
        "shareholders_equity",
        "total_liabilities",
    ):
        value = _finite(latest.get(key))
        if value is not None:
            out[key] = value

    if len(prior_rows) >= 4 and _snapshots_look_quarterly(quarterly[4:8]):
        prior = dict(prior_balance or prior_rows[0])
        prior["net_income"] = _sum_snapshots(prior_rows[:4], "net_income")
        prior["gross_profit"] = _sum_snapshots(prior_rows[:4], "gross_profit")
        prior["revenue"] = _sum_snapshots(prior_rows[:4], "revenue")
        prior["operating_cashflow"] = _sum_snapshots(prior_rows[:4], "operating_cashflow")
        out["_ttm_prior"] = prior
    return out


@dataclass(frozen=True)
class ReplayStats:
    attempted: int = 0
    inserted: int = 0
    updated: int = 0
    skipped_existing: int = 0
    skipped_no_price: int = 0


@dataclass(frozen=True)
class ReplayEnrichStats:
    attempted: int = 0
    updated: int = 0
    skipped_no_price: int = 0
    errors: int = 0


def month_ends(start: str, end: str) -> list[pd.Timestamp]:
    dates = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="ME")
    if dates.empty:
        return []
    return [pd.Timestamp(d).normalize() for d in dates]


def week_ends(start: str, end: str) -> list[pd.Timestamp]:
    dates = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="W-FRI")
    if dates.empty:
        return []
    return [pd.Timestamp(d).normalize() for d in dates]


def business_days(start: str, end: str) -> list[pd.Timestamp]:
    dates = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="B")
    if dates.empty:
        return []
    return [pd.Timestamp(d).normalize() for d in dates]


def replay_dates(start: str, end: str, frequency: str = "monthly") -> list[pd.Timestamp]:
    """Return replay as-of dates for the requested offline cadence."""
    freq = str(frequency or "monthly").lower().strip()
    if freq == "monthly":
        return month_ends(start, end)
    if freq == "weekly":
        return week_ends(start, end)
    if freq in {"daily", "business_daily", "business-daily"}:
        return business_days(start, end)
    if freq == "latest":
        return [pd.Timestamp(end).normalize()]
    raise ValueError(f"Unsupported replay frequency: {frequency!r}")


def signal_backtest_tickers(max_tickers: int | None = None) -> list[str]:
    with _connect() as conn:
        sql = "SELECT DISTINCT ticker FROM signal_backtest ORDER BY ticker"
        if max_tickers:
            sql += f" LIMIT {int(max_tickers)}"
        rows = conn.execute(sql).fetchall()
    return [str(row[0]).upper() for row in rows if row[0]]


def replay_universe(*, from_signal_db: bool = True, max_tickers: int | None = None) -> list[str]:
    tickers: set[str] = set()
    if from_signal_db:
        tickers.update(signal_backtest_tickers(max_tickers=max_tickers))
    tickers.update(pit_tickers())
    out = sorted(tickers)
    if max_tickers:
        out = out[: int(max_tickers)]
    return out


def _bar_at_or_after(frame: pd.DataFrame, as_of: pd.Timestamp, days: int, column: str = "Close") -> float | None:
    if frame is None or frame.empty or column not in frame.columns:
        return None
    target = as_of + pd.Timedelta(days=int(days))
    path = frame.loc[frame.index >= target]
    if path.empty:
        return None
    try:
        return float(path[column].iloc[0])
    except Exception:
        return None


def _price_features(frame: pd.DataFrame, as_of: pd.Timestamp) -> dict | None:
    if frame is None or frame.empty:
        return None
    hist = frame.loc[frame.index <= as_of].copy()
    if len(hist) < 220 or "Close" not in hist.columns:
        return None
    close = hist["Close"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(close) < 220:
        return None
    high = hist["High"].astype(float) if "High" in hist.columns else close
    low = hist["Low"].astype(float) if "Low" in hist.columns else close
    volume = hist["Volume"].astype(float) if "Volume" in hist.columns else pd.Series(index=hist.index, dtype=float)
    price = _finite(close.iloc[-1])
    if price is None or price <= 0:
        return None

    def ret(n: int) -> float | None:
        if len(close) <= n:
            return None
        return float(close.iloc[-1] / close.iloc[-n - 1] - 1.0)

    sma50 = _finite(close.tail(50).mean())
    sma200 = _finite(close.tail(200).mean())
    if sma50 is None or sma200 is None or sma200 <= 0:
        return None
    daily_ret = close.pct_change(fill_method=None)
    vol_20d = _finite(daily_ret.tail(20).std() * np.sqrt(252)) or 0.0
    atr = _finite((high.tail(14) - low.tail(14)).mean())
    ret_10 = ret(10)
    ret_30 = ret(30)
    ret_90 = ret(90)
    ret_12m1m = None
    if len(close) > 252:
        prior_1m = close.iloc[-22]
        prior_12m = close.iloc[-252]
        if prior_12m:
            ret_12m1m = float(prior_1m / prior_12m - 1.0)
    above_200 = 1.0 if price > sma200 else -1.0
    momentum = np.nanmean([
        0.0 if ret_10 is None else np.clip(ret_10 / 0.10, -1.0, 1.0),
        0.0 if ret_30 is None else np.clip(ret_30 / 0.20, -1.0, 1.0),
        0.0 if ret_90 is None else np.clip(ret_90 / 0.35, -1.0, 1.0),
        above_200,
    ])
    volume_ratio = 1.0
    try:
        vol_vals = volume.dropna()
        if len(vol_vals) >= 60:
            avg_10 = float(vol_vals.tail(10).mean())
            avg_60 = float(vol_vals.tail(60).mean())
            volume_ratio = avg_10 / max(avg_60, 1.0)
    except Exception:
        pass
    high_252 = _finite(close.tail(min(252, len(close))).max())
    pct_from_high = float(price / high_252) if high_252 and high_252 > 0 else None
    try:
        technical = analyse_from_df(hist)
    except Exception:
        technical = {}
    technical_score = _finite(technical.get("score"))
    return {
        "signal_price": price,
        "technical_score": float(np.clip(technical_score if technical_score is not None else momentum, -1.0, 1.0)),
        "momentum_score": float(np.clip((momentum + 1.0) / 2.0, 0.0, 1.0)),
        "sma_50": sma50,
        "sma_200": sma200,
        "atr": _finite(technical.get("atr")) or atr,
        "rsi": _finite(technical.get("rsi")),
        "adx": _finite(technical.get("adx")),
        "bb_lower": _finite(technical.get("bb_lower")),
        "bb_pct": _finite(technical.get("bb_pct")),
        "return_10d_prior": ret_10,
        "return_30d_prior": ret_30,
        "return_90d_prior": ret_90,
        "ret_12m1m": ret_12m1m,
        "vol_20d": vol_20d,
        "volume_ratio": volume_ratio,
        "pct_from_high_252d": pct_from_high,
        "price_vs_sma200_stretch": float(price / sma200 - 1.0) if sma200 else None,
    }


def _rank_percentiles(values: list[float | None]) -> list[float]:
    series = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan)
    ranks = series.rank(method="average", pct=True)
    return [float(v) if pd.notna(v) else 0.5 for v in ranks]


def _apply_cross_sectional_momentum_scores(price_features: dict[str, dict]) -> None:
    """Align replay momentum with discovery's cross-sectional momentum lens."""
    items = [(ticker, feat) for ticker, feat in sorted(price_features.items()) if feat]
    if not items:
        return
    rank_12m1m = _rank_percentiles([feat.get("ret_12m1m") for _, feat in items])
    rank_90d = _rank_percentiles([feat.get("return_90d_prior") for _, feat in items])
    rank_30d = _rank_percentiles([feat.get("return_30d_prior") for _, feat in items])
    rank_10d = _rank_percentiles([feat.get("return_10d_prior") for _, feat in items])
    rank_vol = _rank_percentiles([feat.get("volume_ratio") for _, feat in items])
    rank_high = _rank_percentiles([feat.get("pct_from_high_252d") for _, feat in items])

    for idx, (_ticker, feat) in enumerate(items):
        # Discovery also includes a sector-relative term. Replay may not have
        # the sector map here, so reuse 90d rank as a conservative proxy rather
        # than falling back to single-name momentum.
        score = (
            0.20 * rank_12m1m[idx]
            + 0.35 * rank_90d[idx]
            + 0.15 * rank_30d[idx]
            + 0.10 * rank_10d[idx]
            + 0.10 * rank_vol[idx]
            + 0.10 * rank_high[idx]
        )
        feat["momentum_score"] = float(np.clip(score, 0.0, 1.0))


def _fundamental_features(ticker: str, as_of: pd.Timestamp, *, signal_price: float | None = None) -> dict:
    lag_days = int(getattr(config, "PIT_FUNDAMENTAL_LAG_DAYS", 45))
    latest, report_date = _latest_as_of_cached(
        ticker,
        as_of.date(),
        lag_days=lag_days,
    )
    if not latest or not report_date:
        return {}
    prior, _ = _prior_snapshot_cached(ticker, report_date)
    ttm = _ttm_snapshot_features(ticker, as_of, lag_days=lag_days)
    if ttm:
        ttm_prior = ttm.get("_ttm_prior")
        latest = {
            **latest,
            **{k: v for k, v in ttm.items() if not str(k).startswith("_")},
        }
        if isinstance(ttm_prior, dict):
            prior = ttm_prior
    f_result = compute_piotroski_f_score(latest, prior)
    total_assets = _finite(latest.get("total_assets"))
    gross_profit = _finite(latest.get("gross_profit"))
    gpa = None
    gpa_score = None
    try:
        if total_assets and gross_profit is not None:
            gpa = gross_profit / total_assets
            gpa_score = compute_gpa_score(gpa)
    except Exception:
        pass
    revenue = _finite(latest.get("revenue"))
    net_income = _finite(latest.get("net_income"))
    operating_cashflow = _finite(latest.get("operating_cashflow"))
    capex = _finite(latest.get("capital_expenditure"))
    eps = _finite(latest.get("eps"))
    shares = _finite(latest.get("shares_outstanding"))
    ebit = _finite(latest.get("ebit"))
    ebitda = _finite(latest.get("ebitda"))
    debt = _finite(latest.get("total_debt")) or 0.0
    cash = _finite(latest.get("cash")) or 0.0
    current_assets = _finite(latest.get("current_assets"))
    current_liabilities = _finite(latest.get("current_liabilities"))
    equity = _finite(latest.get("shareholders_equity"))
    if equity is None:
        total_liabilities = _finite(latest.get("total_liabilities"))
        if total_assets and total_liabilities is not None:
            equity = total_assets - total_liabilities

    price = _finite(signal_price)
    market_cap = (price * shares) if price and shares and shares > 0 else None
    enterprise_value = None
    if market_cap is not None:
        enterprise_value = market_cap + debt - cash
        if enterprise_value <= 0:
            enterprise_value = None

    ev_ebit = enterprise_value / ebit if enterprise_value and ebit and ebit > 0 else None
    ev_ebitda = enterprise_value / ebitda if enterprise_value and ebitda and ebitda > 0 else None
    ebit_yield = ebit / enterprise_value if enterprise_value and ebit and ebit > 0 else None
    ev_ebit_score = compute_ev_ebit_score(ebit_yield)
    pe_ratio = price / eps if price and eps and eps > 0 else None

    fcf = None
    if operating_cashflow is not None:
        if capex is None:
            fcf = operating_cashflow
        else:
            # FMP capex is usually negative; some feeds store absolute capex.
            fcf = operating_cashflow + capex if capex < 0 else operating_cashflow - capex
    fcf_yield = fcf / market_cap if fcf is not None and market_cap and market_cap > 0 else None
    fcf_to_assets = fcf / total_assets if fcf is not None and total_assets and total_assets > 0 else None
    profit_margin = net_income / revenue if net_income is not None and revenue and revenue > 0 else None
    roe = net_income / equity if net_income is not None and equity and equity > 0 else None
    current_ratio = (
        current_assets / current_liabilities
        if current_assets is not None and current_liabilities and current_liabilities > 0
        else None
    )
    revenue_growth = None
    if prior:
        prior_revenue = _finite(prior.get("revenue"))
        if revenue is not None and prior_revenue and prior_revenue > 0:
            revenue_growth = revenue / prior_revenue - 1.0

    quality_info = {
        "grossProfits": gross_profit,
        "grossProfit": gross_profit,
        "totalAssets": total_assets,
        "returnOnEquity": roe,
        "freeCashflow": fcf,
    }
    quality_metrics = compute_fundamental_quality_metrics(quality_info)
    quality_score = quality_metrics.get("quality_score")
    if quality_metrics.get("gross_profitability") is not None:
        gpa = quality_metrics.get("gross_profitability")
        gpa_score = compute_gpa_score(gpa)
    if quality_metrics.get("fcf_to_assets") is not None:
        fcf_to_assets = quality_metrics.get("fcf_to_assets")

    enterprise_scores = [
        v for v in (f_result.get("f_score_score"), gpa_score, ev_ebit_score)
        if v is not None
    ]
    fundamental_score = (
        float(np.mean(enterprise_scores))
        if enterprise_scores
        else None
    )

    fields = {
        "pit_source": str(latest.get("_source") or "unknown"),
        "f_score": f_result.get("f_score"),
        "f_score_coverage": f_result.get("f_score_coverage"),
        "f_score_gate": f_result.get("f_score_gate"),
        "f_score_score": f_result.get("f_score_score"),
        "gpa": gpa,
        "gpa_score": gpa_score,
        "gross_profitability": gpa,
        "fcf_to_assets": fcf_to_assets,
        "earnings_stability": quality_metrics.get("earnings_stability"),
        "eps_growth_variance_5y": quality_metrics.get("eps_growth_variance_5y"),
        "revenue_growth": revenue_growth,
        "profit_margin": profit_margin,
        "roe": roe,
        "current_ratio": current_ratio,
        "net_debt_ebitda": ((debt - cash) / ebitda) if ebitda and ebitda > 0 else None,
        "cash_to_debt": cash / debt if debt and debt > 0 else None,
        "pe_ratio": pe_ratio,
        "fcf_yield": fcf_yield,
        "ev_ebit": ev_ebit,
        "ev_ebit_score": ev_ebit_score,
        "quality_score_fundamental": quality_score,
        "fundamental_score": fundamental_score,
        "ttm_source": ttm.get("ttm_source") if ttm else None,
        # Used only before insert to derive replay factor scores; unknown DB columns are filtered.
        "market_cap": market_cap,
        "enterprise_value": enterprise_value,
        "ev_ebitda": ev_ebitda,
        "ebit_yield": ebit_yield,
    }
    try:
        factor_scores = compute_factor_scores_from_result(fields)
        if any(fields.get(k) is not None for k in ("quality_score_fundamental", "gpa_score", "f_score_score", "gpa", "f_score")):
            for key in ("quality_factor_score", "gpa_factor_score", "f_score_factor_score"):
                if factor_scores.get(key) is not None:
                    fields[key] = factor_scores[key]
        if any(fields.get(k) is not None for k in ("gpa_score", "gpa", "gross_profitability", "f_score_score", "earnings_stability")):
            for key in ("qmj_factor_score", "qmj_component_count"):
                if factor_scores.get(key) is not None:
                    fields[key] = factor_scores[key]
        if any(fields.get(k) is not None for k in ("pe_ratio", "peg_ratio", "fcf_yield", "ev_ebit_score", "pb_score", "ps_score")):
            fields["value_factor_score"] = factor_scores.get("value_factor_score")
            if factor_scores.get("ev_ebit_factor_score") is not None:
                fields["ev_ebit_factor_score"] = factor_scores["ev_ebit_factor_score"]
    except Exception:
        logger.debug("Replay fundamental factor derivation failed for %s %s", ticker, as_of.date())
    return fields


def _forward_labels(frame: pd.DataFrame, as_of: pd.Timestamp, price: float, atr: float | None) -> dict:
    updates: dict = {}
    for horizon in (5, 10, 30, 60, 90):
        px = _bar_at_or_after(frame, as_of, horizon, "Close")
        if px is None:
            continue
        updates[f"price_{horizon}d"] = px
        updates[f"return_{horizon}d"] = round((px - price) / price * 100.0, 4)
        updates[f"evaluated_{horizon}d"] = 1
    tb = triple_barrier_from_frame(
        frame,
        as_of.date().isoformat(),
        price,
        atr=atr,
        horizon_days=int(getattr(config, "TRIPLE_BARRIER_HORIZON_DAYS", 30)),
    )
    if tb is not None:
        updates.update({
            "tb_label": tb.label,
            "tb_return": round(tb.return_pct, 4),
            "tb_days": tb.days,
            "tb_hit": tb.hit,
            "tb_horizon": int(getattr(config, "TRIPLE_BARRIER_HORIZON_DAYS", 30)),
            "tb_updated_at": datetime.now().isoformat(),
        })
    return updates


def _replay_factor_payload(ticker: str, as_of: pd.Timestamp, base_values: dict) -> dict:
    """Build PIT-safe factor values for a replay row."""
    signal_price = _finite(base_values.get("signal_price"))
    if signal_price is None or signal_price <= 0:
        return {}
    ffeat = _fundamental_features(ticker, as_of, signal_price=signal_price)
    snapshot = {**base_values, **ffeat}
    try:
        factor_scores = compute_factor_scores_from_result(snapshot)
    except Exception as exc:
        logger.debug("Replay factor payload failed for %s %s: %s", ticker, as_of.date(), exc)
        factor_scores = {}
    return {**ffeat, **factor_scores}


def _json_or_none(value) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except Exception:
        return None


def _replay_entry_stance(row: dict) -> str:
    """PIT-safe entry stance from as-of price/technical state only."""
    rsi = _finite(row.get("rsi"))
    stretch = _finite(row.get("price_vs_sma200_stretch"))
    if (
        (rsi is not None and rsi > float(getattr(config, "RSI_NEUTRAL_CAP", 80)))
        or (stretch is not None and stretch > 0.60)
    ):
        return "Watch Only"
    if (
        (rsi is not None and rsi > float(getattr(config, "RSI_STRONG_BUY_MAX", 75)))
        or (
            stretch is not None
            and stretch > float(getattr(config, "STRETCH_200DMA_STRONG_BUY_MAX", 0.35))
        )
    ):
        return "Pullback Preferred"
    return "Ready"


def _replay_data_confidence(row: dict) -> float:
    critical = [
        "technical_score", "momentum_score", "f_score", "gpa", "ev_ebit",
        "quality_factor_score", "value_factor_score", "momentum_factor_score",
        "sma_200", "rsi", "atr",
    ]
    present = sum(1 for key in critical if _finite(row.get(key)) is not None)
    return float(present / max(1, len(critical)))


def _pit_stop_loss(row: Mapping, frame: pd.DataFrame | None) -> dict:
    """PIT-safe support-confluence stop mirroring engine.stops."""
    current_price = _finite(row.get("signal_price"))
    if current_price is None or current_price <= 0:
        return {"stop_loss": None, "method": "N/A"}

    atr = _finite(row.get("atr"))
    sma_200 = _finite(row.get("sma_200"))
    sma_50 = _finite(row.get("sma_50"))
    bb_lower = _finite(row.get("bb_lower"))
    trail_low = float(getattr(config, "TRAIL_PCT_LOW_VOL", 0.08))
    trail_high = float(getattr(config, "TRAIL_PCT_HIGH_VOL", 0.15))
    vol_pct = 50.0
    vix_pct = float(getattr(config, "HISTORICAL_REPLAY_VIX_PERCENTILE_DEFAULT", 50.0))
    trail_pct = float(np.interp(vol_pct, [20, 80], [trail_low, trail_high]))
    atr_mult = 2.5 * (1.0 + 0.5 * (vix_pct / 100.0))

    hist = frame.copy() if frame is not None else pd.DataFrame()
    if not hist.empty and "Close" in hist.columns:
        hist = hist.dropna(subset=["Close"])
    high = hist["High"].astype(float) if not hist.empty and "High" in hist.columns else pd.Series(dtype=float)
    low = hist["Low"].astype(float) if not hist.empty and "Low" in hist.columns else pd.Series(dtype=float)

    swing_low = None
    try:
        if len(low.dropna()) >= 20:
            swing_low = float(low.dropna().tail(20).min())
    except Exception:
        swing_low = None

    candidates: dict[str, tuple[float, float]] = {}
    if atr is not None and atr > 0:
        atr_stop = current_price - atr * atr_mult
        if 0 < atr_stop < current_price:
            candidates["atr"] = (atr_stop, 0.35)
    if sma_200 is not None and 0.85 * current_price < sma_200 < current_price:
        candidates["sma_200"] = (sma_200 * 0.97, 0.25)
    if sma_50 is not None and 0.90 * current_price < sma_50 < current_price:
        candidates["sma_50"] = (sma_50 * 0.98, 0.20)
    if swing_low is not None and 0.85 * current_price < swing_low < current_price:
        candidates["swing_low"] = (swing_low * 0.99, 0.15)
    if bb_lower is not None and 0 < bb_lower < current_price:
        candidates["bb_lower"] = (bb_lower, 0.05)

    if candidates:
        total_weight = sum(weight for _, weight in candidates.values())
        weighted_avg = sum(price * weight for price, weight in candidates.values()) / total_weight
        valid = {
            key: (price, weight)
            for key, (price, weight) in candidates.items()
            if price <= weighted_avg * 1.02
        }
        if valid:
            method = max(valid, key=lambda key: valid[key][0])
            stop_price = valid[method][0]
            method = f"{method} confluence"
        else:
            stop_price = weighted_avg
            method = "weighted confluence"
    else:
        recent_high = None
        try:
            if len(high.dropna()) > 0:
                recent_high = float(high.dropna().tail(63).max())
        except Exception:
            recent_high = None
        reference = min(recent_high, current_price) if recent_high else current_price
        stop_price = reference * (1.0 - trail_pct)
        method = "pct_fallback"

    if stop_price >= current_price:
        stop_price = current_price * (1.0 - trail_pct)
        method = "pct_fallback"
    return {
        "stop_loss": round(float(stop_price), 2),
        "method": method,
        "stop_distance_pct": round((current_price - float(stop_price)) / current_price * 100.0, 2),
    }


def _pit_take_profit(
    row: Mapping,
    frame: pd.DataFrame | None,
    *,
    stop_loss: float | None,
    entry_price: float | None,
    entry_lens: str | None,
) -> dict:
    """PIT-safe target using live target rules against as-of resistance."""
    current_price = _finite(row.get("signal_price"))
    reference_price = _finite(entry_price) or current_price
    if reference_price is None or reference_price <= 0:
        return {"take_profit": None, "method": "N/A"}

    lens = str(entry_lens or "").strip()
    if lens == "momentum":
        rr_multiple = 2.5
    elif lens in ("quality", "value"):
        rr_multiple = 2.0
    else:
        rr_multiple = float(getattr(config, "RISK_REWARD_RATIO", 2.5))

    targets: dict[str, float] = {}
    stop = _finite(stop_loss)
    if stop is not None and stop < reference_price:
        targets["R/R ratio"] = reference_price + (reference_price - stop) * rr_multiple

    high = pd.Series(dtype=float)
    if frame is not None and not frame.empty and "High" in frame.columns:
        high = frame["High"].astype(float).dropna()
    resistance = None
    if len(high) > 20:
        high_6m = float(high.tail(126).max())
        if high_6m > reference_price:
            targets["resistance"] = high_6m
            resistance = high_6m
        high_52w = float(high.tail(min(252, len(high))).max())
        if high_52w > reference_price and high_52w != high_6m:
            targets["52w high"] = high_52w
            if resistance is None:
                resistance = high_52w

    if not targets:
        targets["default 15%"] = reference_price * 1.15

    if "R/R ratio" in targets and resistance is not None and lens in ("momentum", "quality", "value"):
        if lens == "value":
            final_target = min(targets["R/R ratio"], resistance)
            method = "value rr/resistance"
        else:
            final_target = max(targets["R/R ratio"], resistance)
            method = f"{lens} rr/resistance"
    else:
        method = min(targets, key=targets.get)
        final_target = targets[method]
    return {"take_profit": round(float(final_target), 2), "method": method}


def _replay_execution_fields(
    *,
    ticker: str,
    frame: pd.DataFrame | None,
    as_of: pd.Timestamp,
    price_features: Mapping,
    candidate_metadata: Mapping | None = None,
) -> dict:
    """Build PIT-safe execution fields, optionally lens-aligned to live."""
    hist = frame.loc[frame.index <= as_of].copy() if frame is not None and not frame.empty else pd.DataFrame()
    lens = None
    if candidate_metadata:
        lens = candidate_metadata.get("_entry_lens") or candidate_metadata.get("entry_lens")
    lens = str(lens or "").strip() or None

    try:
        from engine.stops import calculate_entry_strategy

        entry_data = calculate_entry_strategy(
            _finite(price_features.get("signal_price")),
            _finite(price_features.get("atr")),
            sma_50=_finite(price_features.get("sma_50")),
            bb_lower=_finite(price_features.get("bb_lower")),
            vol_percentile=50.0,
            entry_lens=lens or "momentum",
        )
    except Exception:
        entry_data = {
            "entry_price": price_features.get("signal_price"),
            "entry_method": "replay_close",
            "fill_probability": 1.0,
        }

    stop_data = _pit_stop_loss(price_features, hist)
    target_data = _pit_take_profit(
        price_features,
        hist,
        stop_loss=stop_data.get("stop_loss"),
        entry_price=entry_data.get("entry_price"),
        entry_lens=lens,
    )
    out = {
        "entry_lens": lens,
        "entry_price": entry_data.get("entry_price"),
        "entry_method": entry_data.get("entry_method"),
        "fill_probability": entry_data.get("fill_probability"),
        "stop_loss": stop_data.get("stop_loss"),
        "stop_method": stop_data.get("method"),
        "stop_distance_pct": stop_data.get("stop_distance_pct"),
        "take_profit": target_data.get("take_profit"),
        "target_method": target_data.get("method"),
    }
    return out


def _apply_replay_readiness_fields(rows: dict[str, dict]) -> None:
    """Populate PIT-safe action-gate and ready-contract fields on replay rows.

    Deliberately excludes live-only sentiment and current holdings. The
    institutional prior is recomputed from the historical replay cohort for the
    same as-of date, so percentile inputs are point-in-time cross-sectional.
    """
    if not rows:
        return

    for ticker, row in rows.items():
        row["ticker"] = ticker
    priors = score_universe(list(rows.values()))
    candidates: list[SimpleNamespace] = []
    for ticker, row in rows.items():
        signal_price = _finite(row.get("signal_price"))
        entry_price = _finite(row.get("entry_price")) or signal_price
        stop_loss = _finite(row.get("stop_loss"))
        take_profit = _finite(row.get("take_profit"))
        rr_ratio = None
        if entry_price is not None and stop_loss is not None and take_profit is not None:
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
            if risk > 0 and reward > 0:
                rr_ratio = reward / risk
        prior = priors.get(str(ticker).upper(), neutral_prior())
        fill_probability = _finite(row.get("fill_probability"))
        if fill_probability is None and signal_price is not None:
            fill_probability = 1.0
        row.update({
            "entry_lens": row.get("entry_lens") or "replay_close",
            "entry_price": entry_price,
            "entry_method": row.get("entry_method") or "replay_close",
            "entry_stance": row.get("entry_stance") or _replay_entry_stance(row),
            "fill_probability": fill_probability,
            "planned_position_weight": float(getattr(config, "READY_STRONG_BUY_MIN_POSITION_WEIGHT", 0.005)) * 2.0,
            "planned_risk_amount": None,
            "position_sizing_method": "replay_fixed_fraction",
            "r_r_ratio": rr_ratio,
            "current_price": signal_price,
            "institutional_prior_score": prior.score,
            "institutional_prior_percentile": prior.percentile,
            "institutional_prior_confidence": prior.confidence,
            "institutional_prior_components": _json_or_none(prior.components),
        })
        payload = dict(row)
        payload.update({
            "ticker": ticker,
            "sector": row.get("sector") or "Unknown",
            "action": row.get("action") or "NEUTRAL",
            "ready_contract_core_status": "FAIL",
            "ready_contract_score": None,
        })
        candidates.append(SimpleNamespace(**payload))

    try:
        from engine.action_gates import apply_action_gates, cap_action

        apply_action_gates(candidates)
        for candidate in candidates:
            row = rows[str(candidate.ticker)]
            ceiling = getattr(candidate, "action_gate_ceiling", "STRONG BUY")
            action = str(row.get("action") or "NEUTRAL")
            row["action_gate_ceiling"] = ceiling
            row["action_gate_reasons"] = _json_or_none(getattr(candidate, "action_gate_reasons", []))
            row["action_gate_flags_json"] = _json_or_none(getattr(candidate, "action_gate_flags", {}))
            row["limit_price"] = getattr(candidate, "limit_price", None)
            row["limit_price_method"] = getattr(candidate, "limit_price_method", None)
            row["limit_price_rationale"] = getattr(candidate, "limit_price_rationale", None)
            row["action"] = cap_action(action, ceiling)
    except Exception as exc:
        logger.debug("Replay action-gate backfill failed: %s", exc)

    try:
        from engine.discovery import (
            _evaluate_discovery_gates_v2,
            _evaluate_ready_strong_buy_contract,
            _evaluate_trap_safeguard,
        )
    except Exception as exc:
        logger.debug("Replay ready-contract helpers unavailable: %s", exc)
        return

    for ticker, row in rows.items():
        sector = str(row.get("sector") or "Unknown")
        industry = str(row.get("industry") or "")
        stretch = _finite(row.get("price_vs_sma200_stretch"))
        gate_v2_status, gate_v2_reasons = _evaluate_discovery_gates_v2(
            row, sector=sector, industry=industry, stretch=stretch
        )
        trap_triggered, trap_reason = _evaluate_trap_safeguard(
            row, sector=sector, industry=industry, stretch=stretch
        )
        row["gate_v2_status"] = gate_v2_status
        row["gate_v2_reasons"] = _json_or_none(gate_v2_reasons)
        row["trap_safeguard_triggered"] = 1 if trap_triggered else 0
        row["trap_safeguard_reason"] = trap_reason

        prior = priors.get(str(ticker).upper(), neutral_prior())
        status, reasons, details = _evaluate_ready_strong_buy_contract(
            gate_status=gate_v2_status,
            trap_triggered=trap_triggered,
            prior=prior,
            entry_stance=str(row.get("entry_stance") or ""),
            entry_price=row.get("entry_price"),
            stop_loss=row.get("stop_loss"),
            take_profit=row.get("take_profit"),
            rr_ratio=row.get("r_r_ratio"),
            position_weight=row.get("planned_position_weight"),
            data_confidence=_replay_data_confidence(row),
            gate_reasons=gate_v2_reasons,
            return_details=True,
        )
        ceiling = str(row.get("action_gate_ceiling") or "STRONG BUY")
        if ceiling != "STRONG BUY":
            status = "FAIL"
            reasons = list(reasons) + [f"action gate ceiling {ceiling}"]
        row["ready_contract_status"] = status
        row["ready_contract_reasons"] = _json_or_none(reasons)
        row["ready_contract_score"] = details.get("score")
        row["strong_buy_eligible"] = 1 if status == "PASS" else 0
        row["strong_buy_blockers"] = _json_or_none([] if status == "PASS" else reasons)

    try:
        from engine.value_cap_shadow import annotate_value_cap_shadow

        annotations = annotate_value_cap_shadow(list(rows.values()), config_module=config)
        for ann in annotations:
            ticker = str(ann.get("ticker") or "").upper()
            if ticker in rows:
                rows[ticker]["value_cap_shadow_bucket"] = ann.get("primary_bucket")
                rows[ticker]["value_cap_shadow_json"] = _json_or_none(ann)
    except Exception as exc:
        logger.debug("Replay value-cap shadow annotation failed: %s", exc)


def _insert_replay_row(
    ticker: str,
    as_of: pd.Timestamp,
    values: dict,
    conn=None,
    *,
    refresh_existing: bool = False,
) -> str:
    run_date = as_of.date().isoformat()
    values = dict(values)
    signal_price = _finite(values.get("signal_price"))
    if signal_price is None or signal_price <= 0:
        return "no_price"
    values["signal_price"] = signal_price
    if conn is None:
        with _connect() as owned_conn:
            return _insert_replay_row(
                ticker,
                as_of,
                values,
                conn=owned_conn,
                refresh_existing=refresh_existing,
            )
    values.update({
        "run_date": run_date,
        "ticker": ticker,
        "source": REPLAY_SOURCE,
        "name": ticker,
        "action": values.get("action") or ("BUY" if values.get("aggregate_score", 0) >= 0.2 else "NEUTRAL"),
        "replay_version": REPLAY_SOURCE,
        "replay_asof": run_date,
    })
    valid_columns = _table_columns(conn)
    columns = sorted(k for k in values.keys() if k in valid_columns)
    placeholders = ", ".join("?" for _ in columns)
    exists = conn.execute(
        "SELECT id FROM signal_backtest WHERE ticker=? AND source=? AND run_date=?",
        (ticker, REPLAY_SOURCE, run_date),
    ).fetchone()
    if exists:
        if refresh_existing:
            update_columns = [c for c in columns if c != "id"]
            if update_columns:
                assignments = ", ".join(f"{c}=?" for c in update_columns)
                conn.execute(
                    f"UPDATE signal_backtest SET {assignments} WHERE id=?",
                    [values[c] for c in update_columns] + [exists[0]],
                )
                return "updated"
        return "existing"
    conn.execute(
        f"INSERT INTO signal_backtest ({', '.join(columns)}) VALUES ({placeholders})",
        [values[c] for c in columns],
    )
    return "inserted"


def enrich_existing_replay_rows(
    *,
    tickers: Iterable[str] | None = None,
    limit: int | None = None,
    batch_size: int = 1000,
    only_missing: bool = True,
) -> ReplayEnrichStats:
    """Backfill factor columns on existing PIT replay rows without live data."""
    init_backtest_db()
    attempted = updated = skipped_no_price = errors = 0
    with _connect() as conn:
        valid_columns = _table_columns(conn)
        wanted = [
            "id", "ticker", "run_date", "replay_asof", "signal_price",
            "technical_score", "momentum_score", "return_10d_prior",
            "return_30d_prior", "return_90d_prior", "vol_20d", "pe_ratio",
            "fcf_yield", "ev_ebit", "ev_ebit_score", "gpa", "gpa_score",
            "f_score", "f_score_coverage", "quality_score_fundamental",
            "gross_profitability", "fcf_to_assets", "revenue_growth",
            "profit_margin", "roe", "current_ratio", "net_debt_ebitda",
            "cash_to_debt",
        ]
        select_cols = [c for c in wanted if c in valid_columns]
        where = ["source=?"]
        params: list = [REPLAY_SOURCE]
        ticker_filter = sorted({str(t).upper().strip() for t in (tickers or []) if str(t or "").strip()})
        if ticker_filter:
            where.append(f"ticker IN ({', '.join('?' for _ in ticker_filter)})")
            params.extend(ticker_filter)
        if only_missing:
            where.append(
                "("
                "value_factor_score IS NULL OR momentum_factor_score IS NULL "
                "OR volatility_factor_score IS NULL OR qmj_factor_score IS NULL"
                ")"
            )
        sql = (
            f"SELECT {', '.join(select_cols)} FROM signal_backtest "
            f"WHERE {' AND '.join(where)} ORDER BY run_date, ticker"
        )
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = conn.execute(sql, params).fetchall()
        for start in range(0, len(rows), max(1, int(batch_size))):
            batch = rows[start:start + max(1, int(batch_size))]
            for raw in batch:
                attempted += 1
                row = dict(zip(select_cols, raw))
                row_id = row.get("id")
                ticker = str(row.get("ticker") or "").upper().strip()
                as_of_raw = row.get("replay_asof") or row.get("run_date")
                as_of_dt = _coerce_date(as_of_raw)
                signal_price = _finite(row.get("signal_price"))
                if not ticker or as_of_dt is None or signal_price is None or signal_price <= 0:
                    skipped_no_price += 1
                    continue
                try:
                    payload = _replay_factor_payload(ticker, pd.Timestamp(as_of_dt), row)
                    updates = {
                        k: v for k, v in payload.items()
                        if k in valid_columns and k not in {"id", "ticker", "run_date", "source"}
                    }
                    if not updates:
                        continue
                    assignments = ", ".join(f"{k}=?" for k in sorted(updates))
                    values = [updates[k] for k in sorted(updates)]
                    values.append(row_id)
                    conn.execute(
                        f"UPDATE signal_backtest SET {assignments} WHERE id=?",
                        values,
                    )
                    updated += 1
                except Exception as exc:
                    errors += 1
                    logger.debug("Replay enrichment update failed for id=%s ticker=%s: %s", row_id, ticker, exc)
            conn.commit()
            logger.info(
                "Replay enrichment batch %d-%d/%d (updated=%d errors=%d)",
                start + 1,
                min(start + len(batch), len(rows)),
                len(rows),
                updated,
                errors,
            )
    return ReplayEnrichStats(
        attempted=attempted,
        updated=updated,
        skipped_no_price=skipped_no_price,
        errors=errors,
    )


def run_replay(
    *,
    start: str,
    end: str,
    tickers: Iterable[str] | None = None,
    candidate_metadata_by_ticker: Mapping[str, Mapping] | None = None,
    max_tickers: int | None = None,
    batch_size: int | None = None,
    frequency: str = "monthly",
    include_forward_labels: bool = True,
    refresh_existing: bool = False,
) -> ReplayStats:
    init_backtest_db()
    dates = replay_dates(start, end, frequency)
    requested_universe = {
        resolve_yahoo_ticker(str(t).upper().strip())
        for t in (tickers or replay_universe(max_tickers=max_tickers))
        if str(t or "").strip()
    }
    universe = sorted(t for t in requested_universe if not is_excluded_ticker(t))
    if max_tickers:
        universe = universe[: int(max_tickers)]
    if not dates or not universe:
        return ReplayStats()

    price_start = dates[0] - pd.Timedelta(days=420)
    forward_buffer = 120 if include_forward_labels else 3
    price_end = dates[-1] + pd.Timedelta(days=forward_buffer)
    today_cap = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    if price_end > today_cap:
        price_end = today_cap
    frames = download_price_history(
        universe,
        start=price_start,
        end=price_end,
        batch_size=batch_size or getattr(config, "PRICE_CACHE_BATCH_SIZE", 75),
    )

    stats = ReplayStats()
    attempted = inserted = updated = skipped_existing = skipped_no_price = 0
    for as_of in dates:
        price_features_by_ticker: dict[str, dict] = {}
        for ticker in universe:
            frame = frames.get(ticker)
            pfeat = _price_features(frame, as_of)
            if pfeat:
                price_features_by_ticker[ticker] = pfeat
        if bool(getattr(config, "HISTORICAL_REPLAY_CROSS_SECTIONAL_MOMENTUM", False)):
            _apply_cross_sectional_momentum_scores(price_features_by_ticker)
        replay_rows: dict[str, dict] = {}
        frames_for_rows: dict[str, pd.DataFrame] = {}
        for ticker in universe:
            attempted += 1
            frame = frames.get(ticker)
            pfeat = price_features_by_ticker.get(ticker)
            if not pfeat:
                skipped_no_price += 1
                continue
            metadata = None
            if candidate_metadata_by_ticker:
                metadata = candidate_metadata_by_ticker.get(ticker) or candidate_metadata_by_ticker.get(ticker.upper())
            enriched = _replay_factor_payload(ticker, as_of, pfeat)
            execution = _replay_execution_fields(
                ticker=ticker,
                frame=frame,
                as_of=as_of,
                price_features=pfeat,
                candidate_metadata=metadata,
            )
            fund_score = enriched.get("fundamental_score")
            tech_score = pfeat.get("technical_score") or 0.0
            aggregate = 0.65 * tech_score + 0.35 * (fund_score if fund_score is not None else 0.0)
            action = (
                "STRONG BUY" if aggregate >= float(getattr(config, "SCORE_STRONG_BUY_THRESHOLD", 0.40))
                else "BUY" if aggregate >= float(getattr(config, "SCORE_BUY_THRESHOLD", 0.20))
                else "NEUTRAL" if aggregate >= float(getattr(config, "SCORE_KEEP_THRESHOLD", -0.25))
                else "AVOID"
            )
            row = {
                **pfeat,
                **enriched,
                "aggregate_score": round(float(aggregate), 4),
                "final_rank": round(float(aggregate), 4),
                "action": action,
                "sentiment_score": None,
                "forecast_score": None,
                **execution,
            }
            replay_rows[ticker] = row
            if frame is not None:
                frames_for_rows[ticker] = frame

        _apply_replay_readiness_fields(replay_rows)

        with _connect() as conn:
            for ticker, row in replay_rows.items():
                frame = frames_for_rows.get(ticker)
                if include_forward_labels and frame is not None:
                    row.update(_forward_labels(frame, as_of, row["signal_price"], row.get("atr")))
                status = _insert_replay_row(
                    ticker,
                    as_of,
                    row,
                    conn=conn,
                    refresh_existing=refresh_existing,
                )
                if status == "inserted":
                    inserted += 1
                elif status == "updated":
                    updated += 1
                elif status == "existing":
                    skipped_existing += 1
                elif status == "no_price":
                    skipped_no_price += 1
        logger.info(
            "Replay %s complete (attempted=%d inserted=%d updated=%d existing=%d no_price=%d)",
            as_of.date().isoformat(),
            attempted,
            inserted,
            updated,
            skipped_existing,
            skipped_no_price,
        )
    return ReplayStats(
        attempted=attempted,
        inserted=inserted,
        updated=updated,
        skipped_existing=skipped_existing,
        skipped_no_price=skipped_no_price,
    )


def _parse_tickers(value: str) -> list[str]:
    return [part.strip().upper() for part in value.split(",") if part.strip()]


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run PIT-safe synthetic replay")
    parser.add_argument(
        "--enrich-existing",
        action="store_true",
        help="Update factor columns on existing replay_pit_v1 rows instead of inserting new replay rows.",
    )
    parser.add_argument("--start", default="2022-01-31")
    parser.add_argument("--end", default=datetime.now().date().isoformat())
    parser.add_argument(
        "--frequency",
        choices=["monthly", "weekly", "daily", "latest"],
        default="monthly",
        help="Replay cadence. Use latest for a same-as-of parity refresh.",
    )
    parser.add_argument("--tickers", default="", help="Comma-separated tickers. Default uses signal DB + PIT store.")
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Maximum existing replay rows to enrich.")
    parser.add_argument("--batch-size", type=int, default=getattr(config, "PRICE_CACHE_BATCH_SIZE", 75))
    parser.add_argument(
        "--no-forward-labels",
        action="store_true",
        help="Skip forward labels for fresh parity rows that do not have mature outcomes yet.",
    )
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Update existing replay rows for the same ticker/source/run_date instead of skipping them.",
    )
    parser.add_argument(
        "--all-existing",
        action="store_true",
        help="When enriching existing rows, recompute all replay rows instead of only missing factor columns.",
    )
    args = parser.parse_args()

    if args.enrich_existing:
        tickers = _parse_tickers(args.tickers) or None
        stats = enrich_existing_replay_rows(
            tickers=tickers,
            limit=args.limit,
            batch_size=args.batch_size,
            only_missing=not args.all_existing,
        )
        print(
            "replay_enrich_attempted={attempted} updated={updated} "
            "skipped_no_price={skipped_no_price} errors={errors}".format(**stats.__dict__)
        )
        return

    tickers = _parse_tickers(args.tickers) or None
    stats = run_replay(
        start=args.start,
        end=args.end,
        tickers=tickers,
        max_tickers=args.max_tickers,
        batch_size=args.batch_size,
        frequency=args.frequency,
        include_forward_labels=not args.no_forward_labels,
        refresh_existing=args.refresh_existing,
    )
    print(
        "replay_attempted={attempted} inserted={inserted} updated={updated} "
        "skipped_existing={skipped_existing} skipped_no_price={skipped_no_price}".format(**stats.__dict__)
    )


if __name__ == "__main__":
    _main()
