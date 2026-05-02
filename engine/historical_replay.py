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
import logging
import math
from typing import Iterable

import numpy as np
import pandas as pd

import config
from engine.discovery_backtest import _connect, init_backtest_db
from engine.enterprise_factors import compute_piotroski_f_score
from engine.labeling import triple_barrier_from_frame
from utils.pit_store import _available_date, _coerce_date, _load_store
from utils.pit_store import all_tickers as pit_tickers
from utils.price_store import download_price_history

logger = logging.getLogger(__name__)

REPLAY_SOURCE = "replay_pit_v1"
_PIT_STORE_CACHE: dict | None = None


def _finite(value) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _pit_store_cached() -> dict:
    global _PIT_STORE_CACHE
    if _PIT_STORE_CACHE is None:
        _PIT_STORE_CACHE = _load_store()
    return _PIT_STORE_CACHE


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


@dataclass(frozen=True)
class ReplayStats:
    attempted: int = 0
    inserted: int = 0
    skipped_existing: int = 0
    skipped_no_price: int = 0


def month_ends(start: str, end: str) -> list[pd.Timestamp]:
    dates = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="ME")
    if dates.empty:
        return []
    return [pd.Timestamp(d).normalize() for d in dates]


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
        return float(close.iloc[-1] / close.iloc[-n - 1] - 1.0) * 100.0

    sma50 = _finite(close.tail(50).mean())
    sma200 = _finite(close.tail(200).mean())
    if sma50 is None or sma200 is None or sma200 <= 0:
        return None
    daily_ret = close.pct_change(fill_method=None)
    vol_20d = _finite(daily_ret.tail(20).std() * np.sqrt(252) * 100.0) or 0.0
    atr = _finite((high.tail(14) - low.tail(14)).mean())
    ret_10 = ret(10)
    ret_30 = ret(30)
    ret_90 = ret(90)
    above_200 = 1.0 if price > sma200 else -1.0
    momentum = np.nanmean([
        0.0 if ret_10 is None else np.clip(ret_10 / 10.0, -1.0, 1.0),
        0.0 if ret_30 is None else np.clip(ret_30 / 20.0, -1.0, 1.0),
        0.0 if ret_90 is None else np.clip(ret_90 / 35.0, -1.0, 1.0),
        above_200,
    ])
    return {
        "signal_price": price,
        "technical_score": float(np.clip(momentum, -1.0, 1.0)),
        "momentum_score": float(np.clip((momentum + 1.0) / 2.0, 0.0, 1.0)),
        "sma_50": sma50,
        "sma_200": sma200,
        "atr": atr,
        "return_10d_prior": ret_10,
        "return_30d_prior": ret_30,
        "return_90d_prior": ret_90,
        "vol_20d": vol_20d,
        "price_vs_sma200_stretch": float(price / sma200 - 1.0) if sma200 else None,
    }


def _fundamental_features(ticker: str, as_of: pd.Timestamp) -> dict:
    latest, report_date = _latest_as_of_cached(
        ticker,
        as_of.date(),
        lag_days=int(getattr(config, "PIT_FUNDAMENTAL_LAG_DAYS", 45)),
    )
    if not latest or not report_date:
        return {}
    prior, _ = _prior_snapshot_cached(ticker, report_date)
    f_result = compute_piotroski_f_score(latest, prior)
    total_assets = latest.get("total_assets")
    gross_profit = latest.get("gross_profit")
    gpa = None
    gpa_score = None
    try:
        if total_assets and gross_profit is not None:
            gpa = float(gross_profit) / float(total_assets)
            gpa_score = float(np.clip((gpa - 0.30) / 0.20, -1.0, 1.0))
    except Exception:
        pass
    return {
        "f_score": f_result.get("f_score"),
        "f_score_coverage": f_result.get("f_score_coverage"),
        "gpa": gpa,
        "gpa_score": gpa_score,
        "gross_profitability": gpa,
        "quality_factor_score": gpa_score,
        "qmj_factor_score": gpa_score,
        "fundamental_score": float(np.nanmean([
            f_result.get("f_score_score") if f_result.get("f_score_score") is not None else np.nan,
            gpa_score if gpa_score is not None else np.nan,
        ])) if (f_result.get("f_score_score") is not None or gpa_score is not None) else None,
    }


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


def _insert_replay_row(ticker: str, as_of: pd.Timestamp, values: dict, conn=None) -> str:
    run_date = as_of.date().isoformat()
    values = dict(values)
    signal_price = _finite(values.get("signal_price"))
    if signal_price is None or signal_price <= 0:
        return "no_price"
    values["signal_price"] = signal_price
    if conn is None:
        with _connect() as owned_conn:
            return _insert_replay_row(ticker, as_of, values, conn=owned_conn)
    values.update({
        "run_date": run_date,
        "ticker": ticker,
        "source": REPLAY_SOURCE,
        "name": ticker,
        "action": values.get("action") or ("BUY" if values.get("aggregate_score", 0) >= 0.2 else "NEUTRAL"),
        "replay_version": REPLAY_SOURCE,
        "replay_asof": run_date,
    })
    columns = sorted(values.keys())
    placeholders = ", ".join("?" for _ in columns)
    exists = conn.execute(
        "SELECT id FROM signal_backtest WHERE ticker=? AND source=? AND run_date=?",
        (ticker, REPLAY_SOURCE, run_date),
    ).fetchone()
    if exists:
        return "existing"
    conn.execute(
        f"INSERT INTO signal_backtest ({', '.join(columns)}) VALUES ({placeholders})",
        [values[c] for c in columns],
    )
    return "inserted"


def run_replay(
    *,
    start: str,
    end: str,
    tickers: Iterable[str] | None = None,
    max_tickers: int | None = None,
    batch_size: int | None = None,
) -> ReplayStats:
    init_backtest_db()
    dates = month_ends(start, end)
    universe = sorted({str(t).upper().strip() for t in (tickers or replay_universe(max_tickers=max_tickers)) if str(t or "").strip()})
    if max_tickers:
        universe = universe[: int(max_tickers)]
    if not dates or not universe:
        return ReplayStats()

    price_start = dates[0] - pd.Timedelta(days=420)
    price_end = dates[-1] + pd.Timedelta(days=120)
    frames = download_price_history(
        universe,
        start=price_start,
        end=price_end,
        batch_size=batch_size or getattr(config, "PRICE_CACHE_BATCH_SIZE", 75),
    )

    stats = ReplayStats()
    attempted = inserted = skipped_existing = skipped_no_price = 0
    for as_of in dates:
        with _connect() as conn:
            for ticker in universe:
                attempted += 1
                frame = frames.get(ticker)
                pfeat = _price_features(frame, as_of)
                if not pfeat:
                    skipped_no_price += 1
                    continue
                ffeat = _fundamental_features(ticker, as_of)
                fund_score = ffeat.get("fundamental_score")
                tech_score = pfeat.get("technical_score") or 0.0
                aggregate = 0.65 * tech_score + 0.35 * (fund_score if fund_score is not None else 0.0)
                row = {
                    **pfeat,
                    **ffeat,
                    "aggregate_score": round(float(aggregate), 4),
                    "final_rank": round(float(aggregate), 4),
                    "sentiment_score": 0.0,
                    "forecast_score": 0.0,
                    "take_profit": pfeat["signal_price"] + 2.0 * (pfeat.get("atr") or pfeat["signal_price"] * 0.08),
                    "stop_loss": pfeat["signal_price"] - 1.0 * (pfeat.get("atr") or pfeat["signal_price"] * 0.08),
                }
                row.update(_forward_labels(frame, as_of, pfeat["signal_price"], pfeat.get("atr")))
                status = _insert_replay_row(ticker, as_of, row, conn=conn)
                if status == "inserted":
                    inserted += 1
                elif status == "existing":
                    skipped_existing += 1
                elif status == "no_price":
                    skipped_no_price += 1
        logger.info(
            "Replay month %s complete (attempted=%d inserted=%d existing=%d no_price=%d)",
            as_of.date().isoformat(),
            attempted,
            inserted,
            skipped_existing,
            skipped_no_price,
        )
    return ReplayStats(
        attempted=attempted,
        inserted=inserted,
        skipped_existing=skipped_existing,
        skipped_no_price=skipped_no_price,
    )


def _parse_tickers(value: str) -> list[str]:
    return [part.strip().upper() for part in value.split(",") if part.strip()]


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run PIT-safe synthetic monthly replay")
    parser.add_argument("--start", default="2022-01-31")
    parser.add_argument("--end", default=datetime.now().date().isoformat())
    parser.add_argument("--tickers", default="", help="Comma-separated tickers. Default uses signal DB + PIT store.")
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=getattr(config, "PRICE_CACHE_BATCH_SIZE", 75))
    args = parser.parse_args()

    tickers = _parse_tickers(args.tickers) or None
    stats = run_replay(
        start=args.start,
        end=args.end,
        tickers=tickers,
        max_tickers=args.max_tickers,
        batch_size=args.batch_size,
    )
    print(
        "replay_attempted={attempted} inserted={inserted} "
        "skipped_existing={skipped_existing} skipped_no_price={skipped_no_price}".format(**stats.__dict__)
    )


if __name__ == "__main__":
    _main()
