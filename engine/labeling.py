"""Trade-aware labels for discovery signals.

The old learner used a plain forward return label. Triple-barrier labels
(Lopez de Prado, 2018) are more faithful to the app's recommendations because
they ask which happened first: target, stop, or the time limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

import pandas as pd

import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TripleBarrierResult:
    label: int
    hit: str
    days: int
    return_pct: float
    target_price: float | None
    stop_price: float | None


def _coerce_date(value) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(str(value)[:10]).tz_localize(None).normalize()
    except Exception:
        return None


def _pick_col(frame: pd.DataFrame, name: str):
    if name in frame.columns:
        col = frame[name]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        return col
    return None


def _estimate_atr_pct(frame: pd.DataFrame, entry_price: float, run_ts: pd.Timestamp) -> float:
    highs = _pick_col(frame, "High")
    lows = _pick_col(frame, "Low")
    closes = _pick_col(frame, "Close")
    if highs is None or lows is None or closes is None:
        return 0.08
    pre = frame.loc[frame.index <= run_ts].tail(20)
    if len(pre) < 5:
        pre = frame.tail(20)
    try:
        h = pre["High"].astype(float)
        l = pre["Low"].astype(float)
        c = pre["Close"].astype(float)
        tr = (h - l).abs() / c.replace(0, pd.NA)
        val = float(tr.dropna().mean())
        if val == val and val > 0:
            return max(0.03, min(0.20, val))
    except Exception:
        pass
    return 0.08


def triple_barrier_from_frame(
    frame: pd.DataFrame,
    run_date,
    entry_price: float,
    *,
    atr: float | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    horizon_days: int | None = None,
    target_atr_mult: float | None = None,
    stop_atr_mult: float | None = None,
) -> TripleBarrierResult | None:
    """Return a triple-barrier event label for one signal.

    ``label`` is +1 when the target hits first, -1 when the stop hits first,
    and otherwise the sign of the vertical-barrier return.
    """
    if frame is None or frame.empty or not entry_price:
        return None
    frame = frame.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame[~frame.index.isna()].sort_index()
    frame.index = frame.index.tz_localize(None).normalize()
    run_ts = _coerce_date(run_date)
    if run_ts is None:
        return None

    horizon_days = int(horizon_days or getattr(config, "TRIPLE_BARRIER_HORIZON_DAYS", 30))
    target_atr_mult = float(target_atr_mult or getattr(config, "TRIPLE_BARRIER_TARGET_ATR_MULT", 2.0))
    stop_atr_mult = float(stop_atr_mult or getattr(config, "TRIPLE_BARRIER_STOP_ATR_MULT", 1.0))

    atr_value = None
    try:
        atr_value = float(atr) if atr is not None else None
    except (TypeError, ValueError):
        atr_value = None
    if atr_value is None or atr_value <= 0:
        atr_value = entry_price * _estimate_atr_pct(frame, entry_price, run_ts)

    stop = float(stop_loss) if stop_loss else entry_price - stop_atr_mult * atr_value
    target = float(take_profit) if take_profit else entry_price + target_atr_mult * atr_value
    if stop >= entry_price:
        stop = entry_price - stop_atr_mult * atr_value
    if target <= entry_price:
        target = entry_price + target_atr_mult * atr_value

    end_ts = run_ts + pd.Timedelta(days=horizon_days + 7)
    path = frame.loc[(frame.index > run_ts) & (frame.index <= end_ts)]
    if path.empty:
        path = frame.loc[(frame.index >= run_ts) & (frame.index <= end_ts)]
    if path.empty:
        return None

    highs = _pick_col(path, "High")
    lows = _pick_col(path, "Low")
    closes = _pick_col(path, "Close")
    if highs is None or lows is None or closes is None:
        return None

    last_close = float(closes.dropna().iloc[-1]) if not closes.dropna().empty else float(entry_price)
    for i, idx in enumerate(path.index, start=1):
        try:
            high = float(highs.loc[idx])
            low = float(lows.loc[idx])
        except Exception:
            continue
        # With daily OHLC we do not know intraday ordering. If both barriers
        # hit on the same bar, use the conservative stop-first label.
        if low <= stop:
            return TripleBarrierResult(
                label=-1,
                hit="stop",
                days=i,
                return_pct=(stop - entry_price) / entry_price * 100.0,
                target_price=target,
                stop_price=stop,
            )
        if high >= target:
            return TripleBarrierResult(
                label=1,
                hit="target",
                days=i,
                return_pct=(target - entry_price) / entry_price * 100.0,
                target_price=target,
                stop_price=stop,
            )

    ret = (last_close - entry_price) / entry_price * 100.0
    return TripleBarrierResult(
        label=1 if ret > 0 else (-1 if ret < 0 else 0),
        hit="vertical",
        days=min(len(path), horizon_days),
        return_pct=ret,
        target_price=target,
        stop_price=stop,
    )


def update_triple_barrier_labels(
    *,
    horizon_days: int | None = None,
    batch_size: int | None = None,
    limit: int | None = None,
    force: bool = False,
) -> int:
    """Batch-label signal_backtest rows using the cached price store."""
    from engine.discovery_backtest import _connect, init_backtest_db
    from utils.price_store import download_price_history

    init_backtest_db()
    horizon_days = int(horizon_days or getattr(config, "TRIPLE_BARRIER_HORIZON_DAYS", 30))
    cutoff = (datetime.now() - timedelta(days=horizon_days)).isoformat()
    where = "run_date < ? AND signal_price IS NOT NULL"
    params: list = [cutoff]
    if not force:
        where += " AND tb_label IS NULL"
    sql = f"SELECT * FROM signal_backtest WHERE {where} ORDER BY run_date"
    if limit:
        sql += f" LIMIT {int(limit)}"

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    if not rows:
        return 0

    tickers = sorted({str(r["ticker"]).upper() for r in rows if r["ticker"]})
    min_date = min(pd.Timestamp(str(r["run_date"])[:10]) for r in rows) - pd.Timedelta(days=40)
    max_date = max(pd.Timestamp(str(r["run_date"])[:10]) for r in rows) + pd.Timedelta(days=horizon_days + 10)
    frames = download_price_history(
        tickers,
        start=min_date,
        end=max_date,
        batch_size=batch_size,
    )

    updated = 0
    with _connect() as conn:
        for row in rows:
            ticker = str(row["ticker"]).upper()
            result = triple_barrier_from_frame(
                frames.get(ticker),
                row["run_date"],
                float(row["signal_price"]),
                atr=row["atr"] if "atr" in row.keys() else None,
                stop_loss=row["stop_loss"] if "stop_loss" in row.keys() else None,
                take_profit=row["take_profit"] if "take_profit" in row.keys() else None,
                horizon_days=horizon_days,
            )
            if result is None:
                continue
            conn.execute(
                """UPDATE signal_backtest
                   SET tb_label=?, tb_return=?, tb_days=?, tb_hit=?,
                       tb_horizon=?, tb_updated_at=?
                   WHERE id=?""",
                (
                    result.label,
                    round(result.return_pct, 4),
                    result.days,
                    result.hit,
                    horizon_days,
                    datetime.now().isoformat(),
                    row["id"],
                ),
            )
            updated += 1
    logger.info("Triple-barrier labels updated: %d", updated)
    return updated


if __name__ == "__main__":
    print(update_triple_barrier_labels())

