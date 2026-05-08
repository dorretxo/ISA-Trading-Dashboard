"""Per-ticker score-history reader for the exit-action smoother.

Pure read helpers around the existing ``signal_backtest`` table — no schema
changes, no parallel store.  The smoother needs three things from history:

1. The previous smoothed action for this ticker (for hysteresis).
2. Realised volatility of ``aggregate_score`` over a lookback window.
3. A persistence count: how many of the last N days had a downside action.

The same table powers ``engine.exit_engine.get_signal_decay`` (which already
queries 60 days back), so this module is additive.

References
----------
- Constantinides (1986) "Capital Market Equilibrium with Transaction Costs"
- Davis & Norman (1990) "Portfolio Selection with Transaction Costs"
- Wald (1947) "Sequential Analysis"
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass

from engine.paper_trading import _connect

logger = logging.getLogger(__name__)

DOWN_ACTIONS: frozenset[str] = frozenset({"SELL", "STRONG SELL"})
STRONG_DOWN_ACTIONS: frozenset[str] = frozenset({"STRONG SELL"})


@dataclass(frozen=True)
class ScoreHistoryRow:
    """One historical row for a ticker."""
    run_date: str
    aggregate_score: float | None
    action: str | None
    smoothed_action: str | None


def get_score_series(
    ticker: str,
    n_days: int = 60,
    source: str = "portfolio",
) -> list[ScoreHistoryRow]:
    """Return the most recent ``n_days`` rows for ``ticker`` in chronological order.

    Order: oldest → newest.  Returns an empty list if no rows exist or on error.
    The query is indexed by ``idx_sb_ticker``; cost is O(n_days) per call.
    """
    if not ticker or n_days <= 0:
        return []

    try:
        with _connect() as conn:
            # Tolerate older DBs that lack ``smoothed_action`` — check column existence.
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(signal_backtest)")}
            smoothed_col = "smoothed_action" if "smoothed_action" in cols else "NULL AS smoothed_action"

            rows = conn.execute(
                f"""SELECT run_date, aggregate_score, action, {smoothed_col}
                    FROM signal_backtest
                    WHERE ticker = ? AND source = ?
                    ORDER BY run_date DESC
                    LIMIT ?""",
                (ticker, source, n_days),
            ).fetchall()
    except Exception as exc:  # pragma: no cover — DB hiccup shouldn't crash scoring
        logger.debug("score_history: read failed for %s: %s", ticker, exc)
        return []

    out = [
        ScoreHistoryRow(
            run_date=r["run_date"],
            aggregate_score=r["aggregate_score"],
            action=r["action"],
            smoothed_action=r["smoothed_action"],
        )
        for r in rows
    ]
    out.reverse()  # chronological order
    return out


def compute_score_vol(
    series: list[ScoreHistoryRow] | None,
    lookback: int = 30,
    min_obs: int = 10,
) -> float | None:
    """Stdev of ``aggregate_score`` over the last ``lookback`` rows.

    Returns ``None`` when fewer than ``min_obs`` finite scores are available;
    the smoother falls back to ``EXIT_SMOOTHER_VOL_FALLBACK`` in that case.
    """
    if not series:
        return None
    tail = series[-lookback:] if lookback > 0 else series
    scores = [r.aggregate_score for r in tail if r.aggregate_score is not None]
    if len(scores) < min_obs:
        return None
    try:
        return float(statistics.pstdev(scores))
    except statistics.StatisticsError:
        return None


def count_recent_action(
    series: list[ScoreHistoryRow] | None,
    target_actions: frozenset[str] | set[str],
    n: int = 5,
) -> tuple[int, str | None]:
    """Count how many of the last ``n`` rows had ``action`` in ``target_actions``.

    Returns ``(count, since_date)`` where ``since_date`` is the ``run_date`` of
    the **earliest** qualifying row in the window — used for the UI "SELL since
    YYYY-MM-DD" chip.  Returns ``(0, None)`` for empty/missing input.

    The base ``action`` column is used (not ``smoothed_action``) so the gate
    measures the *raw* signal pressure, not its own smoothed output.
    """
    if not series or n <= 0:
        return 0, None
    window = series[-n:]
    qualifying = [r for r in window if r.action in target_actions]
    if not qualifying:
        return 0, None
    since_date = qualifying[0].run_date
    if since_date and len(since_date) >= 10:
        since_date = since_date[:10]
    return len(qualifying), since_date


def get_prev_smoothed_action(
    series: list[ScoreHistoryRow] | None,
    fallback_to_action: bool = True,
) -> str | None:
    """Return the most recent persisted ``smoothed_action``.

    Falls back to the most recent base ``action`` when the smoothed column is
    missing on the latest row (cold-start during deployment).  Returns ``None``
    when there is no history at all (true cold-start ticker).
    """
    if not series:
        return None
    latest = series[-1]
    if latest.smoothed_action:
        return latest.smoothed_action
    if fallback_to_action and latest.action:
        return latest.action
    return None
