"""Paper Trading Ledger — SQLite-backed simulation of trade execution.

Logs every BUY/SELL/SWAP signal with the recommended price at signal time,
then resolves fills using the next-session open price to measure slippage.

Tables:
    paper_signals   — raw signal log (one row per recommendation)
    paper_fills     — simulated execution (one row per resolved signal)
    paper_positions — current paper portfolio state (running view)
    paper_pnl       — realized P&L per closed position
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf

import config

logger = logging.getLogger("paper_trading")

DB_PATH = Path(__file__).parent.parent / getattr(config, "PAPER_TRADING_DB", "paper_trading.db")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    ticker          TEXT    NOT NULL,
    side            TEXT    NOT NULL,            -- BUY | SELL
    source          TEXT    NOT NULL,            -- portfolio_alert | discovery_swap | manual
    signal_price    REAL    NOT NULL,            -- price at signal time
    quantity        REAL,                        -- suggested or actual holding qty
    score           REAL,                        -- aggregate_score at signal time
    action          TEXT,                        -- STRONG BUY / BUY / SELL / STRONG SELL
    swap_from       TEXT,                        -- ticker being replaced (for swaps)
    metadata        TEXT,                        -- JSON blob for extra context
    resolved        INTEGER NOT NULL DEFAULT 0   -- 0=pending, 1=filled
);

CREATE TABLE IF NOT EXISTS paper_fills (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id       INTEGER NOT NULL REFERENCES paper_signals(id),
    fill_timestamp  TEXT    NOT NULL,
    fill_price      REAL    NOT NULL,            -- next-session open (simulated fill)
    signal_price    REAL    NOT NULL,            -- copied from signal for easy queries
    slippage_bps    REAL    NOT NULL,            -- (fill - signal) / signal * 10000
    ticker          TEXT    NOT NULL,
    side            TEXT    NOT NULL,
    quantity        REAL
);

CREATE TABLE IF NOT EXISTS paper_positions (
    ticker          TEXT    PRIMARY KEY,
    side            TEXT    NOT NULL DEFAULT 'LONG',
    quantity        REAL    NOT NULL DEFAULT 0,
    avg_entry_price REAL    NOT NULL DEFAULT 0,
    opened_at       TEXT    NOT NULL,
    last_updated    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_pnl (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    entry_price     REAL    NOT NULL,
    exit_price      REAL    NOT NULL,
    quantity        REAL    NOT NULL,
    pnl             REAL    NOT NULL,            -- absolute P&L
    pnl_pct         REAL    NOT NULL,            -- % return
    hold_days       INTEGER,
    opened_at       TEXT,
    closed_at       TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_signals_ticker ON paper_signals(ticker);
CREATE INDEX IF NOT EXISTS idx_signals_resolved ON paper_signals(resolved);
CREATE INDEX IF NOT EXISTS idx_fills_ticker ON paper_fills(ticker);
CREATE INDEX IF NOT EXISTS idx_pnl_ticker ON paper_pnl(ticker);
"""


# ---------------------------------------------------------------------------
# DB connection — WAL mode for OneDrive safety
# ---------------------------------------------------------------------------

def _get_db_path() -> Path:
    return DB_PATH


@contextmanager
def _connect():
    """Context manager for SQLite connection with OneDrive-safe settings."""
    db = _get_db_path()
    conn = sqlite3.connect(str(db), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")       # Write-ahead log — crash-safe
    conn.execute("PRAGMA synchronous=NORMAL")      # Good balance of safety + speed
    conn.execute("PRAGMA busy_timeout=5000")        # Wait up to 5s if locked
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist."""
    with _connect() as conn:
        conn.executescript(_SCHEMA)
    logger.info("Paper trading DB initialized at %s", _get_db_path())


# ---------------------------------------------------------------------------
# Signal logging — called by the orchestrator when signals fire
# ---------------------------------------------------------------------------

def log_signal(
    ticker: str,
    side: str,
    source: str,
    signal_price: float,
    quantity: float | None = None,
    score: float | None = None,
    action: str | None = None,
    swap_from: str | None = None,
    metadata: str | None = None,
) -> int:
    """Record a new paper trade signal. Returns the signal ID."""
    import json as _json

    now = datetime.now().isoformat(timespec="seconds")
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO paper_signals
               (timestamp, ticker, side, source, signal_price, quantity, score, action, swap_from, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now, ticker, side, source, signal_price, quantity, score, action, swap_from, metadata),
        )
        signal_id = cur.lastrowid
    logger.info("Paper signal #%d: %s %s @ %.4f (source=%s, score=%.3f)",
                signal_id, side, ticker, signal_price, source, score or 0)
    return signal_id


# ---------------------------------------------------------------------------
# Fill resolution — fetch next-session open and calculate slippage
# ---------------------------------------------------------------------------

def _get_next_open(ticker: str, signal_date: str) -> float | None:
    """Fetch the opening price on the first trading session after signal_date."""
    try:
        sig_dt = datetime.fromisoformat(signal_date)
        # Look forward up to 5 calendar days to find next trading day
        start = (sig_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        end = (sig_dt + timedelta(days=6)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            open_price = float(df["Open"].iloc[0])
            return open_price
    except Exception as e:
        logger.warning("Failed to fetch next open for %s: %s", ticker, e)
    return None


def resolve_pending_signals():
    """Resolve all unresolved signals by fetching actual fill prices.

    Should be called on the next orchestrator run (T+1) so that
    market open data is available for the signal date.
    """
    init_db()
    with _connect() as conn:
        pending = conn.execute(
            "SELECT * FROM paper_signals WHERE resolved = 0"
        ).fetchall()

    if not pending:
        logger.info("No pending paper signals to resolve.")
        return 0

    resolved_count = 0
    for sig in pending:
        fill_price = _get_next_open(sig["ticker"], sig["timestamp"])
        if fill_price is None:
            # Market data not yet available — try again next run
            logger.debug("Fill price not yet available for signal #%d (%s)", sig["id"], sig["ticker"])
            continue

        signal_price = sig["signal_price"]
        if sig["side"] == "BUY":
            slippage_bps = (fill_price - signal_price) / signal_price * 10000
        else:  # SELL
            slippage_bps = (signal_price - fill_price) / signal_price * 10000

        now = datetime.now().isoformat(timespec="seconds")

        with _connect() as conn:
            # Insert fill record
            conn.execute(
                """INSERT INTO paper_fills
                   (signal_id, fill_timestamp, fill_price, signal_price, slippage_bps, ticker, side, quantity)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (sig["id"], now, fill_price, signal_price, round(slippage_bps, 2),
                 sig["ticker"], sig["side"], sig["quantity"]),
            )
            # Mark signal as resolved
            conn.execute("UPDATE paper_signals SET resolved = 1 WHERE id = ?", (sig["id"],))

            # Update paper positions
            _update_position(conn, sig["ticker"], sig["side"], fill_price, sig["quantity"] or 0)

        logger.info(
            "Filled signal #%d: %s %s @ %.4f (signal: %.4f, slippage: %+.1f bps)",
            sig["id"], sig["side"], sig["ticker"], fill_price, signal_price, slippage_bps,
        )
        resolved_count += 1

    logger.info("Resolved %d/%d pending paper signals.", resolved_count, len(pending))
    return resolved_count


def _update_position(conn: sqlite3.Connection, ticker: str, side: str, fill_price: float, quantity: float):
    """Update paper_positions and paper_pnl after a fill."""
    now = datetime.now().isoformat(timespec="seconds")
    existing = conn.execute("SELECT * FROM paper_positions WHERE ticker = ?", (ticker,)).fetchone()

    if side == "BUY":
        if existing:
            # Average up
            old_qty = existing["quantity"]
            old_avg = existing["avg_entry_price"]
            new_qty = old_qty + quantity
            new_avg = ((old_avg * old_qty) + (fill_price * quantity)) / new_qty if new_qty > 0 else fill_price
            conn.execute(
                "UPDATE paper_positions SET quantity=?, avg_entry_price=?, last_updated=? WHERE ticker=?",
                (new_qty, round(new_avg, 6), now, ticker),
            )
        else:
            conn.execute(
                "INSERT INTO paper_positions (ticker, side, quantity, avg_entry_price, opened_at, last_updated) VALUES (?, 'LONG', ?, ?, ?, ?)",
                (ticker, quantity, fill_price, now, now),
            )

    elif side == "SELL":
        if existing and existing["quantity"] > 0:
            entry_price = existing["avg_entry_price"]
            sell_qty = min(quantity, existing["quantity"]) if quantity > 0 else existing["quantity"]
            pnl = (fill_price - entry_price) * sell_qty
            pnl_pct = (fill_price - entry_price) / entry_price * 100 if entry_price > 0 else 0

            # Calculate hold duration
            hold_days = None
            if existing["opened_at"]:
                try:
                    opened = datetime.fromisoformat(existing["opened_at"])
                    hold_days = (datetime.now() - opened).days
                except ValueError:
                    pass

            # Record realized P&L
            conn.execute(
                """INSERT INTO paper_pnl
                   (ticker, entry_price, exit_price, quantity, pnl, pnl_pct, hold_days, opened_at, closed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ticker, entry_price, fill_price, sell_qty, round(pnl, 2), round(pnl_pct, 2),
                 hold_days, existing["opened_at"], now),
            )

            remaining = existing["quantity"] - sell_qty
            if remaining <= 0:
                conn.execute("DELETE FROM paper_positions WHERE ticker = ?", (ticker,))
            else:
                conn.execute(
                    "UPDATE paper_positions SET quantity=?, last_updated=? WHERE ticker=?",
                    (remaining, now, ticker),
                )


# ---------------------------------------------------------------------------
# Query helpers — used by the Streamlit UI
# ---------------------------------------------------------------------------

def get_all_signals(limit: int = 100) -> list[dict]:
    """Return recent signals with fill data joined."""
    init_db()
    with _connect() as conn:
        rows = conn.execute("""
            SELECT s.*, f.fill_price, f.fill_timestamp, f.slippage_bps
            FROM paper_signals s
            LEFT JOIN paper_fills f ON f.signal_id = s.id
            ORDER BY s.timestamp DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_open_positions() -> list[dict]:
    """Return all open paper positions."""
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM paper_positions ORDER BY ticker").fetchall()
    return [dict(r) for r in rows]


def get_realized_pnl() -> list[dict]:
    """Return all realized P&L records."""
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM paper_pnl ORDER BY closed_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_slippage_stats() -> dict:
    """Aggregate slippage statistics across all fills."""
    init_db()
    with _connect() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total_fills,
                AVG(slippage_bps) as avg_slippage_bps,
                MIN(slippage_bps) as min_slippage_bps,
                MAX(slippage_bps) as max_slippage_bps,
                AVG(ABS(slippage_bps)) as avg_abs_slippage_bps,
                SUM(CASE WHEN side='BUY' THEN 1 ELSE 0 END) as buy_fills,
                SUM(CASE WHEN side='SELL' THEN 1 ELSE 0 END) as sell_fills,
                AVG(CASE WHEN side='BUY' THEN slippage_bps END) as avg_buy_slippage,
                AVG(CASE WHEN side='SELL' THEN slippage_bps END) as avg_sell_slippage
            FROM paper_fills
        """).fetchone()
    return dict(row) if row else {}


def get_pnl_summary() -> dict:
    """Aggregate P&L statistics across all closed trades."""
    init_db()
    with _connect() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(pnl) as total_pnl,
                AVG(pnl_pct) as avg_return_pct,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winners,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losers,
                AVG(CASE WHEN pnl > 0 THEN pnl_pct END) as avg_win_pct,
                AVG(CASE WHEN pnl <= 0 THEN pnl_pct END) as avg_loss_pct,
                MAX(pnl_pct) as best_trade_pct,
                MIN(pnl_pct) as worst_trade_pct,
                AVG(hold_days) as avg_hold_days
            FROM paper_pnl
        """).fetchone()
    return dict(row) if row else {}


def get_slippage_by_ticker() -> list[dict]:
    """Slippage breakdown per ticker."""
    init_db()
    with _connect() as conn:
        rows = conn.execute("""
            SELECT
                ticker,
                COUNT(*) as fills,
                AVG(slippage_bps) as avg_slippage_bps,
                AVG(ABS(slippage_bps)) as avg_abs_slippage_bps,
                MIN(slippage_bps) as min_slippage_bps,
                MAX(slippage_bps) as max_slippage_bps
            FROM paper_fills
            GROUP BY ticker
            ORDER BY avg_abs_slippage_bps DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_unrealized_pnl() -> list[dict]:
    """Calculate unrealized P&L for all open positions using current prices."""
    positions = get_open_positions()
    if not positions:
        return []

    from utils.data_fetch import get_current_price

    results = []
    for pos in positions:
        current = get_current_price(pos["ticker"])
        if current and current > 0:
            pnl = (current - pos["avg_entry_price"]) * pos["quantity"]
            pnl_pct = (current - pos["avg_entry_price"]) / pos["avg_entry_price"] * 100
            results.append({
                **pos,
                "current_price": current,
                "unrealized_pnl": round(pnl, 2),
                "unrealized_pnl_pct": round(pnl_pct, 2),
            })
    return results
